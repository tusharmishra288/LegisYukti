#!/usr/bin/env python3
"""
Show exactly what the app's evaluator says, and what the app makes of it.

The baseline shows correlation(evaluation_score, section_recall) = +0.16 and
eight answers with perfect retrieval scored 0, which drives a 75% retry rate.
That is either the model judging badly or the app parsing badly, and the two
have completely different fixes.

This replays evaluate_response_node against a case's SAVED answer and context
from a results file - one LLM call, no pipeline run - and prints the raw model
output next to the score the app extracts from it. The real node is invoked
(fast_llm is wrapped, not reimplemented), so the prompt cannot drift from the app.

    python -m eval.debug_evaluator court-marriage
    python -m eval.debug_evaluator court-marriage --results eval/results/baseline-final.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id")
    ap.add_argument("--results", type=Path,
                    default=PROJECT_ROOT / "eval" / "results" / "baseline-final.json")
    args = ap.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    row = next((r for r in data["rows"] if r["id"] == args.case_id), None)
    if row is None:
        sys.exit(f"case {args.case_id!r} not in {args.results}")

    answer = (row.get("raw_state_answer") or row.get("raw_answer") or "").strip()
    context = row.get("raw_context") or ""
    if not answer:
        sys.exit(f"case {args.case_id!r} has no saved answer to evaluate")

    import src.agent as agent
    from langchain_core.messages import AIMessage

    # ChatGroq is a pydantic model, so attributes cannot be assigned onto the
    # instance. evaluate_response_node resolves `fast_llm` as a module global, so
    # swap the whole name for a transparent proxy instead - the node still builds
    # the real prompt and calls the real model.
    captured: dict[str, str] = {}

    class _SpyLLM:
        def __init__(self, inner):
            self._inner = inner

        def invoke(self, prompt, *a, **kw):
            captured["prompt"] = prompt if isinstance(prompt, str) else str(prompt)
            reply = self._inner.invoke(prompt, *a, **kw)
            captured["raw"] = getattr(reply, "content", str(reply))
            return reply

        def __getattr__(self, name):          # everything else passes through
            return getattr(self._inner, name)

    original_llm = agent.fast_llm
    agent.fast_llm = _SpyLLM(original_llm)
    try:
        result = agent.evaluate_response_node(
            {"messages": [AIMessage(content=answer)], "context": [context], "intent": "LEGAL"}
        )
    finally:
        agent.fast_llm = original_llm

    print("=" * 74)
    print(f" case: {args.case_id}")
    print(f" recorded in results: eval_score={row['app_eval_score']}  "
          f"section_recall={row['section_recall']}  act_recall={row['act_recall']}")
    print("=" * 74)

    print("\n--- PROMPT SENT (first 600 chars) ---")
    print(captured.get("prompt", "<not captured>")[:600])

    raw = captured.get("raw", "")
    print(f"\n--- RAW MODEL OUTPUT ({len(raw)} chars) ---")
    print(repr(raw[:1200]))

    print("\n--- WHAT THE APP'S PARSER FINDS ---")
    primary = re.search(r"(?:SCORE|Score|score)[:\s]*(\d+)", raw)
    print(f"  primary regex  r'(?:SCORE|Score|score)[:\\s]*(\\d+)'  -> "
          f"{primary.group(1) if primary else 'NO MATCH'}")
    if not primary:
        first_line = raw.split("\n")[0]
        digit = re.search(r"(\d+)", first_line)
        print(f"  fallback: first digit on first line {first_line[:70]!r} -> "
              f"{digit.group(1) if digit else 'none -> defaults to 5'}")

    print(f"\n  node returned score = {result.get('evaluation_score')}")
    print(f"  retry threshold is  < 6  ->  "
          f"{'RETRY (full pipeline runs again)' if result.get('evaluation_score', 10) < 6 else 'no retry'}")

    print("\n--- DIAGNOSIS ---")
    if not primary and result.get("evaluation_score") is not None:
        print("  PARSING failure: the model did not emit 'SCORE: <n>' in the expected shape,")
        print("  so the score came from the digit fallback. Fix the output contract, not the model.")
    elif primary and int(primary.group(1)) < 6:
        print("  JUDGEMENT failure: the model really did return a low score for this answer.")
        print("  Fix the rubric/prompt, or stop gating retries on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
