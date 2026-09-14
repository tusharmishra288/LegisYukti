#!/usr/bin/env python3
"""
Run the golden set through the real pipeline and score it.

Needs the same environment as the app: GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY,
and a populated `indian_legal_library` collection. POSTGRES_URI is optional -
without it an in-memory checkpointer is used, which is what you want for eval
anyway so runs do not pollute real workspaces.

    python -m eval.run_eval --label baseline
    # ...refactor...
    python -m eval.run_eval --label soft-prior
    python -m eval.compare eval/results/baseline.json eval/results/soft-prior.json

Latency comes from src/telemetry.py, so the same run measures quality and speed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def build_graph():
    """Prefer an in-memory checkpointer; fall back to Postgres only if asked."""
    from src.agent import create_graph
    from langgraph.checkpoint.memory import MemorySaver
    return create_graph(MemorySaver())


def run_case(graph, case: dict, thread: str, max_retries: int = 3) -> dict:
    from langchain_core.messages import HumanMessage
    from src.telemetry import start_trace, finish_trace, get_callbacks
    from src.streaming import iter_answer, final_answer_from_messages

    config = {"configurable": {"thread_id": thread}}

    for attempt in range(max_retries):
        trace = start_trace(case["question"])
        stream_config = {**config, "callbacks": get_callbacks()}
        streamed = ""
        error = None
        wall_start = time.perf_counter()
        try:
            # Shared with app.py so the eval measures exactly what the UI shows:
            # deltas accumulated, and a retry's draft replacing the previous one.
            for partial in iter_answer(graph.stream(
                {"messages": [HumanMessage(content=case["question"])]},
                config=stream_config, stream_mode="messages",
            )):
                streamed = partial
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            t = finish_trace(trace)
        wall = time.perf_counter() - wall_start

        # Back off and retry on provider rate limits rather than recording a
        # zeroed row that would poison the averages.
        if error and "RateLimit" in error and attempt < max_retries - 1:
            backoff = 30 * (attempt + 1)
            print(f"    rate limited; sleeping {backoff}s then retrying", flush=True)
            time.sleep(backoff)
            continue
        break

    streamed = streamed.strip()
    state = graph.get_state(config).values
    final = final_answer_from_messages(state.get("messages", []))

    # The checkpoint holds the delivered (post-auditor) answer; fall back to the
    # stream only when the graph left no usable message at all.
    answer = final if final else streamed

    return {
        "answer": answer,
        "streamed_answer": streamed,
        "state_answer": final,
        "context": "\n\n".join(state.get("context", []) or []),
        "eval_score": state.get("evaluation_score"),
        "wall": wall,
        "trace": t.as_dict() if t else None,
        "error": error,
    }


def score_case(case: dict, result: dict) -> dict:
    from eval.metrics import (parse_verified_references, retrieval_score,
                              citation_score, forbidden_citations)

    retrieved = parse_verified_references(result["context"])
    expected_sections = case.get("expected_sections") or {}
    expected = dict(expected_sections)
    for act in case.get("expected_acts", []):
        expected.setdefault(act, [])

    rs = retrieval_score(expected, retrieved) if expected else None
    cs = citation_score(result["answer"], retrieved)
    forbidden = forbidden_citations(result["answer"], case.get("must_not_cite", []))

    trace = result.get("trace") or {}
    spans = trace.get("spans", [])
    node_names = [n["name"] for n in trace.get("nodes", [])]

    return {
        "id": case["id"],
        "category": case["category"],
        "error": result["error"],
        "section_recall": rs.section_recall if rs else None,
        "act_recall": rs.act_recall if rs else None,
        "missing": rs.missing if rs else [],
        "retrieved_acts": sorted(rs.retrieved_acts) if rs else [],
        "citation_precision": cs.precision,   # None when the answer cites nothing
        "citations_total": cs.total,
        "unsupported": cs.unsupported,
        "out_of_corpus": cs.out_of_corpus,
        "forbidden_cited": forbidden,
        "app_eval_score": result["eval_score"],
        # latency + pipeline shape
        "total_s": trace.get("total"),
        "llm_calls": trace.get("llm_call_count"),
        "retrieval_s": trace.get("retrieval_wall"),
        "used_fallback": any(s["name"] == "retrieval:fallback-global" for s in spans),
        "retried": len(node_names) != len(set(node_names)),
        "answer_chars": len(result["answer"]),
        # Raw text is persisted so eval/rescore.py can recompute metrics offline
        # after a scoring change, without re-running the pipeline.
        "raw_answer": result["answer"],
        "raw_context": result["context"],
        "raw_streamed": result.get("streamed_answer", ""),
        "raw_state_answer": result.get("state_answer", ""),
    }


def summarise(rows: list[dict]) -> dict:
    def mean(key, subset=None):
        pool = rows if subset is None else subset
        vals = [r[key] for r in pool if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    ok = [r for r in rows if not r.get("error")]
    graded = [r for r in ok if r["category"] != "chat"]
    return {
        "cases": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "section_recall": mean("section_recall", graded),
        "act_recall": mean("act_recall", graded),
        "citation_precision": mean("citation_precision", graded),
        "forbidden_hits": sum(1 for r in rows if r["forbidden_cited"]),
        "mean_total_s": mean("total_s", ok),
        "mean_llm_calls": mean("llm_calls", ok),
        "mean_retrieval_s": mean("retrieval_s", ok),
        "fallback_rate": round(sum(r["used_fallback"] for r in ok) / len(ok), 4) if ok else None,
        "retry_rate": round(sum(r["retried"] for r in ok) / len(ok), 4) if ok else None,
        "scored_cases": len(ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="name for this run, e.g. 'baseline'")
    ap.add_argument("--set", type=Path, default=PROJECT_ROOT / "eval" / "golden_set.yaml")
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    ap.add_argument("--delay", type=float, default=5.0,
                    help="seconds to pause between cases; raise if you hit 429s")
    args = ap.parse_args()

    cases = yaml.safe_load(args.set.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if not cases:
        sys.exit("no cases selected")

    graph = build_graph()
    rows = []
    for i, case in enumerate(cases, 1):
        # follow-ups must share the thread of the case they depend on
        thread = f"EVAL-{case.get('depends_on') or case['id']}"
        print(f"[{i}/{len(cases)}] {case['id']}", flush=True)
        rows.append(score_case(case, run_case(graph, case, thread)))
        if i < len(cases) and args.delay:
            time.sleep(args.delay)      # stay under provider rate limits

    summary = summarise(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "summary": summary, "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f" {args.label}")
    print("=" * 68)
    for k, v in summary.items():
        print(f"  {k:<22} {v}")
    print(f"\n  written to {out.relative_to(PROJECT_ROOT)}")

    worst = sorted((r for r in rows if r["section_recall"] is not None),
                   key=lambda r: r["section_recall"])[:5]
    if worst:
        print("\n  weakest retrieval:")
        for r in worst:
            print(f"    {r['section_recall']:.2f}  {r['id']:<26} missing: {', '.join(r['missing']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
