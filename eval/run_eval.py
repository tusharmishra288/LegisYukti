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


QUALITY_METRICS = ("section_recall", "act_recall", "citation_precision")
COST_METRICS = ("total_s", "llm_calls", "retrieval_s")
RATE_METRICS = ("used_fallback", "retried")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def aggregate_cases(rows: list[dict]) -> dict[str, dict]:
    """Collapse repeated runs of the same case into a median and a spread.

    This pipeline is nondeterministic - the dispatcher and the multi-query
    expansion write different search text on every run, so retrieval returns
    different Acts - which means a single run of a case is a sample, not a
    measurement. The median resists a single wild run; max-min is the empirical
    noise for that case, and compare.py refuses to call anything a regression
    unless it exceeds that.
    """
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["id"], []).append(row)

    out: dict[str, dict] = {}
    for case_id, runs in by_case.items():
        ok = [r for r in runs if not r.get("error")]
        entry: dict = {"runs": len(runs), "errors": len(runs) - len(ok),
                       "category": runs[0]["category"]}
        for metric in QUALITY_METRICS + COST_METRICS:
            vals = [r[metric] for r in ok if r.get(metric) is not None]
            entry[metric] = {
                "median": _median(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "spread": (max(vals) - min(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        for metric in RATE_METRICS:
            vals = [1.0 if r.get(metric) else 0.0 for r in ok]
            entry[metric] = {"median": (sum(vals) / len(vals)) if vals else None,
                             "spread": (max(vals) - min(vals)) if len(vals) > 1 else 0.0,
                             "n": len(vals)}
        entry["forbidden_any"] = any(r["forbidden_cited"] for r in runs)
        out[case_id] = entry
    return out


def noise_bands(per_case: dict[str, dict]) -> dict[str, float]:
    """Typical within-case spread per metric: the floor for calling a delta real."""
    bands: dict[str, float] = {}
    for metric in QUALITY_METRICS + COST_METRICS + RATE_METRICS:
        spreads = [c[metric]["spread"] for c in per_case.values()
                   if c.get(metric) and c[metric].get("n", 0) > 1]
        bands[metric] = round(_median(spreads) or 0.0, 4)
    return bands


def summarise(rows: list[dict]) -> dict:
    """Headline numbers, computed from per-case medians so one wild run cannot move them."""
    per_case = aggregate_cases(rows)

    def across(metric, only_graded=False):
        vals = [c[metric]["median"] for cid, c in per_case.items()
                if c[metric]["median"] is not None
                and not (only_graded and c["category"] == "chat")]
        return round(sum(vals) / len(vals), 4) if vals else None

    ok_rows = [r for r in rows if not r.get("error")]
    bands = noise_bands(per_case)
    return {
        "cases": len(per_case),
        "repeats": max((c["runs"] for c in per_case.values()), default=0),
        "total_runs": len(rows),
        "errors": sum(1 for r in rows if r["error"]),
        "section_recall": across("section_recall", only_graded=True),
        "act_recall": across("act_recall", only_graded=True),
        "citation_precision": across("citation_precision", only_graded=True),
        "forbidden_hits": sum(1 for c in per_case.values() if c["forbidden_any"]),
        "mean_total_s": across("total_s"),
        "mean_llm_calls": across("llm_calls"),
        "mean_retrieval_s": across("retrieval_s"),
        "fallback_rate": across("used_fallback"),
        "retry_rate": across("retried"),
        "scored_cases": len(ok_rows),
        "noise": bands,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="name for this run, e.g. 'baseline'")
    ap.add_argument("--set", type=Path, default=PROJECT_ROOT / "eval" / "golden_set.yaml")
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per case; >1 lets compare.py separate signal from noise")
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
    total = args.repeats * len(cases)
    done = 0
    # Repeats outer, cases inner: a follow-up case must run after the case it
    # depends on WITHIN the same repeat, and each repeat gets its own threads so
    # checkpoints never leak between them.
    for rep in range(args.repeats):
        if args.repeats > 1:
            print(f"--- repeat {rep + 1}/{args.repeats} ---", flush=True)
        for case in cases:
            thread = f"EVAL-{case.get('depends_on') or case['id']}-r{rep}"
            done += 1
            print(f"[{done}/{total}] {case['id']}", flush=True)
            row = score_case(case, run_case(graph, case, thread))
            row["repeat"] = rep
            rows.append(row)
            if done < total and args.delay:
                time.sleep(args.delay)      # stay under provider rate limits

    summary = summarise(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.label}.json"
    out.write_text(json.dumps({"label": args.label, "summary": summary,
                               "per_case": aggregate_cases(rows), "rows": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 68)
    print(f" {args.label}   ({summary['repeats']} run(s) x {summary['cases']} cases)")
    print("=" * 68)
    for key, value in summary.items():
        if key == "noise":
            continue
        print(f"  {key:<22} {value}")

    print("\n  within-case spread (max-min across repeats) - the noise floor:")
    for metric, band in summary["noise"].items():
        print(f"    {metric:<20} {band}")
    if summary["repeats"] < 2:
        print("    (only 1 repeat - spreads are 0 and compare.py cannot gauge noise;"
              " re-run with --repeats 3)")

    print(f"\n  written to {out.relative_to(PROJECT_ROOT)}")

    per_case = aggregate_cases(rows)
    graded = [(cid, c) for cid, c in per_case.items()
              if c["category"] != "chat" and c["section_recall"]["median"] is not None]
    worst = sorted(graded, key=lambda kv: kv[1]["section_recall"]["median"])[:5]
    if worst:
        print("\n  weakest retrieval (median, with observed range):")
        for cid, c in worst:
            sr = c["section_recall"]
            rng = f"{sr['min']:.2f}-{sr['max']:.2f}" if sr["n"] > 1 else "single run"
            flag = "  UNSTABLE" if sr["spread"] and sr["spread"] >= 0.5 else ""
            print(f"    {sr['median']:.2f}  [{rng}]  {cid}{flag}")

    unstable = [cid for cid, c in per_case.items()
                if c["act_recall"]["spread"] and c["act_recall"]["spread"] >= 0.5]
    if unstable:
        print(f"\n  {len(unstable)} case(s) returned different Acts across repeats: "
              f"{', '.join(unstable)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
