#!/usr/bin/env python3
"""
Aggregate LegisYukti query traces into a latency profile.

A single trace tells you about one query; optimisation decisions need the
distribution. This reads logs/query_traces.jsonl (written by src/telemetry.py)
and reports where time goes across many queries, plus how often the two
latency-doubling paths fire: the evaluator retry loop and the global retrieval
fallback.

Usage:
    python scripts/analyze_traces.py                 # all traces
    python scripts/analyze_traces.py --last 50       # most recent 50
    python scripts/analyze_traces.py --file path.jsonl

Standard library only, so it runs anywhere the repo does.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_TRACE_FILE = Path(__file__).resolve().parent.parent / "logs" / "query_traces.jsonl"


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency for a 200-line script."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def load(path: Path, last: int | None) -> list[dict]:
    if not path.exists():
        sys.exit(f"No trace file at {path}\nRun some queries with telemetry enabled first.")
    traces = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  (skipped malformed line {line_no})", file=sys.stderr)
    if not traces:
        sys.exit("Trace file is empty.")
    return traces[-last:] if last else traces


def stat_row(label: str, values: list[float], total_p50: float, width: int = 30) -> str:
    p50, p90 = percentile(values, 50), percentile(values, 90)
    share = f"{p50 / total_p50 * 100:5.1f}%" if total_p50 > 0 else "    -"
    return f"  {label:<{width}} p50 {p50:7.2f}s   p90 {p90:7.2f}s   {share} of p50 total   n={len(values)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=DEFAULT_TRACE_FILE)
    ap.add_argument("--last", type=int, default=None, help="only the N most recent traces")
    args = ap.parse_args()

    traces = load(args.file, args.last)
    n = len(traces)
    totals = [t.get("total", 0.0) for t in traces]
    total_p50 = percentile(totals, 50)

    print("=" * 82)
    print(f" LegisYukti latency profile   -   {n} queries from {args.file.name}")
    print("=" * 82)
    print(f"  total          p50 {total_p50:6.2f}s   p90 {percentile(totals, 90):6.2f}s   "
          f"max {max(totals):6.2f}s   min {min(totals):6.2f}s")
    print()

    # ---- nodes ---------------------------------------------------------- #
    # Summed per query first: a node that runs twice (retry loop) should show its
    # full per-query cost, not two separate half-cost samples.
    per_query_node: dict[str, list[float]] = defaultdict(list)
    node_order: dict[str, float] = {}
    for t in traces:
        acc: dict[str, float] = defaultdict(float)
        for node in t.get("nodes", []):
            acc[node["name"]] += node["duration"]
            node_order.setdefault(node["name"], node["order"])
        for name, dur in acc.items():
            per_query_node[name].append(dur)

    if per_query_node:
        print(" NODE COST PER QUERY (summed across re-executions)")
        for name in sorted(per_query_node, key=lambda k: -percentile(per_query_node[k], 50)):
            print(stat_row(name, per_query_node[name], total_p50))
        print()

    # ---- models --------------------------------------------------------- #
    per_query_model: dict[str, list[float]] = defaultdict(list)
    calls_per_query: list[float] = []
    llm_walls: list[float] = []
    out_tokens: dict[str, list[float]] = defaultdict(list)
    for t in traces:
        calls_per_query.append(t.get("llm_call_count", 0))
        llm_walls.append(t.get("llm_wall", 0.0))
        for model, e in (t.get("by_model") or {}).items():
            per_query_model[model].append(e["duration"])
            out_tokens[model].append(e.get("out", 0))

    print(" LLM COST PER QUERY")
    print(f"  calls per query               p50 {percentile(calls_per_query, 50):6.1f}     "
          f"p90 {percentile(calls_per_query, 90):6.1f}     max {max(calls_per_query):.0f}")
    print(stat_row("all LLM wall time", llm_walls, total_p50))
    for model in sorted(per_query_model, key=lambda k: -percentile(per_query_model[k], 50)):
        print(stat_row(f"  {model}", per_query_model[model], total_p50, width=28))
        print(f"      output tokens p50 {percentile(out_tokens[model], 50):.0f}")
    print()

    # ---- retrieval ------------------------------------------------------ #
    retrieval = [t.get("retrieval_wall", 0.0) for t in traces]
    if any(retrieval):
        print(" RETRIEVAL")
        print(stat_row("retrieval wall", retrieval, total_p50))
        span_times: dict[str, list[float]] = defaultdict(list)
        for t in traces:
            for s in t.get("spans", []):
                if s.get("kind") == "retriever":
                    span_times[f"{'  ' * s.get('depth', 0)}{s['name']}"].append(s["duration"])
        for name, vals in sorted(span_times.items(), key=lambda kv: -percentile(kv[1], 50)):
            print(stat_row(name, vals, total_p50))
        print()

    # ---- the two latency doublers --------------------------------------- #
    retried = sum(
        1 for t in traces
        if len({(nd["name"], nd["order"]) for nd in t.get("nodes", [])}) >
           len({nd["name"] for nd in t.get("nodes", [])})
    )
    fallback = sum(
        1 for t in traces
        if any(s["name"] == "retrieval:fallback-global" for s in t.get("spans", []))
    )
    errored = sum(
        1 for t in traces
        if any(c.get("errored") for c in t.get("llm_calls", []))
    )

    print(" PATHOLOGY COUNTERS")
    print(f"  evaluator retry loop fired    {retried:4d} / {n}  ({retried / n * 100:5.1f}%)   re-enters at chat_node: full retrieval again")
    print(f"  global retrieval fallback     {fallback:4d} / {n}  ({fallback / n * 100:5.1f}%)   re-runs multi-query + rerank")
    print(f"  queries with an LLM error     {errored:4d} / {n}  ({errored / n * 100:5.1f}%)")
    print()

    # ---- ranked levers -------------------------------------------------- #
    print(" BIGGEST LEVERS (by p50 seconds per query)")
    levers = [(name, percentile(vals, 50)) for name, vals in per_query_node.items()]
    for name, p50 in sorted(levers, key=lambda kv: -kv[1])[:5]:
        print(f"  {p50:6.2f}s   {name}")
    print("=" * 82)


if __name__ == "__main__":
    main()
