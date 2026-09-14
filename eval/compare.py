#!/usr/bin/env python3
"""
Regression gate: compare two eval runs and fail on quality loss.

    python -m eval.compare eval/results/baseline.json eval/results/soft-prior.json

Exit code is non-zero if any quality metric drops beyond tolerance, so this can
sit in CI. Latency metrics are reported but never fail the run - getting slower
is a judgement call, getting less accurate is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# metric -> (higher_is_better, tolerance). Tolerance absorbs LLM nondeterminism;
# tighten once you have a feel for run-to-run variance on your own set.
QUALITY = {
    "section_recall":     (True, 0.02),
    "act_recall":         (True, 0.02),
    "citation_precision": (True, 0.02),
    "forbidden_hits":     (False, 0),
    "errors":             (False, 0),
}
LATENCY = ["mean_total_s", "mean_llm_calls", "mean_retrieval_s", "fallback_rate", "retry_rate"]


def fmt(v):
    return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text())
    cand = json.loads(args.candidate.read_text())
    b, c = base["summary"], cand["summary"]

    print(f"\n{'metric':<24}{base['label']:>14}{cand['label']:>14}{'delta':>12}")
    print("-" * 64)

    regressions = []
    for metric, (higher_better, tol) in QUALITY.items():
        bv, cv = b.get(metric), c.get(metric)
        if bv is None or cv is None:
            print(f"{metric:<24}{fmt(bv):>14}{fmt(cv):>14}{'n/a':>12}")
            continue
        delta = cv - bv
        worse = (delta < -tol) if higher_better else (delta > tol)
        flag = "  REGRESSION" if worse else ""
        print(f"{metric:<24}{fmt(bv):>14}{fmt(cv):>14}{delta:>+12.4f}{flag}")
        if worse:
            regressions.append((metric, bv, cv))

    print()
    for metric in LATENCY:
        bv, cv = b.get(metric), c.get(metric)
        if bv is None or cv is None:
            continue
        delta = cv - bv
        arrow = "faster" if delta < 0 else ("slower" if delta > 0 else "")
        print(f"{metric:<24}{fmt(bv):>14}{fmt(cv):>14}{delta:>+12.4f}  {arrow}")

    # Per-case movement, so a flat average that hides two cancelling swings is visible.
    brows = {r["id"]: r for r in base["rows"]}
    moved = []
    for r in cand["rows"]:
        prev = brows.get(r["id"])
        if not prev or prev.get("section_recall") is None or r.get("section_recall") is None:
            continue
        d = r["section_recall"] - prev["section_recall"]
        if abs(d) > 0.01:
            moved.append((d, r["id"], prev["section_recall"], r["section_recall"]))
    if moved:
        print("\nper-case section_recall changes:")
        for d, cid, pv, cv_ in sorted(moved):
            print(f"  {d:+.2f}  {cid:<28} {pv:.2f} -> {cv_:.2f}")

    newly_forbidden = [r["id"] for r in cand["rows"]
                       if r["forbidden_cited"] and not brows.get(r["id"], {}).get("forbidden_cited")]
    if newly_forbidden:
        print(f"\nNEW forbidden citations in: {', '.join(newly_forbidden)}")
        regressions.append(("forbidden_cited", "-", newly_forbidden))

    if regressions:
        print(f"\nFAIL: {len(regressions)} quality regression(s).")
        return 1
    print("\nPASS: no quality regression.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
