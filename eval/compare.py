#!/usr/bin/env python3
"""
Regression gate: compare two eval runs, using measured noise as the threshold.

    python -m eval.compare eval/results/baseline.json eval/results/candidate.json

This pipeline is nondeterministic. The dispatcher and the multi-query expansion
write different search text on every run, so retrieval returns different Acts for
the same question - in one observed pair, 10 of 16 cases did. A single run per
case therefore cannot distinguish a real improvement from luck, and an earlier
version of this script reported a "-0.27 section_recall REGRESSION" produced
entirely by that variance.

So a delta is only called real when it exceeds the within-case spread actually
observed in the runs being compared. Runs made with --repeats 1 carry no spread
information; the gate says so rather than pretending to judge.

Exit code is non-zero only on quality regression. Latency is reported but never
fails the run: getting slower is a judgement call, getting less accurate is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# metric -> (higher_is_better, floor). The effective threshold is the larger of
# this floor and the measured noise, so the floor only matters when a pipeline
# is unusually stable.
QUALITY = {
    "section_recall": (True, 0.02),
    "act_recall": (True, 0.02),
    "citation_precision": (True, 0.02),
    "forbidden_hits": (False, 0),
    "errors": (False, 0),
}
LATENCY = ["mean_total_s", "mean_llm_calls", "mean_retrieval_s", "fallback_rate", "retry_rate"]

# summary key -> the per-case metric its noise band is recorded under
NOISE_KEY = {
    "section_recall": "section_recall", "act_recall": "act_recall",
    "citation_precision": "citation_precision", "mean_total_s": "total_s",
    "mean_llm_calls": "llm_calls", "mean_retrieval_s": "retrieval_s",
    "fallback_rate": "used_fallback", "retry_rate": "retried",
}


def fmt(v):
    return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def threshold(metric, base, cand, floor):
    """Larger of the configured floor and the noise seen in either run."""
    bands = [r.get("summary", {}).get("noise", {}).get(NOISE_KEY.get(metric, ""), 0.0) or 0.0
             for r in (base, cand)]
    return max(floor, *bands)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()

    base = json.loads(args.baseline.read_text(encoding="utf-8"))
    cand = json.loads(args.candidate.read_text(encoding="utf-8"))
    b, c = base["summary"], cand["summary"]

    reps = (b.get("repeats", 1) or 1, c.get("repeats", 1) or 1)
    if min(reps) < 2:
        print(f"\n!! baseline has {reps[0]} run(s)/case, candidate {reps[1]}. With fewer than 2,")
        print("   no noise floor can be measured and any delta below is unverified.")
        print("   Re-run with --repeats 3 before trusting a verdict.\n")

    print(f"{'metric':<24}{base['label']:>14}{cand['label']:>14}{'delta':>11}{'noise':>9}")
    print("-" * 72)

    regressions, suppressed = [], []
    for metric, (higher_better, floor) in QUALITY.items():
        bv, cv = b.get(metric), c.get(metric)
        if bv is None or cv is None:
            print(f"{metric:<24}{fmt(bv):>14}{fmt(cv):>14}{'n/a':>11}")
            continue
        delta = cv - bv
        tol = threshold(metric, base, cand, floor)
        worse = (delta < -tol) if higher_better else (delta > tol)
        moved = abs(delta) > tol
        flag = "  REGRESSION" if worse else ("  improved" if moved else "  within noise")
        print(f"{metric:<24}{fmt(bv):>14}{fmt(cv):>14}{delta:>+11.4f}{tol:>9.3f}{flag}")
        if worse:
            regressions.append(metric)
        elif not moved and abs(delta) > floor:
            suppressed.append(metric)

    print()
    for metric in LATENCY:
        bv, cv = b.get(metric), c.get(metric)
        if bv is None or cv is None:
            continue
        delta = cv - bv
        tol = threshold(metric, base, cand, 0.0)
        verdict = "within noise" if abs(delta) <= tol else ("faster" if delta < 0 else "slower")
        print(f"{metric:<24}{fmt(bv):>14}{fmt(cv):>14}{delta:>+11.4f}{tol:>9.3f}  {verdict}")

    # Per-case movement, so a flat average hiding two cancelling swings is visible.
    bpc, cpc = base.get("per_case", {}), cand.get("per_case", {})
    if bpc and cpc:
        moved = []
        for cid, cc in cpc.items():
            bc = bpc.get(cid)
            if not bc:
                continue
            bm, cm = bc["section_recall"]["median"], cc["section_recall"]["median"]
            if bm is None or cm is None:
                continue
            tol = max(bc["section_recall"]["spread"], cc["section_recall"]["spread"], 0.01)
            if abs(cm - bm) > tol:
                moved.append((cm - bm, cid, bm, cm, tol))
        if moved:
            print("\nper-case section_recall moves beyond that case's own spread:")
            for d, cid, bm, cm, tol in sorted(moved):
                print(f"  {d:+.2f}  {cid:<28} {bm:.2f} -> {cm:.2f}   (spread {tol:.2f})")
        else:
            print("\nno per-case section_recall move exceeded that case's own spread.")

    newly_forbidden = [cid for cid, cc in cpc.items()
                       if cc.get("forbidden_any") and not bpc.get(cid, {}).get("forbidden_any")]
    if newly_forbidden:
        print(f"\nNEW forbidden citations in: {', '.join(newly_forbidden)}")
        regressions.append("forbidden_cited")

    if suppressed:
        print(f"\nmoved but within noise (not counted): {', '.join(suppressed)}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} quality regression(s) beyond measured noise.")
        return 1
    print("\nPASS: no quality regression beyond measured noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
