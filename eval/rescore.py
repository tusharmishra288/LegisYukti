#!/usr/bin/env python3
"""
Recompute metrics on an existing results file, offline.

Scoring logic changes; pipeline runs are expensive. Any results file produced by
a version of run_eval.py that persists `raw_answer` / `raw_context` can be
re-scored in milliseconds instead of re-run.

    python -m eval.rescore eval/results/baseline.json --out eval/results/baseline-rescored.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, help="defaults to overwriting in place")
    args = ap.parse_args()

    from eval.metrics import (parse_verified_references, retrieval_score,
                              citation_score, forbidden_citations)
    from eval.run_eval import summarise
    import yaml

    data = json.loads(args.results.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in yaml.safe_load(
        (PROJECT_ROOT / "eval" / "golden_set.yaml").read_text(encoding="utf-8"))}

    missing_raw = [r["id"] for r in data["rows"] if "raw_context" not in r]
    if missing_raw:
        print(f"ERROR: {len(missing_raw)} row(s) have no raw_context - they predate raw-text\n"
              f"persistence and cannot be re-scored. Re-run the eval instead.\n"
              f"  affected: {', '.join(missing_raw[:5])}{' ...' if len(missing_raw) > 5 else ''}")
        return 1

    for row in data["rows"]:
        case = cases.get(row["id"])
        if not case:
            continue
        # Same selection rule as run_eval: the delivered (post-auditor) answer,
        # not the concatenation of every generation pass.
        state_answer = (row.get("raw_state_answer") or "").strip()
        answer = state_answer or row["raw_answer"]
        row["raw_answer"] = answer
        row["answer_chars"] = len(answer)
        retrieved = parse_verified_references(row["raw_context"])
        expected = dict(case.get("expected_sections") or {})
        for act in case.get("expected_acts", []):
            expected.setdefault(act, [])

        rs = retrieval_score(expected, retrieved) if expected else None
        cs = citation_score(answer, retrieved)
        row.update({
            "section_recall": rs.section_recall if rs else None,
            "act_recall": rs.act_recall if rs else None,
            "missing": rs.missing if rs else [],
            "retrieved_acts": sorted(rs.retrieved_acts) if rs else [],
            "citation_precision": cs.precision,
            "citations_total": cs.total,
            "unsupported": cs.unsupported,
            "out_of_corpus": cs.out_of_corpus,
            "forbidden_cited": forbidden_citations(answer, case.get("must_not_cite", [])),
        })

    data["summary"] = summarise(data["rows"])
    out = args.out or args.results
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rescored -> {out}")
    for k, v in data["summary"].items():
        print(f"  {k:<22} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
