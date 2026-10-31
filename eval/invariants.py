"""
Zero-network consistency checks for the retrieval configuration.

These catch a whole class of silent failure: a routing filter value that can never
match a stored `law_name`. When that happens the Qdrant filter returns zero points,
`retrieve_legal_context` falls through to the global fallback, and the query pays a
second full retrieval pipeline - every time, invisibly, with no error anywhere.

Two such mismatches exist as of this writing:
  * 'THE HINDU MARRIAGE ACT 1955 '  -> trailing space
  * '...PYSCHOTROPIC...'            -> transposed letters vs the PDF filename

`law_name` is derived in processor.py as:  Path(pdf).stem.replace("_", " ").upper()
so the routing values must match that transformation exactly.

Run:  python -m eval.invariants
Exits non-zero on any mismatch, which makes it usable as a CI gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
AGENT_PY = PROJECT_ROOT / "src" / "agent.py"


def stored_law_names(docs_dir: Path = DOCS_DIR) -> set[str]:
    """The law_name values ingestion actually writes into Qdrant metadata."""
    return {p.stem.replace("_", " ").upper() for p in sorted(docs_dir.glob("*.pdf"))}


def _extract_block(source: str, pattern: str) -> str:
    match = re.search(pattern, source, re.S)
    if not match:
        raise ValueError(f"could not locate block matching {pattern!r} in agent.py")
    return match.group(1)


def declared_library(agent_source: str) -> list[str]:
    return re.findall(r'"([^"]+)"', _extract_block(agent_source, r"LEGAL_LIBRARY = \[(.*?)\]"))


def declared_routing_keys(agent_source: str) -> list[str]:
    return re.findall(r'"([A-Z][^"]+)":\s*\[', _extract_block(agent_source, r"routing_map = \{(.*?)\n    \}"))


def check(docs_dir: Path = DOCS_DIR, agent_py: Path = AGENT_PY) -> list[str]:
    """Returns a list of problem descriptions; empty means the config is coherent."""
    problems: list[str] = []
    stored = stored_law_names(docs_dir)
    if not stored:
        return [f"no PDFs found in {docs_dir} - cannot validate law_name values"]

    source = agent_py.read_text(encoding="utf-8")
    library = declared_library(source)
    routing_keys = declared_routing_keys(source)

    def nearest(value: str) -> str | None:
        key = value.strip().replace(" ", "")[:14]
        for candidate in stored:
            if candidate.replace(" ", "")[:14] == key:
                return candidate
        return None

    for value in library:
        if value not in stored:
            near = nearest(value)
            hint = f"  did you mean {near!r}?" if near else "  no similar law_name in docs/"
            problems.append(
                f"LEGAL_LIBRARY value {value!r} matches no stored law_name."
                f"{hint}  Filtered search on it returns ZERO points."
            )

    for key in routing_keys:
        if key not in library:
            problems.append(f"routing_map key {key!r} is not in LEGAL_LIBRARY")

    for name in sorted(stored):
        if name not in library:
            problems.append(f"stored law_name {name!r} is unreachable - no filter value targets it")

    return problems


def main() -> int:
    problems = check()
    if not problems:
        print(f"OK: all {len(stored_law_names())} law_name values are reachable by a routing filter.")
        return 0
    print(f"FAIL: {len(problems)} retrieval-config problem(s)\n")
    for p in problems:
        print(f"  - {p}")
    print("\nEach mismatch forces the global-fallback path: a second full retrieval\n"
          "(multi-query expansion, both vector searches, and the rerank) on every\n"
          "affected query.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
