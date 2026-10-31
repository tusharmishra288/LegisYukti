"""
Scoring primitives for the LegisYukti evaluation harness.

Pure functions, no network and no LangChain imports, so the measurement logic can
be unit-tested independently of the pipeline it measures. Everything here answers
one of three questions about a single query:

  1. Did retrieval surface the sections the question actually needs?  -> recall
  2. Is every citation in the answer backed by retrieved context?     -> precision
  3. Did the answer cite an Act it had no business citing?            -> forbidden

Citations are matched on (Act, section) pairs. Act names are normalised through an
alias table onto the canonical `law_name` values that ingestion stores, which are
derived from the PDF filenames: Path(pdf).stem.replace("_", " ").upper()
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Act normalisation
# --------------------------------------------------------------------------- #

CANONICAL_ACTS = [
    "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023",
    "BHARATIYA NYAYA SANHITA BNS 2023",
    "BHARATIYA SAKSHYA ADHINIYAM BSA 2023",
    "CODE OF CIVIL PROCEDURE CPC 1908",
    "CODE ON WAGES 2019",
    "CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS",
    "CONSUMER PROTECTION ACT 2019",
    "INDIAN SUCCESSION ACT 1925",
    "INFORMATION TECHNOLOGY ACT 2000",
    "NARCOTIC DRUGS AND PSYCHOTROPIC SUBSTANCES ACT 1985",
    "NEGOTIABLE INSTRUMENTS ACT 1881",
    "POCSO ACT 2012",
    "REGISTRATION ACT 1908",
    "SPECIAL MARRIAGE ACT 1954",
    "THE HINDU MARRIAGE ACT 1955",
    "THE INDIAN CONTRACT ACT 1872",
    "TRANSFER OF PROPERTY ACT 1882",
]

# Acts a correct answer may reference but which are NOT in the corpus. Citing one
# is not a hallucination against context - the app deliberately points at IPC/CrPC
# for pre-July-2024 incidents - so these are reported separately rather than
# counted as unsupported.
OUT_OF_CORPUS = {"INDIAN PENAL CODE", "CODE OF CRIMINAL PROCEDURE", "HINDU SUCCESSION ACT", "EVIDENCE ACT 1872"}

# Longest-first so "bnss" is never shadowed by "bns".
_ALIASES: list[tuple[str, str]] = [
    ("bharatiya nagarik suraksha sanhita", "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023"),
    ("nagarik suraksha", "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023"),
    ("bnss", "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023"),
    ("bharatiya nyaya sanhita", "BHARATIYA NYAYA SANHITA BNS 2023"),
    ("nyaya sanhita", "BHARATIYA NYAYA SANHITA BNS 2023"),
    ("bns", "BHARATIYA NYAYA SANHITA BNS 2023"),
    ("bharatiya sakshya adhiniyam", "BHARATIYA SAKSHYA ADHINIYAM BSA 2023"),
    ("sakshya adhiniyam", "BHARATIYA SAKSHYA ADHINIYAM BSA 2023"),
    ("bsa", "BHARATIYA SAKSHYA ADHINIYAM BSA 2023"),
    ("code of civil procedure", "CODE OF CIVIL PROCEDURE CPC 1908"),
    ("civil procedure code", "CODE OF CIVIL PROCEDURE CPC 1908"),
    ("cpc", "CODE OF CIVIL PROCEDURE CPC 1908"),
    ("code on wages", "CODE ON WAGES 2019"),
    ("wages code", "CODE ON WAGES 2019"),
    ("constitution of india", "CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS"),
    ("constitution", "CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS"),
    ("consumer protection", "CONSUMER PROTECTION ACT 2019"),
    ("indian succession act", "INDIAN SUCCESSION ACT 1925"),
    ("succession act", "INDIAN SUCCESSION ACT 1925"),
    ("isa", "INDIAN SUCCESSION ACT 1925"),
    ("information technology", "INFORMATION TECHNOLOGY ACT 2000"),
    ("it act", "INFORMATION TECHNOLOGY ACT 2000"),
    ("narcotic drugs and psychotropic substances", "NARCOTIC DRUGS AND PSYCHOTROPIC SUBSTANCES ACT 1985"),
    ("narcotic drugs", "NARCOTIC DRUGS AND PSYCHOTROPIC SUBSTANCES ACT 1985"),
    ("ndps", "NARCOTIC DRUGS AND PSYCHOTROPIC SUBSTANCES ACT 1985"),
    ("negotiable instruments", "NEGOTIABLE INSTRUMENTS ACT 1881"),
    ("n.i. act", "NEGOTIABLE INSTRUMENTS ACT 1881"),
    ("ni act", "NEGOTIABLE INSTRUMENTS ACT 1881"),
    ("pocso", "POCSO ACT 2012"),
    ("registration act", "REGISTRATION ACT 1908"),
    ("special marriage", "SPECIAL MARRIAGE ACT 1954"),
    ("sma", "SPECIAL MARRIAGE ACT 1954"),
    ("hindu marriage", "THE HINDU MARRIAGE ACT 1955"),
    ("hma", "THE HINDU MARRIAGE ACT 1955"),
    ("indian contract act", "THE INDIAN CONTRACT ACT 1872"),
    ("contract act", "THE INDIAN CONTRACT ACT 1872"),
    ("transfer of property", "TRANSFER OF PROPERTY ACT 1882"),
    ("tpa", "TRANSFER OF PROPERTY ACT 1882"),
    # out-of-corpus, recognised so they can be reported rather than mis-scored
    ("indian penal code", "INDIAN PENAL CODE"),
    ("ipc", "INDIAN PENAL CODE"),
    ("code of criminal procedure", "CODE OF CRIMINAL PROCEDURE"),
    ("crpc", "CODE OF CRIMINAL PROCEDURE"),
    ("hindu succession act", "HINDU SUCCESSION ACT"),
]

_ALIAS_RE = re.compile(
    "|".join(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])" for a, _ in _ALIASES),
    re.IGNORECASE,
)
_ALIAS_LOOKUP = {a: c for a, c in _ALIASES}


def normalize_act(text: str | None) -> str | None:
    """Map any spelling of an Act onto its canonical law_name, or None."""
    if not text:
        return None
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    if collapsed.upper() in CANONICAL_ACTS:
        return collapsed.upper()
    for alias, canonical in _ALIASES:          # longest-first ordering preserved
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", collapsed):
            return canonical
    return None


# processor.py stores `section` as the rendered header tag - "Section 106",
# "Article 21", or "General" for preamble chunks - not a bare number. The tool
# output then wraps it again, yielding literal "(Section Section 106)". Strip the
# label so both sides of a comparison are just the identifier.
_SECTION_LABEL_RE = re.compile(r"^\s*(?:sections?|secs?|s|articles?|arts?|orders?)\b[\s.:-]*", re.IGNORECASE)
_NON_SECTION_VALUES = {"", "NA", "GENERAL", "NONE", "UNKNOWN"}


def normalize_section(raw: str | None) -> str | None:
    """'498-A' / 'Section 106' / 'Article 21' -> '498A' / '106' / '21'.

    Subsections are dropped for matching, and placeholder values ingestion emits
    when a chunk has no numbered header ("General", "N/A") resolve to None rather
    than to a bogus section identifier.
    """
    if not raw:
        return None
    cleaned = str(raw)
    # Repeated because the tool output double-labels: "Section Section 106".
    for _ in range(3):
        stripped = _SECTION_LABEL_RE.sub("", cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    cleaned = re.sub(r"\([^)]*\)", "", cleaned)          # drop (1), (a)
    cleaned = re.sub(r"[^0-9A-Za-z]", "", cleaned).upper()
    if cleaned in _NON_SECTION_VALUES:
        return None
    return cleaned or None


# --------------------------------------------------------------------------- #
# Citation extraction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Citation:
    kind: str            # "section" | "article" | "order"
    number: str          # normalised, e.g. "106", "498A", "39R1"
    act: str | None      # canonical law_name, or None when unattributable
    raw: str

    @property
    def pair(self) -> tuple[str | None, str]:
        return (self.act, self.number)


_SECTION_RE = re.compile(
    r"(?P<kind>Sections?|Secs?\.|S\.|u/s|under\s+section|Articles?|Arts?\.)"
    r"\s*"
    r"(?P<num>\d+(?-i:(?:\s*-\s*[A-Z]{1,2}|[A-Z]{1,2})?)(?:\s*\([0-9a-z]+\))?)"
    r"(?P<more>(?:\s*(?:,|and|&|to)\s*\d+(?-i:(?:\s*-\s*[A-Z]{1,2}|[A-Z]{1,2})?))*)",
    re.IGNORECASE,
)
_ORDER_RE = re.compile(r"Order\s+([IVXL]+|\d+)\s*(?:,)?\s*Rule\s+(\d+)", re.IGNORECASE)
_EXTRA_NUM_RE = re.compile(r"\d+(?:\s*-\s*[A-Z]{1,2}|[A-Z]{1,2})?")

_ATTRIBUTION_FORWARD = 90   # chars after the number to look for the Act
_ATTRIBUTION_BACK = 160     # chars before, used only if nothing follows


def _attribute_act(text: str, start: int, end: int) -> str | None:
    ahead = text[end:end + _ATTRIBUTION_FORWARD]
    match = _ALIAS_RE.search(ahead)
    if match:
        return _ALIAS_LOOKUP[match.group(0).lower()]
    behind = text[max(0, start - _ATTRIBUTION_BACK):start]
    matches = list(_ALIAS_RE.finditer(behind))
    if matches:
        return _ALIAS_LOOKUP[matches[-1].group(0).lower()]
    return None


def extract_citations(text: str) -> list[Citation]:
    """Pull every statutory citation out of free-form answer text.

    Handles 'Section 106 of the Transfer of Property Act, 1882', 'u/s 138 NI Act',
    'Sections 13 and 13B of the HMA', 'Article 21', and 'Order 39 Rule 1 CPC'.
    """
    if not text:
        return []
    found: list[Citation] = []

    for m in _SECTION_RE.finditer(text):
        kind = "article" if m.group("kind").lower().startswith(("article", "art")) else "section"
        act = _attribute_act(text, m.start(), m.end())
        numbers = [m.group("num")]
        if m.group("more"):
            numbers += _EXTRA_NUM_RE.findall(m.group("more"))
        for raw_num in numbers:
            num = normalize_section(raw_num)
            if num:
                found.append(Citation(kind, num, act, m.group(0).strip()))

    for m in _ORDER_RE.finditer(text):
        act = _attribute_act(text, m.start(), m.end()) or "CODE OF CIVIL PROCEDURE CPC 1908"
        found.append(Citation("order", f"{m.group(1).upper()}R{m.group(2)}", act, m.group(0).strip()))

    # de-duplicate, preserving order
    seen, unique = set(), []
    for c in found:
        key = (c.kind, c.number, c.act)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


_VERIFIED_RE = re.compile(
    r"VERIFIED REFERENCE:\s*(?P<law>[^(\n]+?)\s*(?:\(Section\s*(?P<sec>[^)]*)\))?\s*---",
    re.IGNORECASE,
)


def parse_verified_references(context: str) -> set[tuple[str | None, str | None]]:
    """Parse the block that `retrieve_legal_context` returns into (act, section) pairs.

    Both output shapes are handled: the normal
    '--- VERIFIED REFERENCE: <LAW> (Section <n>) ---' and the context-starved
    force-add shape that omits the section.
    """
    pairs: set[tuple[str | None, str | None]] = set()
    for m in _VERIFIED_RE.finditer(context or ""):
        act = normalize_act(m.group("law"))
        sec = normalize_section(m.group("sec")) if m.group("sec") else None
        pairs.add((act, sec))
    return pairs


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

@dataclass
class RetrievalScore:
    section_recall: float      # expected (act, section) pairs that reached the context
    act_recall: float          # expected Acts represented at all
    missing: list[str]         # human-readable list of what was not retrieved
    retrieved_acts: set[str]


def retrieval_score(expected: dict[str, list[str]],
                    retrieved: set[tuple[str | None, str | None]]) -> RetrievalScore:
    """How much of what the question needs actually reached the model?

    `expected` maps Act -> list of sections. An expected Act with an empty section
    list scores on Act presence alone, which is the right call for questions where
    any of several sections would be a defensible citation.
    """
    retrieved_pairs = {(a, s) for a, s in retrieved if a}
    retrieved_acts = {a for a, _ in retrieved_pairs}

    wanted_pairs, missing = [], []
    for act, sections in expected.items():
        canonical = normalize_act(act) or act
        if not sections:
            continue
        for section in sections:
            key = (canonical, normalize_section(section))
            wanted_pairs.append(key)
            if key not in retrieved_pairs:
                missing.append(f"{canonical} s.{section}")

    wanted_acts = {normalize_act(a) or a for a in expected}
    act_recall = len(wanted_acts & retrieved_acts) / len(wanted_acts) if wanted_acts else 1.0
    section_recall = (
        sum(1 for p in wanted_pairs if p in retrieved_pairs) / len(wanted_pairs)
        if wanted_pairs else act_recall
    )
    return RetrievalScore(section_recall, act_recall, missing, retrieved_acts)


@dataclass
class CitationScore:
    precision: float | None    # None when the answer cites nothing gradeable
    total: int
    supported: int
    unsupported: list[str]     # cited but absent from context - candidate hallucinations
    out_of_corpus: list[str]   # e.g. IPC/CrPC: legitimate to mention, not gradeable
    unattributed: int          # 'Section 106' with no Act named nearby


def citation_score(answer: str,
                   retrieved: set[tuple[str | None, str | None]],
                   lenient_unattributed: bool = True) -> CitationScore:
    """Is every citation in the answer actually present in the retrieved context?

    This is the deterministic replacement for asking an 8b model whether the
    citations are grounded: exact, instant, and with no false negatives.

    `lenient_unattributed` credits a bare 'Section 106' when that section number
    appears under any retrieved Act, rather than penalising the answer for prose
    style. Set False for a strict count.
    """
    citations = extract_citations(answer)
    retrieved_pairs = {(a, s) for a, s in retrieved if a and s}
    retrieved_sections = {s for _, s in retrieved_pairs}

    supported, unsupported, out_of_corpus, unattributed = 0, [], [], 0
    gradeable = 0

    for c in citations:
        if c.act in OUT_OF_CORPUS:
            out_of_corpus.append(f"{c.act} s.{c.number}")
            continue
        gradeable += 1
        if c.act is None:
            unattributed += 1
            if lenient_unattributed and c.number in retrieved_sections:
                supported += 1
            else:
                unsupported.append(f"<unattributed> s.{c.number}")
            continue
        if (c.act, c.number) in retrieved_pairs:
            supported += 1
        else:
            unsupported.append(f"{c.act} s.{c.number}")

    precision = (supported / gradeable) if gradeable else None
    return CitationScore(precision, gradeable, supported, unsupported, out_of_corpus, unattributed)


def forbidden_citations(answer: str, must_not_cite: list[str]) -> list[str]:
    """Acts the answer must not rely on - encodes known misrouting regressions."""
    banned = {normalize_act(a) or a for a in (must_not_cite or [])}
    if not banned:
        return []
    return sorted({c.act for c in extract_citations(answer) if c.act in banned})
