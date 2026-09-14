"""Offline unit tests for the scoring primitives. No network, no LangChain.

Run:  python -m eval.test_metrics
"""
import sys
from eval.metrics import (
    normalize_act, normalize_section, extract_citations, parse_verified_references,
    retrieval_score, citation_score, forbidden_citations,
)

FAILURES = []

def check(label, got, want):
    if got != want:
        FAILURES.append(f"{label}\n      got:  {got!r}\n      want: {want!r}")
        print(f"  FAIL  {label}")
    else:
        print(f"  ok    {label}")

print("\n--- normalize_act ---")
check("full name", normalize_act("Transfer of Property Act, 1882"), "TRANSFER OF PROPERTY ACT 1882")
check("abbrev TPA", normalize_act("TPA"), "TRANSFER OF PROPERTY ACT 1882")
check("stored form", normalize_act("transfer of property act 1882"), "TRANSFER OF PROPERTY ACT 1882")
check("BNSS not shadowed by BNS", normalize_act("BNSS"), "BHARATIYA NAGARIK SURAKSHA SANHITA BNSS 2023")
check("BNS", normalize_act("BNS"), "BHARATIYA NYAYA SANHITA BNS 2023")
check("NI Act", normalize_act("the N.I. Act"), "NEGOTIABLE INSTRUMENTS ACT 1881")
check("hindu marriage no trailing space", normalize_act("Hindu Marriage Act 1955"), "THE HINDU MARRIAGE ACT 1955")
check("NDPS correct spelling", normalize_act("NDPS Act"), "NARCOTIC DRUGS AND PSYCHOTROPIC SUBSTANCES ACT 1985")
check("out of corpus IPC", normalize_act("IPC"), "INDIAN PENAL CODE")
check("unknown", normalize_act("Motor Vehicles Act"), None)
check("substring guard", normalize_act("bnsx"), None)

print("\n--- normalize_section ---")
check("plain", normalize_section("106"), "106")
check("hyphenated", normalize_section("498-A"), "498A")
check("spaced", normalize_section("66 A"), "66A")
check("subsection stripped", normalize_section("13(1)(ia)"), "13")
# processor.py stores the rendered header tag, and the tool output labels it again
check("stored header tag", normalize_section("Section 106"), "106")
check("double-labelled tool output", normalize_section("Section Section 106"), "106")
check("constitution article tag", normalize_section("Section Article 21"), "21")
check("suffix survives label strip", normalize_section("Section 13B"), "13B")
check("placeholder General -> None", normalize_section("General"), None)
check("placeholder N/A -> None", normalize_section("N/A"), None)

print("\n--- extract_citations ---")
c = extract_citations("Under Section 106 of the Transfer of Property Act, 1882, notice is required.")
check("single attributed", [(x.number, x.act) for x in c], [("106", "TRANSFER OF PROPERTY ACT 1882")])

c = extract_citations("See Sections 13 and 13B of the Hindu Marriage Act.")
check("enumerated list", sorted(x.number for x in c), ["13", "13B"])
check("list attribution", {x.act for x in c}, {"THE HINDU MARRIAGE ACT 1955"})

c = extract_citations("A complaint u/s 138 NI Act must be filed within 30 days.")
check("u/s form", [(x.number, x.act) for x in c], [("138", "NEGOTIABLE INSTRUMENTS ACT 1881")])

c = extract_citations("Article 21 of the Constitution guarantees this.")
check("article kind", [(x.kind, x.number, x.act) for x in c],
      [("article", "21", "CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS")])

c = extract_citations("Seek an injunction under Order 39 Rule 1 CPC.")
check("order/rule", [(x.kind, x.number, x.act) for x in c],
      [("order", "39R1", "CODE OF CIVIL PROCEDURE CPC 1908")])

c = extract_citations("The Contract Act applies. Section 73 provides damages.")
check("backward attribution", [(x.number, x.act) for x in c], [("73", "THE INDIAN CONTRACT ACT 1872")])

c = extract_citations("Section 106 requires notice.")
check("unattributable stays None", [(x.number, x.act) for x in c], [("106", None)])

check("empty text", extract_citations(""), [])

print("\n--- parse_verified_references: real ingestion shape ---")
# What retrieve_legal_context actually emits, given section == "Section 106"
real = ("--- VERIFIED REFERENCE: TRANSFER OF PROPERTY ACT 1882 (Section Section 106) ---\ntext\n"
        "--- VERIFIED REFERENCE: CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS (Section Article 21) ---\nt\n"
        "--- VERIFIED REFERENCE: INDIAN SUCCESSION ACT 1925 (Section General) ---\nt")
check("double-label parsed", parse_verified_references(real),
      {("TRANSFER OF PROPERTY ACT 1882", "106"),
       ("CONSTITUTION OF INDIA FUNDAMENTAL RIGHTS", "21"),
       ("INDIAN SUCCESSION ACT 1925", None)})

print("\n--- parse_verified_references ---")
ctx = ("--- VERIFIED REFERENCE: TRANSFER OF PROPERTY ACT 1882 (Section 106) ---\ntext...\n"
       "--- VERIFIED REFERENCE: CODE OF CIVIL PROCEDURE CPC 1908 (Section 9) ---\nmore...\n")
check("two refs", parse_verified_references(ctx),
      {("TRANSFER OF PROPERTY ACT 1882", "106"), ("CODE OF CIVIL PROCEDURE CPC 1908", "9")})
check("force-add shape (no section)",
      parse_verified_references("--- VERIFIED REFERENCE: POCSO ACT 2012 ---\nbody"),
      {("POCSO ACT 2012", None)})

print("\n--- retrieval_score ---")
retrieved = {("TRANSFER OF PROPERTY ACT 1882", "106"), ("TRANSFER OF PROPERTY ACT 1882", "111")}
s = retrieval_score({"TRANSFER OF PROPERTY ACT 1882": ["106", "111"]}, retrieved)
check("perfect recall", (s.section_recall, s.act_recall), (1.0, 1.0))
s = retrieval_score({"TRANSFER OF PROPERTY ACT 1882": ["106", "111", "114"]}, retrieved)
check("partial recall", round(s.section_recall, 3), 0.667)
check("missing listed", s.missing, ["TRANSFER OF PROPERTY ACT 1882 s.114"])
s = retrieval_score({"INDIAN SUCCESSION ACT 1925": ["59"]}, retrieved)
check("wrong act -> zero", (s.section_recall, s.act_recall), (0.0, 0.0))

print("\n--- citation_score ---")
answer = ("Under Section 106 of the Transfer of Property Act, 1882 you are entitled to notice. "
          "You may also rely on Section 59 of the Indian Succession Act, 1925.")
cs = citation_score(answer, retrieved)
check("one supported one not", (cs.supported, cs.total), (1, 2))
check("hallucination named", cs.unsupported, ["INDIAN SUCCESSION ACT 1925 s.59"])
check("precision", cs.precision, 0.5)

cs = citation_score("For a pre-2024 incident, Section 420 IPC applied.", retrieved)
check("IPC excluded from grading", (cs.total, cs.out_of_corpus), (0, ["INDIAN PENAL CODE s.420"]))
check("precision is None when nothing gradeable", cs.precision, None)

cs = citation_score("Section 106 requires notice.", retrieved)
check("lenient unattributed credited", (cs.supported, cs.unattributed), (1, 1))
cs = citation_score("Section 106 requires notice.", retrieved, lenient_unattributed=False)
check("strict unattributed penalised", cs.supported, 0)

print("\n--- forbidden_citations ---")
check("regression guard fires",
      forbidden_citations(answer, ["INDIAN SUCCESSION ACT 1925"]), ["INDIAN SUCCESSION ACT 1925"])
check("clean answer passes",
      forbidden_citations("Section 106 of the TPA applies.", ["INDIAN SUCCESSION ACT 1925"]), [])

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("ALL TESTS PASS")
