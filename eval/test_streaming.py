"""Offline tests for src/streaming.py against the failure modes seen in baseline.json."""
import sys, types
sys.path.insert(0, '.')

# stub langchain_core.messages.AIMessage without importing the whole stack
class AIMessage:
    def __init__(self, content, id=None): self.content, self.id = content, id
mod = types.ModuleType("langchain_core.messages"); mod.AIMessage = AIMessage
core = types.ModuleType("langchain_core"); core.messages = mod
sys.modules.setdefault("langchain_core", core); sys.modules["langchain_core.messages"] = mod

import importlib.util
_spec = importlib.util.spec_from_file_location("_streaming", "src/streaming.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
iter_answer, collect_answer, final_answer_from_messages = _m.iter_answer, _m.collect_answer, _m.final_answer_from_messages

class C:
    def __init__(self, t): self.content = t
def ev(text, node="final_answer", step=1): return (C(text), {"langgraph_node": node, "langgraph_step": step})

FAIL=[]
def check(l,g,w):
    if g!=w: FAIL.append(f"{l}\n      got:  {g!r}\n      want: {w!r}"); print(f"  FAIL {l}")
    else: print(f"  ok   {l}")

print("\n--- accumulation (the bug: old code kept only the last delta) ---")
deltas = ["Under ","Section 106 ","of the TPA, ","notice is required."]
check("full answer assembled", collect_answer([ev(d) for d in deltas]),
      "Under Section 106 of the TPA, notice is required.")
check("old behaviour would have given", deltas[-1], "notice is required.")

print("\n--- retry: second attempt REPLACES the first ---")
attempt1 = [ev(t, step=3) for t in ["First draft ","that scored low."]]
attempt2 = [ev(t, step=9) for t in ["Second draft ","which is better."]]
check("only the later draft survives", collect_answer(attempt1+attempt2),
      "Second draft which is better.")
check("not concatenated", "First draft" in collect_answer(attempt1+attempt2), False)

print("\n--- filtering ---")
mixed = [ev("Context stored.", node="tools", step=1), ev("routing...", node="router", step=2),
         ev("Real ", step=3), ev("context stored", step=3), ev("answer.", step=3)]
check("other nodes ignored", collect_answer(mixed), "Real answer.")
check("noise marker dropped", "context stored" in collect_answer(mixed).lower(), False)
check("empty deltas skipped", collect_answer([ev("A",step=1), ev("",step=1), ev("B",step=1)]), "AB")
check("empty stream", collect_answer([]), "")

print("\n--- progressive rendering ---")
check("yields answer-so-far", list(iter_answer([ev(d) for d in ["a","b","c"]])), ["a","ab","abc"])

print("\n--- final_answer_from_messages (auditor may rewrite) ---")
check("last non-empty AIMessage wins",
      final_answer_from_messages([AIMessage("draft"), AIMessage("audited final")]), "audited final")
check("skips trailing empties",
      final_answer_from_messages([AIMessage("real"), AIMessage(""), AIMessage("   ")]), "real")
check("no messages", final_answer_from_messages([]), "")
check("non-AIMessage ignored", final_answer_from_messages([AIMessage("keep"), object()]), "keep")

print("\n--- regression: the three broken baseline cases ---")
for name, frag in [("tpa-eviction","‑"), ("contract-breach"," ("), ("ni-cheque"," payment")]:
    full = [ev("The full legal answer ", step=5), ev("ending with", step=5), ev(frag, step=5)]
    got = collect_answer(full)
    check(f"{name}: no longer truncates to {frag!r}", got.startswith("The full legal answer"), True)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S):"); [print("  -",f) for f in FAIL]; sys.exit(1)
print("ALL STREAMING TESTS PASS")
