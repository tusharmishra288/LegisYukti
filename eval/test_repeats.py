"""Offline tests for repeats aggregation + the noise-aware gate."""
import sys, json, random, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from eval.run_eval import aggregate_cases, noise_bands, summarise, _median

FAIL=[]
def check(l,g,w):
    if g!=w: FAIL.append(f"{l}: got {g!r} want {w!r}"); print(f"  FAIL {l}")
    else: print(f"  ok   {l}")

def row(cid, rep, sec, act=None, prec=0.5, total=40, retried=True, err=None, cat="single-act", fb=False):
    return {"id":cid,"repeat":rep,"category":cat,"error":err,"section_recall":sec,
            "act_recall":act if act is not None else sec,"citation_precision":prec,
            "citations_total":4,"unsupported":[],"out_of_corpus":[],"forbidden_cited":[],
            "app_eval_score":5,"total_s":total,"llm_calls":9,"retrieval_s":4.0,
            "used_fallback":fb,"retried":retried,"answer_chars":2000}

print("\n--- _median ---")
check("odd", _median([1,5,3]), 3)
check("even", _median([1,3,5,7]), 4.0)
check("empty", _median([]), None)

print("\n--- aggregate_cases: one flaky case ---")
rows=[row("flaky",0,1.0), row("flaky",1,0.0), row("flaky",2,1.0),
      row("stable",0,1.0), row("stable",1,1.0), row("stable",2,1.0)]
agg=aggregate_cases(rows)
check("flaky median resists the outlier", agg["flaky"]["section_recall"]["median"], 1.0)
check("flaky spread captured", agg["flaky"]["section_recall"]["spread"], 1.0)
check("stable spread is 0", agg["stable"]["section_recall"]["spread"], 0.0)
check("runs counted", agg["flaky"]["runs"], 3)

print("\n--- noise_bands ---")
check("band = median of per-case spreads", noise_bands(agg)["section_recall"], 0.5)

print("\n--- summarise uses per-case medians ---")
s=summarise(rows)
check("repeats reported", s["repeats"], 3)
check("cases (not runs)", s["cases"], 2)
check("total_runs", s["total_runs"], 6)
check("recall from medians, not raw mean", s["section_recall"], 1.0)

print("\n--- errored runs excluded, case still scored from its good runs ---")
rows2=[row("c",0,1.0), row("c",1,0.0,err="RateLimitError: 429"), row("c",2,1.0)]
a2=aggregate_cases(rows2)
check("only non-errored runs aggregated", a2["c"]["section_recall"]["n"], 2)
check("errors counted", a2["c"]["errors"], 1)

print("\n--- THE REAL TEST: replay the false regression ---")
# baseline: 4 cases at 1.00. candidate: same 4 cases, but 4 flipped to 0.00 -
# exactly the pattern from baseline-final vs rubric-fix. With repeats, the
# within-case spread should reveal it as noise, not regression.
def build(label, seed):
    rnd=random.Random(seed); rows=[]
    for cid in ["a","b","c","d","e","f"]:
        for rep in range(3):
            rows.append(row(cid, rep, rnd.choice([0.0,1.0,1.0])))   # genuinely flaky
    return {"label":label,"summary":summarise(rows),"per_case":aggregate_cases(rows),"rows":rows}

d=Path(tempfile.mkdtemp())
(d/"base.json").write_text(json.dumps(build("baseline",1)))
(d/"cand.json").write_text(json.dumps(build("candidate",99)))
r=subprocess.run([sys.executable,"-m","eval.compare",str(d/"base.json"),str(d/"cand.json")],
                 capture_output=True,text=True,cwd=Path.cwd())
print(r.stdout)
check("flaky-vs-flaky is NOT called a regression", r.returncode, 0)
check("reports noise-aware pass", "beyond measured noise" in r.stdout, True)

print("\n--- a REAL regression must still be caught ---")
good={"label":"good","rows":[],"per_case":{},"summary":{}}
grows=[row(c,rep,1.0) for c in ["a","b","c","d"] for rep in range(3)]
brows=[row(c,rep,0.0) for c in ["a","b","c","d"] for rep in range(3)]
(d/"g.json").write_text(json.dumps({"label":"good","summary":summarise(grows),"per_case":aggregate_cases(grows),"rows":grows}))
(d/"b.json").write_text(json.dumps({"label":"bad","summary":summarise(brows),"per_case":aggregate_cases(brows),"rows":brows}))
r2=subprocess.run([sys.executable,"-m","eval.compare",str(d/"g.json"),str(d/"b.json")],
                  capture_output=True,text=True,cwd=Path.cwd())
check("consistent drop IS flagged", r2.returncode, 1)
check("names it a REGRESSION", "REGRESSION" in r2.stdout, True)

print("\n--- single-repeat runs are warned about ---")
srows=[row("a",0,1.0)]
(d/"s.json").write_text(json.dumps({"label":"single","summary":summarise(srows),"per_case":aggregate_cases(srows),"rows":srows}))
r3=subprocess.run([sys.executable,"-m","eval.compare",str(d/"s.json"),str(d/"s.json")],
                  capture_output=True,text=True,cwd=Path.cwd())
check("warns when noise cannot be measured", "no noise floor can be measured" in r3.stdout, True)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S):"); [print("  -",f) for f in FAIL]; sys.exit(1)
print("ALL REPEAT/NOISE TESTS PASS")
