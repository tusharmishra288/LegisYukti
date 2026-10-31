# Evaluation harness (Phase 0)

Exists so that "the refactor didn't hurt answer quality" is a measurement rather
than a hope. Freeze a baseline before changing the pipeline, then gate every
change against it.

## Files

| file | needs network | what it does |
|---|---|---|
| `invariants.py` | no | static consistency checks on the retrieval config |
| `metrics.py` | no | citation extraction and scoring primitives |
| `test_metrics.py` | no | 33 unit tests for the above |
| `golden_set.yaml` | — | the questions, with expected Acts and sections |
| `run_eval.py` | **yes** | runs the golden set through the real graph and scores it |
| `compare.py` | no | regression gate between two runs |

## Start here

```bash
python -m eval.invariants      # zero network, ~instant
python -m eval.test_metrics    # zero network
```

`invariants.py` currently **fails**, and the failure is real: two Acts are
unreachable by filtered search because the routing values do not match the
`law_name` that ingestion writes (`Path(pdf).stem.replace("_"," ").upper()`).

- `'THE HINDU MARRIAGE ACT 1955 '` — trailing space
- `'NARCOTIC DRUGS AND PYSCHOTROPIC SUBSTANCES ACT 1985'` — `PYSCHOTROPIC` vs `PSYCHOTROPIC`

Every divorce, alimony, maintenance, custody and NDPS query therefore applies a
Qdrant filter that matches zero points, gets an empty result, and pays the global
fallback — a second full retrieval (multi-query expansion, both vector searches,
the rerank) every single time. Two characters in `src/agent.py` fix it.

Fix it **after** capturing the baseline, so the harness can prove the effect.

## Full loop

```bash
python -m eval.run_eval --label baseline
# ... make a change ...
python -m eval.run_eval --label soft-prior
python -m eval.compare eval/results/baseline.json eval/results/soft-prior.json
```

`run_eval` needs `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` and a populated
`indian_legal_library`. It uses an in-memory checkpointer, so eval runs never
touch real workspaces. `POSTGRES_URI` is not required.

`compare` exits non-zero only on **quality** regression. Latency is reported but
never fails the run: getting slower is a judgement call, getting less accurate is
not.

## Metrics

- **section_recall** — expected (Act, section) pairs that reached the context.
  The primary number: if retrieval misses it, no prompt can recover it.
- **act_recall** — expected Acts present at all. Scores cases where several
  sections would each be defensible.
- **citation_precision** — citations in the answer that are backed by retrieved
  context. This is the deterministic version of the LLM auditor: exact, instant,
  no false negatives. IPC/CrPC are excluded from grading, since the app
  deliberately points at them for pre-July-2024 incidents.
- **forbidden_hits** — answers citing an Act listed in `must_not_cite`. Pins
  known misroutes so they cannot silently return.
- **fallback_rate / retry_rate** — how often the two latency-doubling paths fire.

## Extending the set

16 seeded cases; aim for ~50. **Review the seeded section numbers** — they are
starting points and you are the domain owner. Keep the balance across
`single-act`, `cross-act`, `procedural`, `followup` and `chat`, and add a
`must_not_cite` entry every time you find a misroute, so each bug fixed stays
fixed.
