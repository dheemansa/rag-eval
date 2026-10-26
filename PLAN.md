# Final plan: rag-eval, "pytest for RAG"

**Pitch:** A RAG demo can sound right while retrieving the wrong passages. `rag-eval` is a reusable Python package that
scores any RAG pipeline on retrieval relevance, answer correctness, citation accuracy, and unsupported answers. It
blocks regressions in CI, and it proves its own judgments by catching deliberately broken pipelines.

## 1. Architecture

```
rag-eval/
├── rag_eval/
│   ├── schema.py      # RAGResult, GoldItem, Report
│   ├── runner.py      # evaluate(rag_fn, dataset, judge="replay")
│   ├── metrics/       # retrieval.py, faithfulness.py, citations.py, correctness.py
│   ├── judges/        # nli.py (local, CPU), api_judge.py, cache.py (record/replay)
│   ├── report.py      # JSON report + run manifest (config hashes, seeds, model versions)
│   └── cli.py         # run | compare | report
├── demo/              # corpus, golden set, 4 RAG variants (the Poison Lab)
├── dashboard/         # static site that reads report JSON
├── tests/
├── .github/workflows/rag-eval.yml
└── README.md
```

**Contract.** Any pipeline plugs in by returning this:

```python
RAGResult(question, answer, contexts=[{"id", "text"}], citations=["id", ...])
report = evaluate(rag_fn, dataset)
```

No framework adapters in the MVP. Developers map their output to this schema.

## 2. Metrics (cheap and keyless first, API judge last)

| Brief requirement   | Metrics                                                                                                           | Method                                                                                   | Needs API key?  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------- |
| Retrieval relevance | hit@k, recall@k, precision@k, MRR                                                                                 | Retrieved IDs vs gold passage IDs                                                        | No              |
| Unsupported answers | faithfulness (supported-claim rate), unsupported-claim rate, abstention accuracy                                  | Split the answer into claims, check each against contexts with a small NLI cross-encoder | No              |
| Citation accuracy   | citation validity (ID exists in contexts), citation support (cited passage entails the claim), uncited-claim rate | ID check plus NLI                                                                        | No              |
| Answer correctness  | token F1 plus judge verdict vs gold answer                                                                        | Token F1 first; API judge for the verdict                                                | Judge part only |

Keep the NLI model small (a DeBERTa-small class model) so it runs on CPU in CI. Check its license and download size
before committing to one.

## 3. Judge modes (settles the CI/key question)

`--judge none | replay | live`

| Environment                              | Mode                                   | Behavior                                                                                                                                       |
| ---------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| GitHub CI (no key)                       | `replay` (default)                     | Uses committed cached judge verdicts. On a cache miss, correctness shows "not evaluated" with a warning. `--strict` makes a miss fail the run. |
| Hosted demo site (key as server env var) | `live`                                 | Live playground calls the API.                                                                                                                 |
| Anyone cloning the repo                  | `replay`, or `live` with their own key | Poison Lab results reproduce with no key.                                                                                                      |

Cache key is a hash of (model, prompt, input), with temperature 0 and the model name pinned in the run manifest. The
judge is provider-agnostic and configurable, so nothing is hardcoded to one vendor or model version. A local Ollama
option sits behind a flag for privacy-minded users (P2).

## 4. Golden dataset

- About 50 questions over a small, open-licensed corpus, such as the docs of one Python library, so judges recognize the
  domain.
- Each item has the question, gold answer, gold passage IDs, and a type: single-hop, multi-hop, or **unanswerable**
  (about 10 items).
- Versioned JSONL with a schema validator. The same command gives the same numbers.

## 5. Poison Lab (the differentiator)

Four variants, each built to fail in a predictable way:

| Variant           | Injected fault                                                       | Expected effect                                                           |
| ----------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Baseline**      | none                                                                 | Reference scores                                                          |
| **Bad Retriever** | Random or weak retrieval                                             | Recall, MRR, and precision drop                                           |
| **Hallucinator**  | Prompt ignores context, invents answers **and fabricates citations** | Faithfulness and citation support drop, abstention fails on unanswerables |
| **Stale Data**    | Retrieves outdated doc versions                                      | **Faithfulness stays high, correctness drops**                            |

Stale Data is the best demo here. The answer is faithful to the retrieved text but wrong, which shows your metrics
measure different things rather than all moving together.

**Evaluate the evaluator.** Publish a "Can you trust this evaluator?" table:

- Expected vs observed metric drop for every injected fault (detection rate and size of the drop)
- Judge self-agreement across repeated runs
- Honest note that small NLI models and LLM judges are noisy, with the numbers to show how much

## 6. CI regression gate

```
rag-eval compare baseline.json candidate.json --fail-under faithfulness=0.85 --fail-under recall@5=0.8
```

- The gate depends only on keyless metrics, so CI never needs a secret.
- Ship `.github/workflows/rag-eval.yml`, with `actions/cache` for the NLI model.
- Commit a screenshot of a PR blocked by a degraded RAG. Better, keep the failing PR open in your repo as a live
  example.
- Output is a readable diff table (metric, baseline, candidate, delta, pass/fail).

## 7. Dashboard and deployment

**Static, precomputed, and impossible to break:**

1. **Leaderboard:** variants × four metrics, color-coded
2. **Drill-down:** per question, with retrieved passages, claims marked supported or unsupported, and citation verdicts
3. **Trust table:** the meta-evaluation results
4. **Live playground:** the one part that is allowed to degrade. Type a prompt and run it through the pipeline plus
   evaluator.

Deploy tabs 1 to 3 on static hosting (Vercel, Netlify, or GitHub Pages) from the report JSON. They need no server and
stay up for the required week.

The live playground:

- API key stays server-side only, never in frontend code
- Per-IP rate limit, input length cap, and a hard spend limit set in the provider dashboard
- If the quota runs out or the call fails, it shows "live mode is rate-limited, showing precomputed examples" and the
  rest of the site keeps working

Use plain HTML/JS or a Next.js static export. Avoid Streamlit for the deployed site, since free tiers tend to sleep apps
after inactivity.

## 8. Testing and evidence

- Unit tests for every metric against hand-computed values
- Fault-injection tests that fail if a poisoned variant's expected metric doesn't drop
- Judge cache tests (hit, miss, strict mode)
- Judge-stability test on repeated runs
- CI badge, coverage report, and a committed benchmark table in the README

## 9. Documentation and demo

**README order:** one-line pitch, results table, 3-command quickstart (`pip install -e .`, `rag-eval run`,
`rag-eval report`), architecture diagram, metric definitions, how to plug in your own RAG, judge modes, limitations,
contributing, license.

**Demo video (3 minutes):** a confident wrong answer, then rag-eval catches it with numbers, then the Poison Lab, then
the CI failure, then the trust table. Finish with the live link.

Test the quickstart on a clean machine or fresh container before submitting. If a judge can't run it, the other work
doesn't count.

## 10. Build order (cut from the bottom)

| Priority | Deliverable                                                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------ |
| **P0**   | Schema, retrieval metrics, 50-question golden set (about 10 unanswerable), Baseline RAG, CLI skeleton, JSON report |
| **P0**   | NLI faithfulness and citation metrics, judge cache with `none/replay/live`, API correctness judge                  |
| **P1**   | The other 3 variants, benchmark run, trust table, unit and fault-injection tests                                   |
| **P1**   | `compare` with `--fail-under`, GitHub Action, failing-PR screenshot, README                                        |
| **P1**   | Static dashboard (tabs 1 to 3) deployed, then demo video                                                           |
| **P2**   | Live playground with fallback, Ollama flag, guard mode (rerank/filter/abstain) with measured deltas, PyPI          |

Guard mode was in your original idea, and it only earns its place as a measured result, for example "unsupported rate
31% → 9%, recall unchanged." If time is tight, cut it first.

## Rubric mapping

| Criterion                           | Weight | What earns the marks                                                                 |
| ----------------------------------- | ------ | ------------------------------------------------------------------------------------ |
| Engineering quality and correctness | 25%    | Clean `evaluate()` API, deterministic runs, manifest hashes, replay cache            |
| Open-source contribution value      | 25%    | Pluggable package, CI gate with Action template, clear contracts, contributing guide |
| Usefulness and originality          | 20%    | Fault injection plus evaluating the evaluator                                        |
| Testing and evidence                | 15%    | Metric tests, fault tests, benchmark and trust tables                                |
| Demo and documentation              | 15%    | Quickstart, video, live link, honest limitations                                     |
| Bonus                               | n/a    | Static site live for at least a week                                                 |

## Biggest risks to watch

1. **Poisoned variants not moving the expected metric.** Test this early, before building anything else on top.
2. **NLI noise on short or paraphrased claims.** Tune the claim-splitting and thresholds on the Baseline, and report the
   noise honestly.
3. **Scope creep into P2.** A solid evaluator with proof beats a flashy playground.

I can save this as `PLAN.md` in a repo, or scaffold the repo now with the schema, the judge modes and cache layer, and
the first retrieval metrics. Which do you want?
