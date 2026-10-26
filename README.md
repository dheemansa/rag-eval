# rag-eval

**pytest for RAG** — score any RAG pipeline on retrieval relevance, answer correctness, citation accuracy, and unsupported answers.

A RAG demo can sound right while retrieving the wrong passages. `rag-eval` catches that with deterministic metrics, a CI compare gate, and a Poison Lab that proves the evaluator notices deliberately broken pipelines.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the baseline Poison Lab variant (no API key needed)
rag-eval run \
  --dataset demo/data/golden.jsonl \
  --rag demo.baseline:rag_fn \
  --judge none \
  -o reports/baseline.json

rag-eval report reports/baseline.json
```

Compare a candidate against baseline and fail CI if metrics drop:

```bash
rag-eval compare reports/baseline.json reports/candidate.json \
  --fail-under recall@5=0.8 \
  --fail-under token_f1=0.7
```

## Plug in your RAG

```python
from rag_eval import RAGResult, Context, evaluate, EvalConfig

def my_rag(question: str) -> RAGResult:
    # call your pipeline, then map to the schema
    return RAGResult(
        question=question,
        answer="...",
        contexts=[Context(id="doc-1", text="...")],
        citations=["doc-1"],
    )

report = evaluate(my_rag, dataset="path/to/golden.jsonl", config=EvalConfig(judge_mode="none"))
print(report.aggregate_metrics)
```

## Metrics

| Family | Metrics | Needs API key? |
| --- | --- | --- |
| Retrieval | hit@k, recall@k, precision@k, MRR | No |
| Faithfulness | faithfulness, unsupported_claim_rate, abstention_accuracy | NLI model (optional) |
| Citations | citation_validity, citation_support, uncited_claim_rate | NLI for support |
| Correctness | token_f1, judge_correctness | Judge part only |

## Judge modes

| Mode | Behavior |
| --- | --- |
| `none` | Keyless metrics only (retrieval, token F1, citation validity, abstention) |
| `replay` | Use cached NLI/API verdicts; cache miss → `not_evaluated` (`--strict` fails) |
| `live` | Call NLI / API judge and cache results |

## Poison Lab

Four variants over a synthetic DataKit docs corpus (~50 questions, ~10 unanswerable):

| Variant | Fault | Expected effect |
| --- | --- | --- |
| `demo.baseline:rag_fn` | none | Strong reference scores |
| `demo.bad_retriever:rag_fn` | weak retrieval | recall / MRR / precision drop |
| `demo.hallucinator:rag_fn` | invents answers + fake citations | citation_validity & abstention drop |
| `demo.stale_data:rag_fn` | outdated docs | token_f1 drops while answers stay grounded |

## Tests

```bash
source .venv/bin/activate
pytest
```

## License

MIT
