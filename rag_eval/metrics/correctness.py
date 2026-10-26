"""Answer correctness metrics.

Measures the quality of the generated answer against the gold answer
using token-level overlap and an optional LLM judge verdict.

Metrics returned
----------------
- ``token_f1``          – token-level F1 between predicted and gold answer
- ``judge_correctness`` – the judge's verdict score (from API judge)
"""

from __future__ import annotations

import string

from rag_eval.schema import GoldItem, JudgeVerdict, MetricResult, RAGResult

# Pre-compiled pattern for removing punctuation.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, and split on whitespace."""
    return text.lower().translate(_PUNCT_TABLE).split()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute(
    result: RAGResult,
    gold: GoldItem,
    *,
    judge_verdict: JudgeVerdict | None = None,
) -> dict[str, MetricResult]:
    """Compute answer-correctness metrics for a single question.

    Parameters
    ----------
    result:
        The RAG pipeline output.
    gold:
        The golden-set item with the reference answer.
    judge_verdict:
        Optional verdict from an LLM judge.  If ``None`` the
        ``judge_correctness`` metric is returned with ``value=-1``.

    Returns
    -------
    dict[str, MetricResult]
        Mapping of metric names to their computed values.
    """
    metrics: dict[str, MetricResult] = {}

    # -- token_f1 -------------------------------------------------------------
    pred_tokens = _tokenize(result.answer)
    gold_tokens = _tokenize(gold.gold_answer)

    pred_bag = _count_tokens(pred_tokens)
    gold_bag = _count_tokens(gold_tokens)

    # Number of overlapping tokens (using multiset intersection).
    overlap = sum((min(pred_bag.get(t, 0), gold_bag.get(t, 0)) for t in gold_bag))

    if overlap == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    else:
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(gold_tokens) if gold_tokens else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["token_f1"] = MetricResult(
        name="token_f1",
        value=f1,
        details={
            "precision": precision,
            "recall": recall,
            "overlap": overlap,
            "pred_length": len(pred_tokens),
            "gold_length": len(gold_tokens),
        },
    )

    # -- judge_correctness ----------------------------------------------------
    if judge_verdict is None or judge_verdict.label == "not_evaluated":
        metrics["judge_correctness"] = MetricResult(
            name="judge_correctness",
            value=-1.0,
            details={"status": "not_evaluated"},
        )
    else:
        if judge_verdict.label == "correct":
            score = judge_verdict.score if judge_verdict.score > 0 else 1.0
        elif judge_verdict.label == "incorrect":
            score = 0.0
        else:
            score = judge_verdict.score

        metrics["judge_correctness"] = MetricResult(
            name="judge_correctness",
            value=score,
            details={
                "label": judge_verdict.label,
                "reason": judge_verdict.reason,
            },
        )

    return metrics


def _count_tokens(tokens: list[str]) -> dict[str, int]:
    """Return a token → count mapping (bag of words)."""
    bag: dict[str, int] = {}
    for t in tokens:
        bag[t] = bag.get(t, 0) + 1
    return bag
