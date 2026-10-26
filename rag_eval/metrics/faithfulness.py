"""Faithfulness metrics.

Measures whether the generated answer is faithful to the retrieved
context, using pre-computed NLI verdicts at the claim level.

Metrics returned
----------------
- ``faithfulness``          – fraction of claims labeled "entailment" with
                              score ≥ *threshold*
- ``unsupported_claim_rate``– 1 − faithfulness
- ``abstention_accuracy``   – for unanswerable questions, 1.0 if the answer
                              indicates abstention, else 0.0
"""

from __future__ import annotations

import re

from rag_eval.schema import GoldItem, JudgeVerdict, MetricResult, QuestionType, RAGResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:\s+|$)")

_ABSTENTION_PHRASES: list[str] = [
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not enough information",
    "no information",
    "i'm not sure",
    "i am not sure",
    "cannot be determined",
    "unanswerable",
    "no answer",
    "not answerable",
]


def split_claims(text: str) -> list[str]:
    """Split *text* into individual claims using sentence boundaries.

    This is a simple MVP splitter: it splits on ``.``, ``!``, or ``?``
    followed by whitespace or end-of-string, then filters out very short
    fragments (< 5 characters).

    Parameters
    ----------
    text:
        The answer text to split.

    Returns
    -------
    list[str]
        A list of claim strings.
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]


def _is_abstention(answer: str) -> bool:
    """Return True if *answer* looks like an abstention."""
    lower = answer.lower()
    return any(phrase in lower for phrase in _ABSTENTION_PHRASES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute(
    result: RAGResult,
    gold: GoldItem,
    *,
    verdicts: list[JudgeVerdict] | None = None,
    threshold: float = 0.5,
) -> dict[str, MetricResult]:
    """Compute faithfulness metrics for a single question.

    Parameters
    ----------
    result:
        The RAG pipeline output.
    gold:
        The golden-set item.
    verdicts:
        Pre-computed NLI verdicts (one per claim).  If ``None`` the
        judge-dependent metrics are returned with ``value=-1``.
    threshold:
        Minimum NLI score to consider a claim as entailed.

    Returns
    -------
    dict[str, MetricResult]
        Mapping of metric names to their computed values.
    """
    metrics: dict[str, MetricResult] = {}

    # -- faithfulness & unsupported_claim_rate --------------------------------
    evaluated = (
        None
        if verdicts is None
        else [v for v in verdicts if v.label != "not_evaluated"]
    )
    if evaluated is None or (verdicts is not None and len(evaluated) == 0 and len(verdicts) > 0):
        # No verdicts, or every verdict was a cache miss / not evaluated
        metrics["faithfulness"] = MetricResult(
            name="faithfulness",
            value=-1.0,
            details={"status": "not_evaluated"},
        )
        metrics["unsupported_claim_rate"] = MetricResult(
            name="unsupported_claim_rate",
            value=-1.0,
            details={"status": "not_evaluated"},
        )
    elif evaluated is None or len(evaluated) == 0:
        # Empty answer → no claims
        metrics["faithfulness"] = MetricResult(
            name="faithfulness",
            value=0.0,
            details={"threshold": threshold, "total_claims": 0, "entailed_claims": 0},
        )
        metrics["unsupported_claim_rate"] = MetricResult(
            name="unsupported_claim_rate",
            value=0.0,
            details={"threshold": threshold},
        )
    else:
        entailed = sum(
            1 for v in evaluated if v.label == "entailment" and v.score >= threshold
        )
        faithfulness = entailed / len(evaluated)

        metrics["faithfulness"] = MetricResult(
            name="faithfulness",
            value=faithfulness,
            details={
                "threshold": threshold,
                "total_claims": len(evaluated),
                "entailed_claims": entailed,
            },
        )
        metrics["unsupported_claim_rate"] = MetricResult(
            name="unsupported_claim_rate",
            value=1.0 - faithfulness,
            details={"threshold": threshold},
        )

    # -- abstention_accuracy --------------------------------------------------
    if gold.question_type == QuestionType.UNANSWERABLE:
        abstained = _is_abstention(result.answer)
        metrics["abstention_accuracy"] = MetricResult(
            name="abstention_accuracy",
            value=1.0 if abstained else 0.0,
            details={"abstained": abstained},
        )
    else:
        # For answerable questions the metric is trivially 1.0 if the system
        # did *not* abstain (it attempted to answer).
        abstained = _is_abstention(result.answer)
        metrics["abstention_accuracy"] = MetricResult(
            name="abstention_accuracy",
            value=0.0 if abstained else 1.0,
            details={"abstained": abstained, "question_type": gold.question_type.value},
        )

    return metrics
