"""Citation quality metrics.

Measures whether the generated answer properly cites retrieved passages
and whether those citations are supported by NLI evidence.

Metrics returned
----------------
- ``citation_validity``  – fraction of cited IDs that exist in retrieved
                           context IDs
- ``citation_support``   – fraction of cited passages where the NLI verdict
                           says the claim is entailed (requires verdicts)
- ``uncited_claim_rate`` – fraction of claims that have no citation at all
"""

from __future__ import annotations

from rag_eval.schema import GoldItem, JudgeVerdict, MetricResult, RAGResult

from rag_eval.metrics.faithfulness import split_claims


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
    """Compute citation metrics for a single question.

    Parameters
    ----------
    result:
        The RAG pipeline output.
    gold:
        The golden-set item.
    verdicts:
        Pre-computed NLI verdicts aligned to cited passages.  If ``None``
        the ``citation_support`` metric is returned with ``value=-1``.
    threshold:
        Minimum NLI score to consider a citation as supported.

    Returns
    -------
    dict[str, MetricResult]
        Mapping of metric names to their computed values.
    """
    metrics: dict[str, MetricResult] = {}

    cited_ids: list[str] = result.citations
    retrieved_ids: set[str] = {ctx.id for ctx in result.contexts}

    # -- citation_validity ----------------------------------------------------
    if len(cited_ids) == 0:
        validity = 0.0
    else:
        valid_count = sum(1 for cid in cited_ids if cid in retrieved_ids)
        validity = valid_count / len(cited_ids)

    metrics["citation_validity"] = MetricResult(
        name="citation_validity",
        value=validity,
        details={
            "cited_count": len(cited_ids),
            "valid_count": sum(1 for cid in cited_ids if cid in retrieved_ids),
        },
    )

    # -- citation_support -----------------------------------------------------
    evaluated = (
        None
        if verdicts is None
        else [v for v in verdicts if v.label != "not_evaluated"]
    )
    if evaluated is None or (verdicts is not None and len(evaluated) == 0 and len(verdicts) > 0):
        metrics["citation_support"] = MetricResult(
            name="citation_support",
            value=-1.0,
            details={"status": "not_evaluated"},
        )
    elif len(evaluated) == 0:
        metrics["citation_support"] = MetricResult(
            name="citation_support",
            value=0.0,
            details={"threshold": threshold, "total_verdicts": 0, "supported_count": 0},
        )
    else:
        supported = sum(
            1 for v in evaluated if v.label == "entailment" and v.score >= threshold
        )
        support = supported / len(evaluated)

        metrics["citation_support"] = MetricResult(
            name="citation_support",
            value=support,
            details={
                "threshold": threshold,
                "total_verdicts": len(evaluated),
                "supported_count": supported,
            },
        )

    # -- uncited_claim_rate ---------------------------------------------------
    claims = split_claims(result.answer)
    if len(claims) == 0:
        uncited_rate = 0.0
    else:
        # A claim is "cited" if there is at least one citation on the answer.
        # With the current schema (answer-level citation list, not per-claim),
        # we consider a claim uncited when the global citation list is empty.
        if len(cited_ids) == 0:
            uncited_rate = 1.0
        else:
            # All claims share the same citation list, so uncited_rate = 0.
            uncited_rate = 0.0

    metrics["uncited_claim_rate"] = MetricResult(
        name="uncited_claim_rate",
        value=uncited_rate,
        details={
            "total_claims": len(claims),
            "total_citations": len(cited_ids),
        },
    )

    return metrics
