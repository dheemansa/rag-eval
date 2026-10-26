"""Retrieval quality metrics.

Measures how well the retriever surfaces relevant passages compared to
the golden passage IDs.  All metrics operate on the top-k retrieved
contexts.

Metrics returned
----------------
- ``hit@k``      – 1.0 if *any* gold passage ID appears in the top-k, else 0.0
- ``recall@k``   – |retrieved ∩ gold| / |gold|
- ``precision@k``– |retrieved ∩ gold| / k
- ``mrr``        – 1 / rank of the first gold passage in retrieved contexts
"""

from __future__ import annotations

from rag_eval.schema import GoldItem, MetricResult, RAGResult


def compute(
    result: RAGResult,
    gold: GoldItem,
    *,
    k: int = 5,
) -> dict[str, MetricResult]:
    """Compute retrieval metrics for a single question.

    Parameters
    ----------
    result:
        The RAG pipeline output.
    gold:
        The golden-set item with reference passage IDs.
    k:
        Cutoff for top-k evaluation.

    Returns
    -------
    dict[str, MetricResult]
        Mapping of metric names to their computed values.
    """
    gold_ids: set[str] = set(gold.gold_passage_ids)
    retrieved_ids: list[str] = [ctx.id for ctx in result.contexts[:k]]
    retrieved_set: set[str] = set(retrieved_ids)

    # -- hit@k ---------------------------------------------------------------
    hit = 1.0 if gold_ids & retrieved_set else 0.0

    # -- recall@k -------------------------------------------------------------
    if len(gold_ids) == 0:
        recall = 0.0
    else:
        recall = len(gold_ids & retrieved_set) / len(gold_ids)

    # -- precision@k ----------------------------------------------------------
    if k == 0:
        precision = 0.0
    else:
        precision = len(gold_ids & retrieved_set) / k

    # -- mrr ------------------------------------------------------------------
    mrr = 0.0
    for rank, ctx_id in enumerate(retrieved_ids, start=1):
        if ctx_id in gold_ids:
            mrr = 1.0 / rank
            break

    return {
        f"hit@{k}": MetricResult(
            name=f"hit@{k}",
            value=hit,
            details={"k": k},
        ),
        f"recall@{k}": MetricResult(
            name=f"recall@{k}",
            value=recall,
            details={"k": k, "gold_count": len(gold_ids)},
        ),
        f"precision@{k}": MetricResult(
            name=f"precision@{k}",
            value=precision,
            details={"k": k},
        ),
        "mrr": MetricResult(
            name="mrr",
            value=mrr,
            details={"first_relevant_rank": int(1 / mrr) if mrr > 0 else -1},
        ),
    }
