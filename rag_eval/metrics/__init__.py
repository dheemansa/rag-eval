"""Evaluation metrics for RAG pipelines.

Each metric module exposes a ``compute()`` function with the signature::

    def compute(
        result: RAGResult,
        gold: GoldItem,
        **kwargs,
    ) -> dict[str, MetricResult]:
        ...

The returned dict maps metric names (e.g. ``"recall@5"``) to ``MetricResult``
objects.
"""

from rag_eval.metrics.retrieval import compute as compute_retrieval
from rag_eval.metrics.faithfulness import compute as compute_faithfulness
from rag_eval.metrics.citations import compute as compute_citations
from rag_eval.metrics.correctness import compute as compute_correctness

__all__ = [
    "compute_retrieval",
    "compute_faithfulness",
    "compute_citations",
    "compute_correctness",
]
