"""Poison Lab demo: four RAG variants over a small DataKit docs corpus."""

from demo.baseline import rag_fn as baseline_rag
from demo.bad_retriever import rag_fn as bad_retriever_rag
from demo.hallucinator import rag_fn as hallucinator_rag
from demo.stale_data import rag_fn as stale_data_rag

VARIANTS = {
    "baseline": baseline_rag,
    "bad_retriever": bad_retriever_rag,
    "hallucinator": hallucinator_rag,
    "stale_data": stale_data_rag,
}

__all__ = [
    "VARIANTS",
    "baseline_rag",
    "bad_retriever_rag",
    "hallucinator_rag",
    "stale_data_rag",
]
