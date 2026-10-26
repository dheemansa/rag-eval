"""rag-eval: pytest for RAG – score any RAG pipeline on retrieval, faithfulness, citations, and correctness."""

from rag_eval.runner import evaluate, load_dataset
from rag_eval.schema import (
    Context,
    EvalConfig,
    GoldItem,
    QuestionType,
    RAGResult,
    Report,
)

__version__ = "0.1.0"

__all__ = [
    "Context",
    "EvalConfig",
    "GoldItem",
    "QuestionType",
    "RAGResult",
    "Report",
    "evaluate",
    "load_dataset",
    "__version__",
]
