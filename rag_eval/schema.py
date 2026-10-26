"""Core data models for rag-eval.

Every component in the package operates on these types. A RAG pipeline plugs in
by returning a ``RAGResult``; the golden evaluation dataset is a list of
``GoldItem``; and the output of ``evaluate()`` is a ``Report``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pipeline I/O
# ---------------------------------------------------------------------------


class Context(BaseModel):
    """A single retrieved passage."""

    id: str
    text: str


class RAGResult(BaseModel):
    """Output from a RAG pipeline for a single question.

    Any RAG system can be evaluated by mapping its output to this schema.
    """

    question: str
    answer: str
    contexts: list[Context] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------


class QuestionType(str, Enum):
    """Classification of a golden-set question."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class GoldItem(BaseModel):
    """One row in the golden evaluation dataset (versioned JSONL)."""

    question: str
    gold_answer: str
    gold_passage_ids: list[str] = Field(default_factory=list)
    question_type: QuestionType = QuestionType.SINGLE_HOP
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


class MetricResult(BaseModel):
    """The result of computing a single metric for one question."""

    name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)


class QuestionReport(BaseModel):
    """All evaluation results for a single question."""

    question: str
    question_type: QuestionType
    rag_result: RAGResult
    gold_item: GoldItem
    metrics: dict[str, MetricResult] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Judge verdict (used by faithfulness / citation / correctness judges)
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    """Verdict from an NLI or LLM judge on a single claim or answer."""

    claim: str
    label: str  # "entailment" | "contradiction" | "neutral" | "not_evaluated"
    score: float = 0.0  # confidence or probability
    reason: str = ""


# ---------------------------------------------------------------------------
# Configuration and manifest
# ---------------------------------------------------------------------------


class EvalConfig(BaseModel):
    """Configuration for an evaluation run."""

    judge_mode: str = "replay"  # "none" | "replay" | "live"
    judge_model: str = ""  # LLM model name for live API judge
    nli_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    seed: int = 42
    k: int = 5  # cutoff for retrieval metrics
    faithfulness_threshold: float = 0.5  # NLI score above which a claim is "supported"
    cache_dir: str = ".rag_eval_cache"
    strict: bool = False  # fail on cache miss in replay mode


class RunManifest(BaseModel):
    """Metadata recorded with every evaluation run for reproducibility."""

    config: EvalConfig = Field(default_factory=EvalConfig)
    config_hash: str = ""
    timestamp: str = ""
    rag_eval_version: str = ""
    python_version: str = ""
    nli_model_hash: str = ""
    judge_model: str = ""


# ---------------------------------------------------------------------------
# Top-level report
# ---------------------------------------------------------------------------


class Report(BaseModel):
    """Full evaluation report produced by ``evaluate()``."""

    question_reports: list[QuestionReport] = Field(default_factory=list)
    aggregate_metrics: dict[str, float] = Field(default_factory=dict)
    manifest: RunManifest = Field(default_factory=RunManifest)
