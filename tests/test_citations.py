"""Unit tests for citation metrics."""

from __future__ import annotations

from rag_eval.metrics.citations import compute
from rag_eval.schema import (
    Context,
    GoldItem,
    JudgeVerdict,
    QuestionType,
    RAGResult,
)


def _gold() -> GoldItem:
    return GoldItem(
        question="q",
        gold_answer="a",
        gold_passage_ids=["c1"],
        question_type=QuestionType.SINGLE_HOP,
    )


def test_citation_validity_all_valid():
    result = RAGResult(
        question="q",
        answer="An answer sentence.",
        contexts=[Context(id="c1", text="t"), Context(id="c2", text="t")],
        citations=["c1", "c2"],
    )
    metrics = compute(result, _gold())
    assert metrics["citation_validity"].value == 1.0
    assert metrics["uncited_claim_rate"].value == 0.0


def test_citation_validity_fabricated():
    result = RAGResult(
        question="q",
        answer="An answer sentence.",
        contexts=[Context(id="c1", text="t")],
        citations=["fake-1", "fake-2"],
    )
    metrics = compute(result, _gold())
    assert metrics["citation_validity"].value == 0.0


def test_citation_validity_mixed():
    result = RAGResult(
        question="q",
        answer="An answer sentence.",
        contexts=[Context(id="c1", text="t")],
        citations=["c1", "fake"],
    )
    metrics = compute(result, _gold())
    assert metrics["citation_validity"].value == 0.5


def test_uncited_claim_rate_when_no_citations():
    result = RAGResult(
        question="q",
        answer="Claim one. Claim two.",
        contexts=[Context(id="c1", text="t")],
        citations=[],
    )
    metrics = compute(result, _gold())
    assert metrics["uncited_claim_rate"].value == 1.0


def test_citation_support_with_verdicts():
    result = RAGResult(
        question="q",
        answer="Claim one.",
        contexts=[Context(id="c1", text="t")],
        citations=["c1"],
    )
    verdicts = [
        JudgeVerdict(claim="Claim one.", label="entailment", score=0.95),
    ]
    metrics = compute(result, _gold(), verdicts=verdicts)
    assert metrics["citation_support"].value == 1.0
