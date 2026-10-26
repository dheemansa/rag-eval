"""Unit tests for faithfulness metrics and claim splitting."""

from __future__ import annotations

from rag_eval.metrics.faithfulness import compute, split_claims
from rag_eval.schema import (
    Context,
    GoldItem,
    JudgeVerdict,
    QuestionType,
    RAGResult,
)


def _result(answer: str) -> RAGResult:
    return RAGResult(
        question="q",
        answer=answer,
        contexts=[Context(id="c1", text="ctx")],
        citations=["c1"],
    )


def _gold(qtype: QuestionType = QuestionType.SINGLE_HOP) -> GoldItem:
    return GoldItem(
        question="q",
        gold_answer="a",
        gold_passage_ids=["c1"],
        question_type=qtype,
    )


def test_split_claims_basic():
    claims = split_claims("First claim. Second claim! Third?")
    assert claims == ["First claim.", "Second claim!", "Third?"]


def test_split_claims_filters_short():
    assert split_claims("Ok. A longer sentence here.") == ["A longer sentence here."]


def test_faithfulness_all_entailed():
    verdicts = [
        JudgeVerdict(claim="c1", label="entailment", score=0.9),
        JudgeVerdict(claim="c2", label="entailment", score=0.8),
    ]
    metrics = compute(_result("A. B."), _gold(), verdicts=verdicts, threshold=0.5)
    assert metrics["faithfulness"].value == 1.0
    assert metrics["unsupported_claim_rate"].value == 0.0


def test_faithfulness_partial():
    verdicts = [
        JudgeVerdict(claim="c1", label="entailment", score=0.9),
        JudgeVerdict(claim="c2", label="neutral", score=0.7),
        JudgeVerdict(claim="c3", label="contradiction", score=0.6),
    ]
    metrics = compute(_result("A. B. C."), _gold(), verdicts=verdicts)
    assert metrics["faithfulness"].value == 1 / 3
    assert abs(metrics["unsupported_claim_rate"].value - 2 / 3) < 1e-9


def test_faithfulness_not_evaluated():
    metrics = compute(_result("A sentence here."), _gold(), verdicts=None)
    assert metrics["faithfulness"].value == -1.0
    assert metrics["faithfulness"].details["status"] == "not_evaluated"


def test_faithfulness_cache_miss_verdicts():
    verdicts = [
        JudgeVerdict(claim="c1", label="not_evaluated", score=0.0),
        JudgeVerdict(claim="c2", label="not_evaluated", score=0.0),
    ]
    metrics = compute(_result("A. B."), _gold(), verdicts=verdicts)
    assert metrics["faithfulness"].value == -1.0


def test_abstention_unanswerable_correct():
    metrics = compute(
        _result("I don't know. Not enough information."),
        _gold(QuestionType.UNANSWERABLE),
        verdicts=[],
    )
    assert metrics["abstention_accuracy"].value == 1.0


def test_abstention_unanswerable_fails():
    metrics = compute(
        _result("The answer is forty-two."),
        _gold(QuestionType.UNANSWERABLE),
        verdicts=[],
    )
    assert metrics["abstention_accuracy"].value == 0.0
