"""Unit tests for answer correctness metrics."""

from __future__ import annotations

from rag_eval.metrics.correctness import compute
from rag_eval.schema import (
    Context,
    GoldItem,
    JudgeVerdict,
    QuestionType,
    RAGResult,
)


def _pair(pred: str, gold: str) -> tuple[RAGResult, GoldItem]:
    result = RAGResult(
        question="q",
        answer=pred,
        contexts=[Context(id="c1", text="t")],
        citations=[],
    )
    item = GoldItem(
        question="q",
        gold_answer=gold,
        gold_passage_ids=["c1"],
        question_type=QuestionType.SINGLE_HOP,
    )
    return result, item


def test_token_f1_exact_match():
    result, gold = _pair("hello world", "hello world")
    metrics = compute(result, gold)
    assert metrics["token_f1"].value == 1.0


def test_token_f1_partial():
    # pred={hello, world}, gold={hello, there} → overlap=1
    # P=1/2, R=1/2, F1=0.5
    result, gold = _pair("hello world", "hello there")
    metrics = compute(result, gold)
    assert metrics["token_f1"].value == 0.5


def test_token_f1_no_overlap():
    result, gold = _pair("foo bar", "baz qux")
    metrics = compute(result, gold)
    assert metrics["token_f1"].value == 0.0


def test_token_f1_punctuation_ignored():
    result, gold = _pair("Hello, World!", "hello world")
    metrics = compute(result, gold)
    assert metrics["token_f1"].value == 1.0


def test_judge_correctness_not_evaluated():
    result, gold = _pair("a", "a")
    metrics = compute(result, gold, judge_verdict=None)
    assert metrics["judge_correctness"].value == -1.0


def test_judge_correctness_label():
    result, gold = _pair("a", "a")
    verdict = JudgeVerdict(claim="a", label="correct", score=0.9, reason="ok")
    metrics = compute(result, gold, judge_verdict=verdict)
    assert metrics["judge_correctness"].value == 0.9

    verdict2 = JudgeVerdict(claim="a", label="incorrect", score=0.8)
    metrics2 = compute(result, gold, judge_verdict=verdict2)
    assert metrics2["judge_correctness"].value == 0.0
