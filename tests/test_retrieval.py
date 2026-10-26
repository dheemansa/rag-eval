"""Unit tests for retrieval metrics against hand-computed values."""

from __future__ import annotations

from rag_eval.metrics.retrieval import compute
from rag_eval.schema import Context, GoldItem, QuestionType, RAGResult


def _result(ids: list[str]) -> RAGResult:
    return RAGResult(
        question="q",
        answer="a",
        contexts=[Context(id=i, text=f"text-{i}") for i in ids],
        citations=[],
    )


def _gold(ids: list[str]) -> GoldItem:
    return GoldItem(
        question="q",
        gold_answer="a",
        gold_passage_ids=ids,
        question_type=QuestionType.SINGLE_HOP,
    )


def test_perfect_retrieval():
    metrics = compute(_result(["a", "b", "c"]), _gold(["a", "b"]), k=5)
    assert metrics["hit@5"].value == 1.0
    assert metrics["recall@5"].value == 1.0
    assert metrics["precision@5"].value == 2 / 5
    assert metrics["mrr"].value == 1.0


def test_miss_all():
    metrics = compute(_result(["x", "y", "z"]), _gold(["a", "b"]), k=5)
    assert metrics["hit@5"].value == 0.0
    assert metrics["recall@5"].value == 0.0
    assert metrics["precision@5"].value == 0.0
    assert metrics["mrr"].value == 0.0


def test_partial_recall_and_mrr():
    # gold={a,b}; retrieved=[x, a, y] → hit=1, recall=0.5, precision=1/5, mrr=1/2
    metrics = compute(_result(["x", "a", "y"]), _gold(["a", "b"]), k=5)
    assert metrics["hit@5"].value == 1.0
    assert metrics["recall@5"].value == 0.5
    assert metrics["precision@5"].value == 1 / 5
    assert metrics["mrr"].value == 0.5


def test_k_cutoff():
    # Relevant doc is at rank 4; with k=3 it is cut off
    metrics = compute(_result(["x", "y", "z", "a"]), _gold(["a"]), k=3)
    assert metrics["hit@3"].value == 0.0
    assert metrics["recall@3"].value == 0.0
    assert metrics["mrr"].value == 0.0


def test_empty_gold():
    metrics = compute(_result(["a"]), _gold([]), k=5)
    assert metrics["hit@5"].value == 0.0
    assert metrics["recall@5"].value == 0.0
