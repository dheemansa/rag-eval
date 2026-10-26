"""Integration tests for evaluate() and dataset loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.report import compare_reports, load_report, save_report
from rag_eval.runner import evaluate, load_dataset
from rag_eval.schema import Context, EvalConfig, GoldItem, QuestionType, RAGResult


def _tiny_dataset() -> list[GoldItem]:
    return [
        GoldItem(
            question="What is X?",
            gold_answer="X is one.",
            gold_passage_ids=["p1"],
            question_type=QuestionType.SINGLE_HOP,
        ),
        GoldItem(
            question="What is Y?",
            gold_answer="I don't know.",
            gold_passage_ids=[],
            question_type=QuestionType.UNANSWERABLE,
        ),
    ]


def _good_rag(question: str) -> RAGResult:
    if "Y" in question:
        return RAGResult(
            question=question,
            answer="I don't know. Not enough information.",
            contexts=[],
            citations=[],
        )
    return RAGResult(
        question=question,
        answer="X is one.",
        contexts=[Context(id="p1", text="X is one.")],
        citations=["p1"],
    )


def test_evaluate_judge_none():
    report = evaluate(
        _good_rag,
        gold_items=_tiny_dataset(),
        config=EvalConfig(judge_mode="none", k=5),
    )
    assert len(report.question_reports) == 2
    assert "recall@5" in report.aggregate_metrics
    assert "token_f1" in report.aggregate_metrics
    # Judge-dependent metrics should be excluded / not_evaluated
    assert report.aggregate_metrics.get("faithfulness", -1) == -1.0 or "faithfulness" not in report.aggregate_metrics
    assert report.manifest.config_hash


def test_evaluate_replay_without_cache(tmp_path: Path):
    report = evaluate(
        _good_rag,
        gold_items=_tiny_dataset(),
        config=EvalConfig(judge_mode="replay", cache_dir=str(tmp_path), strict=False),
    )
    # Retrieval + token_f1 still computed
    assert report.aggregate_metrics["hit@5"] == 0.5  # only first item has a gold passage hit
    assert report.aggregate_metrics["token_f1"] > 0.5


def test_load_dataset_roundtrip(tmp_path: Path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        '{"question":"q","gold_answer":"a","gold_passage_ids":["p"],"question_type":"single_hop"}\n',
        encoding="utf-8",
    )
    items = load_dataset(path)
    assert len(items) == 1
    assert items[0].question == "q"


def test_load_dataset_bad_line(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_save_load_compare_report(tmp_path: Path):
    report = evaluate(
        _good_rag,
        gold_items=_tiny_dataset(),
        config=EvalConfig(judge_mode="none"),
    )
    path = tmp_path / "report.json"
    save_report(report, path)
    loaded = load_report(path)
    assert loaded.aggregate_metrics == report.aggregate_metrics

    # Degraded candidate
    def bad_rag(question: str) -> RAGResult:
        return RAGResult(
            question=question,
            answer="wrong",
            contexts=[Context(id="zzz", text="nope")],
            citations=["zzz"],
        )

    degraded = evaluate(
        bad_rag,
        gold_items=_tiny_dataset(),
        config=EvalConfig(judge_mode="none"),
    )
    diff = compare_reports(report, degraded)
    assert "token_f1" in diff
    assert diff["token_f1"]["delta"] < 0
