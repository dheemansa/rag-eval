"""Fault-injection tests: poisoned variants must move the expected metrics."""

from __future__ import annotations

import pytest

from demo.bad_retriever import rag_fn as bad_retriever
from demo.baseline import rag_fn as baseline
from demo.corpus import GOLDEN_PATH
from demo.hallucinator import rag_fn as hallucinator
from demo.stale_data import rag_fn as stale_data
from rag_eval.runner import evaluate, load_dataset
from rag_eval.schema import EvalConfig


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(GOLDEN_PATH)


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    cache = tmp_path_factory.mktemp("cache")
    return EvalConfig(judge_mode="none", k=5, seed=42, cache_dir=str(cache))


@pytest.fixture(scope="module")
def baseline_report(dataset, config):
    return evaluate(baseline, gold_items=dataset, config=config)


def test_golden_dataset_size(dataset):
    assert len(dataset) >= 50
    unanswerable = [i for i in dataset if i.question_type.value == "unanswerable"]
    assert len(unanswerable) >= 10


def test_baseline_retrieval_is_strong(baseline_report):
    recall = baseline_report.aggregate_metrics["recall@5"]
    hit = baseline_report.aggregate_metrics["hit@5"]
    assert recall >= 0.7
    assert hit >= 0.7
    assert baseline_report.aggregate_metrics["token_f1"] >= 0.7


def test_bad_retriever_drops_retrieval(dataset, config, baseline_report):
    report = evaluate(bad_retriever, gold_items=dataset, config=config)
    assert report.aggregate_metrics["recall@5"] < baseline_report.aggregate_metrics["recall@5"] - 0.2
    assert report.aggregate_metrics["mrr"] < baseline_report.aggregate_metrics["mrr"] - 0.2
    assert report.aggregate_metrics["precision@5"] < baseline_report.aggregate_metrics["precision@5"]


def test_hallucinator_breaks_citations_and_abstention(dataset, config, baseline_report):
    report = evaluate(hallucinator, gold_items=dataset, config=config)
    assert report.aggregate_metrics["citation_validity"] < 0.1
    assert report.aggregate_metrics["citation_validity"] < baseline_report.aggregate_metrics["citation_validity"]
    assert report.aggregate_metrics["token_f1"] < baseline_report.aggregate_metrics["token_f1"] - 0.3
    # Unanswerables should not be abstained → abstention accuracy drops
    assert report.aggregate_metrics["abstention_accuracy"] < baseline_report.aggregate_metrics["abstention_accuracy"]


def test_stale_data_hurts_correctness(dataset, config, baseline_report):
    """Stale Data: answers stay grounded in retrieved text but correctness drops.

    Without NLI we use token_f1 as the correctness signal. Retrieval may still
    look reasonable because stale twins share vocabulary with the questions.
    """
    report = evaluate(stale_data, gold_items=dataset, config=config)
    assert report.aggregate_metrics["token_f1"] < baseline_report.aggregate_metrics["token_f1"] - 0.05

    # On questions that have a stale twin, token_f1 should be especially hurt
    # relative to baseline for the version question.
    version_q = "What is the current stable release of DataKit?"
    base_qr = next(qr for qr in baseline_report.question_reports if qr.question == version_q)
    stale_qr = next(qr for qr in report.question_reports if qr.question == version_q)
    assert stale_qr.metrics["token_f1"].value < base_qr.metrics["token_f1"].value
