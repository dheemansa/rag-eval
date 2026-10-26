"""Judge cache hit / miss / strict-mode tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval.judges.base import BaseJudge
from rag_eval.judges.cache import CachedJudge, CacheMissError, JudgeCache
from rag_eval.schema import JudgeVerdict


class StubJudge(BaseJudge):
    def __init__(self) -> None:
        self.calls = 0

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(claim=claim, label="entailment", score=0.99)

    def judge_answer(
        self, question: str, predicted: str, reference: str
    ) -> JudgeVerdict:
        self.calls += 1
        return JudgeVerdict(claim=predicted, label="correct", score=1.0)


def test_cache_put_get(tmp_path: Path):
    cache = JudgeCache(tmp_path)
    key = cache.make_key("a", "b")
    verdict = JudgeVerdict(claim="c", label="entailment", score=0.5)
    cache.put(key, {"claim": "c"}, verdict)
    loaded = cache.get(key)
    assert loaded is not None
    assert loaded.label == "entailment"
    assert loaded.score == 0.5


def test_live_mode_caches(tmp_path: Path):
    inner = StubJudge()
    cache = JudgeCache(tmp_path)
    judge = CachedJudge(inner=inner, cache=cache, mode="live")

    v1 = judge.judge_claim("claim", "context")
    v2 = judge.judge_claim("claim", "context")
    assert v1.label == "entailment"
    assert v2.label == "entailment"
    assert inner.calls == 1  # second call served from cache


def test_replay_miss_soft(tmp_path: Path):
    cache = JudgeCache(tmp_path)
    judge = CachedJudge(inner=None, cache=cache, mode="replay", strict=False)
    v = judge.judge_claim("missing", "ctx")
    assert v.label == "not_evaluated"


def test_replay_miss_strict(tmp_path: Path):
    cache = JudgeCache(tmp_path)
    judge = CachedJudge(inner=None, cache=cache, mode="replay", strict=True)
    with pytest.raises(CacheMissError):
        judge.judge_claim("missing", "ctx")


def test_replay_hit(tmp_path: Path):
    cache = JudgeCache(tmp_path)
    # Seed cache as if a live run already wrote it
    live = CachedJudge(
        inner=StubJudge(), cache=cache, mode="live", judge_name="StubJudge"
    )
    live.judge_claim("claim", "context")

    replay = CachedJudge(
        inner=None, cache=cache, mode="replay", strict=True, judge_name="StubJudge"
    )
    v = replay.judge_claim("claim", "context")
    assert v.label == "entailment"
