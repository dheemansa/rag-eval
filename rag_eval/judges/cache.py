"""Caching layer for judge verdicts."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from rag_eval.schema import JudgeVerdict

from .base import BaseJudge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CacheMissError(Exception):
    """Raised in strict replay mode when a verdict is not in the cache."""


# ---------------------------------------------------------------------------
# Cache storage
# ---------------------------------------------------------------------------


class JudgeCache:
    """Persistent on-disk cache of :class:`JudgeVerdict` objects.

    Each verdict is stored as a JSON file named by its SHA-256 key hash inside
    ``<cache_dir>/judge_cache/``.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self._dir = Path(cache_dir) / "judge_cache"
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def make_key(*parts: str) -> str:
        """Return a SHA-256 hex digest for the given input parts."""
        raw = "\n".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key_hash: str) -> Path:
        return self._dir / f"{key_hash}.json"

    # -- public API ---------------------------------------------------------

    def get(self, key_hash: str) -> JudgeVerdict | None:
        """Return a cached verdict, or ``None`` on miss."""
        path = self._path(key_hash)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return JudgeVerdict(**data["verdict"])
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt cache entry %s – ignoring.", key_hash)
            return None

    def put(
        self, key_hash: str, inputs: dict, verdict: JudgeVerdict
    ) -> None:
        """Persist a verdict to disk."""
        payload = {
            "key_hash": key_hash,
            "inputs": inputs,
            "verdict": verdict.model_dump(),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        self._path(key_hash).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CachedJudge wrapper
# ---------------------------------------------------------------------------


class CachedJudge(BaseJudge):
    """Wraps an inner :class:`BaseJudge` with transparent caching.

    Parameters
    ----------
    inner:
        The real judge to delegate to on cache miss.  May be ``None`` in
        *replay* mode (when no live model is needed).
    cache:
        A :class:`JudgeCache` instance.
    mode:
        ``"live"`` – call *inner* on miss and cache the result.
        ``"replay"`` – never call *inner*; return *not_evaluated* or raise.
    strict:
        Only meaningful when *mode* is ``"replay"``.  If ``True``, raise
        :class:`CacheMissError` instead of returning a soft miss verdict.
    """

    def __init__(
        self,
        inner: BaseJudge | None,
        cache: JudgeCache,
        mode: str = "replay",
        strict: bool = False,
        judge_name: str | None = None,
    ) -> None:
        if mode not in ("live", "replay"):
            raise ValueError(f"mode must be 'live' or 'replay', got {mode!r}")
        if mode == "live" and inner is None:
            raise ValueError("An inner judge is required in live mode.")
        self._inner = inner
        self._cache = cache
        self._mode = mode
        self._strict = strict
        # Stable cache namespace so replay (inner=None) hits entries written in live mode.
        if judge_name is not None:
            self._judge_name = judge_name
        elif inner is not None:
            self._judge_name = type(inner).__name__
        else:
            self._judge_name = "NLIJudge"

    # -- helpers ------------------------------------------------------------

    def _judge_class_name(self) -> str:
        return self._judge_name

    def _on_miss(self, claim_text: str) -> JudgeVerdict:
        """Handle a cache miss in replay mode."""
        if self._strict:
            raise CacheMissError(
                f"No cached verdict for claim: {claim_text!r}"
            )
        logger.warning("Cache miss (replay mode) for claim: %s", claim_text)
        return JudgeVerdict(
            claim=claim_text,
            label="not_evaluated",
            score=0.0,
            reason="Cache miss in replay mode.",
        )

    # -- BaseJudge implementation -------------------------------------------

    def judge_claim(self, claim: str, context: str) -> JudgeVerdict:
        key = self._cache.make_key(self._judge_class_name(), claim, context)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._mode == "replay":
            return self._on_miss(claim)

        # live mode – delegate to inner judge
        assert self._inner is not None  # guaranteed by __init__
        verdict = self._inner.judge_claim(claim, context)
        self._cache.put(
            key,
            {"judge": self._judge_class_name(), "claim": claim, "context": context},
            verdict,
        )
        return verdict

    def judge_answer(
        self, question: str, predicted: str, reference: str
    ) -> JudgeVerdict:
        key = self._cache.make_key(
            self._judge_class_name(), question, predicted, reference
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if self._mode == "replay":
            return self._on_miss(predicted)

        assert self._inner is not None
        verdict = self._inner.judge_answer(question, predicted, reference)
        self._cache.put(
            key,
            {
                "judge": self._judge_class_name(),
                "question": question,
                "predicted": predicted,
                "reference": reference,
            },
            verdict,
        )
        return verdict
