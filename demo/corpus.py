"""Corpus loading helpers for the Poison Lab demo."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"
GOLDEN_PATH = DATA_DIR / "golden.jsonl"


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    """Load all passages from corpus.json."""
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return list(data["passages"])


def current_passages() -> list[dict]:
    """Passages marked current (plus noise for distractors)."""
    return [p for p in load_corpus() if p["version"] in ("current", "noise")]


def stale_passages() -> list[dict]:
    """Stale passages plus current ones that have no stale twin, plus noise."""
    all_p = load_corpus()
    stale = [p for p in all_p if p["version"] == "stale"]
    stale_bases = {p["id"].removesuffix("-stale") for p in stale}
    # Prefer stale twin when available; otherwise keep current
    chosen: list[dict] = list(stale)
    for p in all_p:
        if p["version"] == "current" and p["id"] not in stale_bases:
            chosen.append(p)
        elif p["version"] == "noise":
            chosen.append(p)
    return chosen


def passage_by_id(pid: str) -> dict | None:
    for p in load_corpus():
        if p["id"] == pid:
            return p
    return None


_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def score_passage(question: str, passage: dict) -> float:
    """Simple token-overlap score for deterministic retrieval."""
    q_toks = tokenize(question)
    p_toks = tokenize(passage["text"])
    if not q_toks or not p_toks:
        return 0.0
    overlap = len(q_toks & p_toks)
    return overlap / len(q_toks)


def retrieve(
    question: str,
    passages: list[dict],
    k: int = 5,
) -> list[dict]:
    """Return top-k passages by token overlap, stable-sorted by id on ties."""
    scored = [(score_passage(question, p), p) for p in passages]
    scored.sort(key=lambda x: (-x[0], x[1]["id"]))
    return [p for _, p in scored[:k]]
