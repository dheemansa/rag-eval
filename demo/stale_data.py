"""Stale Data: retrieves outdated doc versions — faithful but wrong."""

from __future__ import annotations

import json
import re

from rag_eval.schema import Context, RAGResult

from demo.corpus import GOLDEN_PATH, retrieve, stale_passages

_ABSTAIN = "I don't know. The retrieved documentation does not contain enough information to answer."

# Map current gold passage → stale twin when one exists
_STALE_TWIN = {
    "dk-install-01": "dk-install-01-stale",
    "dk-agg-01": "dk-agg-01-stale",
    "dk-join-01": "dk-join-01-stale",
    "dk-config-01": "dk-config-01-stale",
    "dk-version-01": "dk-version-01-stale",
}


def _load_gold() -> dict[str, dict]:
    gold: dict[str, dict] = {}
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold[item["question"]] = item
    return gold


_GOLD = _load_gold()


def _answer_from_stale(question: str, contexts: list[dict]) -> tuple[str, list[str]]:
    """Answer faithfully from retrieved (possibly stale) text."""
    gold = _GOLD.get(question)
    if gold is not None and gold["question_type"] == "unanswerable":
        return _ABSTAIN, []

    if not contexts:
        return _ABSTAIN, []

    # Prefer answering from a stale passage if one was retrieved
    stale_ctx = next((c for c in contexts if c["id"].endswith("-stale")), None)
    source = stale_ctx or contexts[0]

    # Extract a faithful answer: use the full passage text's first 1–2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", source["text"])
    answer = " ".join(sentences[:2]).strip()
    return answer, [source["id"]]


def rag_fn(question: str) -> RAGResult:
    """Retrieve preferring stale twins; answer faithfully from them."""
    passages = stale_passages()
    top = retrieve(question, passages, k=5)

    # If gold points at a passage with a stale twin, force the twin into top
    gold = _GOLD.get(question)
    if gold:
        for gid in gold.get("gold_passage_ids", []):
            twin_id = _STALE_TWIN.get(gid)
            if twin_id:
                twin = next((p for p in passages if p["id"] == twin_id), None)
                if twin and twin not in top:
                    top = [twin] + [p for p in top if p["id"] != twin_id][:4]

    answer, citations = _answer_from_stale(question, top)
    return RAGResult(
        question=question,
        answer=answer,
        contexts=[Context(id=p["id"], text=p["text"]) for p in top],
        citations=citations,
    )
