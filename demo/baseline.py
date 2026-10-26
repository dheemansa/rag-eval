"""Baseline RAG: keyword retrieval over current docs + extractive answers."""

from __future__ import annotations

import re

from rag_eval.schema import Context, RAGResult

from demo.corpus import GOLDEN_PATH, current_passages, retrieve

# Preload gold answers keyed by question for extractive answering
_GOLD: dict[str, dict] | None = None

_ABSTAIN = "I don't know. The retrieved documentation does not contain enough information to answer."


def _load_gold() -> dict[str, dict]:
    global _GOLD
    if _GOLD is not None:
        return _GOLD
    import json

    gold: dict[str, dict] = {}
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold[item["question"]] = item
    _GOLD = gold
    return gold


def _answer_from_contexts(question: str, contexts: list[dict]) -> tuple[str, list[str]]:
    """Produce an answer grounded in retrieved contexts.

    For known golden questions with overlapping gold passages, return the gold
    answer (simulating a faithful generator). For unanswerable items, abstain.
    Otherwise extract the first sentence from the top context.
    """
    gold = _load_gold().get(question)
    retrieved_ids = {c["id"] for c in contexts}

    if gold is not None:
        if gold["question_type"] == "unanswerable":
            return _ABSTAIN, []
        gold_ids = set(gold["gold_passage_ids"])
        if gold_ids & retrieved_ids:
            citations = [pid for pid in gold["gold_passage_ids"] if pid in retrieved_ids]
            return gold["gold_answer"], citations

    if not contexts:
        return _ABSTAIN, []

    # Fallback: first sentence of top context
    text = contexts[0]["text"]
    sentence = re.split(r"(?<=[.!?])\s+", text)[0]
    return sentence, [contexts[0]["id"]]


def rag_fn(question: str) -> RAGResult:
    """Baseline RAG pipeline."""
    passages = current_passages()
    top = retrieve(question, passages, k=5)
    answer, citations = _answer_from_contexts(question, top)
    return RAGResult(
        question=question,
        answer=answer,
        contexts=[Context(id=p["id"], text=p["text"]) for p in top],
        citations=citations,
    )
