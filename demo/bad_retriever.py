"""Bad Retriever: deliberately weak / random retrieval."""

from __future__ import annotations

import hashlib
import re

from rag_eval.schema import Context, RAGResult

from demo.corpus import current_passages


def _pseudo_random_order(question: str, passages: list[dict]) -> list[dict]:
    """Deterministic shuffle keyed by the question (reproducible 'random')."""

    def key(p: dict) -> str:
        digest = hashlib.sha256(f"{question}:{p['id']}".encode()).hexdigest()
        return digest

    return sorted(passages, key=key)


def rag_fn(question: str) -> RAGResult:
    """Retrieve unrelated passages and answer from them anyway."""
    passages = current_passages()
    # Prefer noise / low-relevance by taking the "random" tail
    ordered = _pseudo_random_order(question, passages)
    top = ordered[:5]

    if top:
        sentence = re.split(r"(?<=[.!?])\s+", top[0]["text"])[0]
        answer = sentence
        citations = [top[0]["id"]]
    else:
        answer = "I don't know."
        citations = []

    return RAGResult(
        question=question,
        answer=answer,
        contexts=[Context(id=p["id"], text=p["text"]) for p in top],
        citations=citations,
    )
