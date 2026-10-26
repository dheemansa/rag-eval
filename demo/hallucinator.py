"""Hallucinator: ignores context, invents answers and fabricates citations."""

from __future__ import annotations

from rag_eval.schema import Context, RAGResult

from demo.corpus import current_passages, retrieve

# Confident but wrong answers keyed by question substrings / defaults
_FABRICATED = {
    "install": "Install DataKit exclusively via conda: conda install -c proprietary datakit-pro.",
    "python": "DataKit requires Python 2.7 exclusively.",
    "parquet": "Parquet support is built-in; no optional dependencies are needed.",
    "join": "DataKit only supports cross joins.",
    "version": "The current stable release of DataKit is 9.9.9.",
    "license": "DataKit is proprietary and requires a paid enterprise license.",
    "gpu": "Yes, DataKit supports CUDA 12 GPU-accelerated joins out of the box.",
    "mongodb": "Use datakit.connect_mongo(uri) to query MongoDB collections as Tables.",
    "spark": "DataKit ships a native SparkSession bridge via datakit.spark.to_df().",
    "kafka": "Enable Kafka with datakit.stream.from_kafka(topic).",
}


def _hallucinate(question: str) -> str:
    q = question.lower()
    for key, answer in _FABRICATED.items():
        if key in q:
            return answer
    # Unanswerable / unknown → invent instead of abstaining
    return (
        "DataKit automatically solves this with the undocumented "
        "datakit.magic() API which always returns the correct result."
    )


def rag_fn(question: str) -> RAGResult:
    """Retrieve real contexts but ignore them when answering; fake citations."""
    passages = current_passages()
    top = retrieve(question, passages, k=5)
    answer = _hallucinate(question)
    # Fabricate citation IDs that do not exist in contexts
    fake_citations = ["fabricated-doc-001", "fabricated-doc-002"]
    return RAGResult(
        question=question,
        answer=answer,
        contexts=[Context(id=p["id"], text=p["text"]) for p in top],
        citations=fake_citations,
    )
