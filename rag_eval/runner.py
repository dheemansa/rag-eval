"""Core evaluation runner for rag-eval.

The :func:`evaluate` function is the main entry point: it takes a RAG
function, a golden dataset, and an evaluation config, and returns a
complete :class:`~rag_eval.schema.Report`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from rag_eval.schema import (
    EvalConfig,
    GoldItem,
    JudgeVerdict,
    MetricResult,
    QuestionReport,
    RAGResult,
    Report,
    RunManifest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset I/O
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> list[GoldItem]:
    """Load a golden evaluation dataset from a JSONL file.

    Each line in the file must be a valid JSON object that conforms to the
    :class:`~rag_eval.schema.GoldItem` schema.

    Parameters
    ----------
    path:
        Path to the JSONL file.

    Returns
    -------
    list[GoldItem]
        Parsed golden-set items.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a line cannot be parsed as a valid ``GoldItem``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    items: list[GoldItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(GoldItem(**data))
            except (json.JSONDecodeError, Exception) as exc:
                raise ValueError(
                    f"Error parsing line {line_no} of {path}: {exc}"
                ) from exc

    logger.info("Loaded %d items from %s", len(items), path)
    return items


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _build_manifest(config: EvalConfig) -> RunManifest:
    """Create a :class:`RunManifest` capturing reproducibility metadata.

    Parameters
    ----------
    config:
        The evaluation config for this run.

    Returns
    -------
    RunManifest
    """
    config_json = config.model_dump_json(indent=2)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:16]

    try:
        from rag_eval import __version__ as rag_eval_version
    except ImportError:
        rag_eval_version = "unknown"

    return RunManifest(
        config=config,
        config_hash=config_hash,
        timestamp=datetime.now(timezone.utc).isoformat(),
        rag_eval_version=rag_eval_version,
        python_version=platform.python_version(),
        nli_model_hash="",  # populated by NLIJudge if used
        judge_model=config.judge_model,
    )


# ---------------------------------------------------------------------------
# Judge construction
# ---------------------------------------------------------------------------


def _build_judge(config: EvalConfig):
    """Build the appropriate judge based on the config's ``judge_mode``.

    Returns
    -------
    BaseJudge | None
        A judge instance, or ``None`` when ``judge_mode == "none"``.
    """
    from rag_eval.judges import CachedJudge, JudgeCache, NullJudge

    mode = config.judge_mode.lower()

    if mode == "none":
        logger.info("Judge mode: none — skipping judge-dependent metrics.")
        return NullJudge()

    cache = JudgeCache(config.cache_dir)

    if mode == "replay":
        logger.info("Judge mode: replay — using cached verdicts only.")
        return CachedJudge(
            inner=None,
            cache=cache,
            mode="replay",
            strict=config.strict,
        )

    if mode == "live":
        logger.info("Judge mode: live — will call models and cache results.")
        # For claim-level faithfulness we use the NLI model
        from rag_eval.judges import NLIJudge

        nli_judge = NLIJudge(model_name=config.nli_model)

        # For answer-level correctness we prefer the API judge if configured,
        # otherwise fall back to the NLI judge.
        if config.judge_model:
            import os

            from rag_eval.judges import APIJudge

            api_key = os.environ.get("RAG_EVAL_API_KEY", "")
            api_base = os.environ.get(
                "RAG_EVAL_API_BASE", "https://api.openai.com/v1"
            )
            correctness_judge = APIJudge(
                model=config.judge_model,
                api_key=api_key,
                api_base=api_base,
            )
        else:
            correctness_judge = nli_judge

        # We return a tuple of judges: (claims_judge, correctness_judge)
        # Both are wrapped in their own CachedJudge.
        claims_judge = CachedJudge(
            inner=nli_judge,
            cache=cache,
            mode="live",
            strict=False,
        )
        correctness_cached = CachedJudge(
            inner=correctness_judge,
            cache=cache,
            mode="live",
            strict=False,
        )
        return (claims_judge, correctness_cached)

    raise ValueError(f"Unknown judge_mode: {config.judge_mode!r}")


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    rag_fn: Callable[[str], RAGResult],
    gold_items: list[GoldItem] | None = None,
    dataset: list[GoldItem] | str | Path | None = None,
    config: EvalConfig | None = None,
) -> Report:
    """Run a full RAG evaluation.

    Parameters
    ----------
    rag_fn:
        A callable that takes a question string and returns a
        :class:`~rag_eval.schema.RAGResult`.
    gold_items:
        Pre-loaded list of golden-set items.  Mutually exclusive with
        *dataset* as a path.
    dataset:
        Either a pre-loaded list of :class:`GoldItem`, or a path
        (``str`` / ``Path``) to a JSONL file.  If both *gold_items* and
        *dataset* are given, *gold_items* takes precedence.
    config:
        Evaluation configuration.  Defaults to :class:`EvalConfig` with
        default values.

    Returns
    -------
    Report
        The complete evaluation report with per-question and aggregate
        metrics.
    """
    # Resolve the dataset
    if gold_items is not None:
        items = gold_items
    elif dataset is not None:
        if isinstance(dataset, (str, Path)):
            items = load_dataset(dataset)
        else:
            items = dataset
    else:
        raise ValueError("Either 'gold_items' or 'dataset' must be provided.")

    if not items:
        raise ValueError("Dataset is empty.")

    config = config or EvalConfig()
    logger.info(
        "Starting evaluation: %d questions, judge_mode=%s",
        len(items),
        config.judge_mode,
    )

    # Import metric modules
    from rag_eval.metrics.citations import compute as compute_citations
    from rag_eval.metrics.correctness import compute as compute_correctness
    from rag_eval.metrics.faithfulness import (
        compute as compute_faithfulness,
        split_claims,
    )
    from rag_eval.metrics.retrieval import compute as compute_retrieval

    # Build judge(s)
    judge_obj = _build_judge(config)

    # Determine if we have separate claims / correctness judges
    from rag_eval.judges.base import BaseJudge, NullJudge

    if isinstance(judge_obj, tuple):
        claims_judge, correctness_judge = judge_obj
        has_judge = True
    elif isinstance(judge_obj, NullJudge):
        claims_judge = judge_obj
        correctness_judge = judge_obj
        has_judge = False
    else:
        claims_judge = judge_obj
        correctness_judge = judge_obj
        has_judge = True

    # Process each question
    question_reports: list[QuestionReport] = []

    for idx, gold_item in enumerate(items, 1):
        logger.info(
            "Evaluating question %d/%d: %s",
            idx,
            len(items),
            gold_item.question[:80],
        )

        # --- Call the RAG pipeline ---
        try:
            rag_result = rag_fn(gold_item.question)
        except Exception:
            logger.exception(
                "RAG function failed for question: %s", gold_item.question
            )
            raise

        # --- Retrieval metrics (always computed) ---
        metrics: dict[str, MetricResult] = {}
        retrieval_metrics = compute_retrieval(
            rag_result, gold_item, k=config.k
        )
        metrics.update(retrieval_metrics)

        # --- Split answer into claims ---
        claims = split_claims(rag_result.answer)
        logger.debug("Split answer into %d claims.", len(claims))

        # --- Get verdicts for claims ---
        claim_verdicts: list[JudgeVerdict] | None = None
        if has_judge and claims:
            context_text = "\n\n".join(ctx.text for ctx in rag_result.contexts)
            claim_verdicts = claims_judge.judge_claims_batch(
                claims, context_text
            )
            logger.debug("Got %d claim verdicts.", len(claim_verdicts))
        elif not has_judge:
            claim_verdicts = None

        # --- Faithfulness metrics ---
        faith_metrics = compute_faithfulness(
            rag_result,
            gold_item,
            verdicts=claim_verdicts,
            threshold=config.faithfulness_threshold,
        )
        metrics.update(faith_metrics)

        # --- Citation metrics ---
        citation_metrics = compute_citations(
            rag_result,
            gold_item,
            verdicts=claim_verdicts,
            threshold=config.faithfulness_threshold,
        )
        metrics.update(citation_metrics)

        # --- Correctness metrics ---
        correctness_verdict: JudgeVerdict | None = None
        if has_judge:
            correctness_verdict = correctness_judge.judge_answer(
                gold_item.question,
                rag_result.answer,
                gold_item.gold_answer,
            )

        correctness_metrics = compute_correctness(
            rag_result,
            gold_item,
            judge_verdict=correctness_verdict,
        )
        metrics.update(correctness_metrics)

        # --- Build QuestionReport ---
        qr = QuestionReport(
            question=gold_item.question,
            question_type=gold_item.question_type,
            rag_result=rag_result,
            gold_item=gold_item,
            metrics=metrics,
        )
        question_reports.append(qr)

    # --- Aggregate metrics ---
    aggregate_metrics = _aggregate_metrics(question_reports)
    logger.info("Aggregate metrics: %s", aggregate_metrics)

    # --- Build manifest ---
    manifest = _build_manifest(config)

    return Report(
        question_reports=question_reports,
        aggregate_metrics=aggregate_metrics,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_metrics(
    question_reports: list[QuestionReport],
) -> dict[str, float]:
    """Compute the mean of each metric across all questions.

    Metric values of ``-1.0`` are treated as *not evaluated* and excluded
    from the mean computation.

    Parameters
    ----------
    question_reports:
        The per-question evaluation reports.

    Returns
    -------
    dict[str, float]
        Mapping of metric name to its mean value.
    """
    from collections import defaultdict

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for qr in question_reports:
        for name, mr in qr.metrics.items():
            # Skip not-evaluated sentinel values
            if mr.value == -1.0:
                continue
            if mr.details.get("status") == "not_evaluated":
                continue
            sums[name] += mr.value
            counts[name] += 1

    aggregated: dict[str, float] = {}
    for name in sorted(sums):
        if counts[name] > 0:
            aggregated[name] = sums[name] / counts[name]
        else:
            aggregated[name] = -1.0

    return aggregated
