"""Command-line interface for rag-eval.

Entry point: ``rag-eval`` (see ``pyproject.toml``).

All heavy imports (runner, report, rich) are deferred to the command bodies
so that ``rag-eval --help`` stays fast.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("rag_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_rag_function(dotted_path: str) -> Any:
    """Dynamically import a RAG function from a ``module:function`` path.

    Parameters
    ----------
    dotted_path:
        A string of the form ``some.module:function_name``.

    Returns
    -------
    callable
        The imported function object.
    """
    if ":" not in dotted_path:
        raise click.BadParameter(
            f"RAG function path must be in 'module:function' format, got '{dotted_path}'"
        )
    module_path, func_name = dotted_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise click.BadParameter(f"Could not import module '{module_path}': {exc}") from exc
    try:
        func = getattr(module, func_name)
    except AttributeError as exc:
        raise click.BadParameter(
            f"Module '{module_path}' has no attribute '{func_name}'"
        ) from exc
    if not callable(func):
        raise click.BadParameter(f"'{dotted_path}' is not callable")
    return func


def _color_for_value(value: float) -> str:
    """Return a rich color name based on metric value thresholds."""
    if value >= 0.8:
        return "green"
    if value >= 0.5:
        return "yellow"
    return "red"


def _parse_fail_under(values: tuple[str, ...]) -> dict[str, float]:
    """Parse ``--fail-under metric=threshold`` values into a dict."""
    result: dict[str, float] = {}
    for entry in values:
        if "=" not in entry:
            raise click.BadParameter(
                f"--fail-under must be in 'metric=threshold' format, got '{entry}'"
            )
        metric, raw_threshold = entry.split("=", 1)
        try:
            threshold = float(raw_threshold)
        except ValueError as exc:
            raise click.BadParameter(
                f"Invalid threshold '{raw_threshold}' for metric '{metric}'"
            ) from exc
        result[metric.strip()] = threshold
    return result


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.version_option(package_name="rag-eval", prog_name="rag-eval")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """rag-eval — pytest for RAG pipelines.

    Score any RAG pipeline on retrieval, faithfulness, citations, and
    correctness.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# rag-eval run
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--dataset",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the golden dataset JSONL file.",
)
@click.option(
    "--rag",
    required=True,
    type=str,
    help="Python import path to the RAG function (e.g. 'demo.baseline:rag_fn').",
)
@click.option(
    "--judge",
    type=click.Choice(["none", "replay", "live"], case_sensitive=False),
    default="replay",
    show_default=True,
    help="Judge mode for faithfulness / correctness evaluation.",
)
@click.option("--judge-model", type=str, default="", show_default=True, help="LLM model name for the live API judge.")
@click.option(
    "--nli-model",
    type=str,
    default="cross-encoder/nli-deberta-v3-xsmall",
    show_default=True,
    help="NLI model for entailment scoring.",
)
@click.option("--k", type=int, default=5, show_default=True, help="Cutoff k for retrieval metrics.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for reproducibility.")
@click.option(
    "--cache-dir",
    type=str,
    default=".rag_eval_cache",
    show_default=True,
    help="Directory to cache judge / NLI results.",
)
@click.option("--strict", is_flag=True, default=False, help="Fail on cache miss in replay mode.")
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default="report.json",
    show_default=True,
    help="Output path for the JSON report.",
)
@click.pass_context
def run(
    ctx: click.Context,
    dataset: Path,
    rag: str,
    judge: str,
    judge_model: str,
    nli_model: str,
    k: int,
    seed: int,
    cache_dir: str,
    strict: bool,
    output: Path,
) -> None:
    """Run a full RAG evaluation against a golden dataset."""
    from rich.console import Console
    from rich.table import Table

    from rag_eval.report import save_report
    from rag_eval.runner import evaluate, load_dataset
    from rag_eval.schema import EvalConfig

    console = Console()

    # --- Import the RAG function ------------------------------------------------
    try:
        rag_fn = _import_rag_function(rag)
    except click.BadParameter as exc:
        console.print(f"[bold red]Error:[/bold red] {exc.format_message()}")
        raise SystemExit(1) from exc

    # --- Build config -----------------------------------------------------------
    config = EvalConfig(
        judge_mode=judge,
        judge_model=judge_model,
        nli_model=nli_model,
        seed=seed,
        k=k,
        cache_dir=cache_dir,
        strict=strict,
    )
    logger.debug("EvalConfig: %s", config.model_dump_json(indent=2))

    # --- Load dataset -----------------------------------------------------------
    console.print(f"[bold]Loading dataset from[/bold] {dataset} …")
    try:
        gold_items = load_dataset(dataset)
    except Exception as exc:
        console.print(f"[bold red]Error loading dataset:[/bold red] {exc}")
        raise SystemExit(1) from exc
    console.print(f"  Loaded [cyan]{len(gold_items)}[/cyan] items.")

    # --- Evaluate ---------------------------------------------------------------
    console.print("[bold]Running evaluation …[/bold]")
    try:
        report = evaluate(rag_fn=rag_fn, gold_items=gold_items, config=config)
    except Exception as exc:
        console.print(f"[bold red]Evaluation failed:[/bold red] {exc}")
        logger.debug("Evaluation traceback", exc_info=True)
        raise SystemExit(1) from exc

    # --- Save report ------------------------------------------------------------
    save_report(report, output)
    console.print(f"[bold green]Report saved to[/bold green] {output}")

    # --- Print summary table ----------------------------------------------------
    table = Table(title="Evaluation Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")

    for metric_name, value in sorted(report.aggregate_metrics.items()):
        color = _color_for_value(value)
        table.add_row(metric_name, f"[{color}]{value:.4f}[/{color}]")

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# rag-eval compare
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--fail-under",
    multiple=True,
    type=str,
    help="Threshold in 'metric=value' format (repeatable). Exit 1 if not met.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format for the comparison.",
)
def compare(baseline: Path, candidate: Path, fail_under: tuple[str, ...], fmt: str) -> None:
    """Compare two evaluation reports (BASELINE vs CANDIDATE)."""
    from rich.console import Console
    from rich.table import Table

    from rag_eval.report import compare_reports, load_report

    console = Console()

    # --- Load reports -----------------------------------------------------------
    try:
        baseline_report = load_report(baseline)
        candidate_report = load_report(candidate)
    except Exception as exc:
        console.print(f"[bold red]Error loading reports:[/bold red] {exc}")
        raise SystemExit(1) from exc

    # --- Parse thresholds -------------------------------------------------------
    try:
        thresholds = _parse_fail_under(fail_under)
    except click.BadParameter as exc:
        console.print(f"[bold red]Error:[/bold red] {exc.format_message()}")
        raise SystemExit(1) from exc

    # --- Compare ----------------------------------------------------------------
    diff = compare_reports(baseline_report, candidate_report)

    # --- Output -----------------------------------------------------------------
    if fmt == "json":
        console.print_json(json.dumps(diff, indent=2, default=str))
    else:
        table = Table(title="Report Comparison (Baseline → Candidate)", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Baseline", justify="right")
        table.add_column("Candidate", justify="right")
        table.add_column("Δ (delta)", justify="right")

        for metric_name, info in sorted(diff.items()):
            if isinstance(info, dict):
                b_val = info.get("baseline", 0.0)
                c_val = info.get("candidate", 0.0)
                delta = info.get("delta", c_val - b_val)
            else:
                # Simple numeric diff
                b_val = baseline_report.aggregate_metrics.get(metric_name, 0.0)
                c_val = candidate_report.aggregate_metrics.get(metric_name, 0.0)
                delta = float(info)

            delta_color = "green" if delta >= 0 else "red"
            sign = "+" if delta >= 0 else ""
            table.add_row(
                metric_name,
                f"{b_val:.4f}",
                f"[{_color_for_value(c_val)}]{c_val:.4f}[/{_color_for_value(c_val)}]",
                f"[{delta_color}]{sign}{delta:.4f}[/{delta_color}]",
            )

        console.print()
        console.print(table)

    # --- Check fail-under -------------------------------------------------------
    failed: list[str] = []
    for metric, threshold in thresholds.items():
        actual = candidate_report.aggregate_metrics.get(metric)
        if actual is None:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] metric '{metric}' not found in candidate report."
            )
            failed.append(metric)
        elif actual < threshold:
            console.print(
                f"[bold red]FAIL:[/bold red] {metric} = {actual:.4f} < {threshold:.4f}"
            )
            failed.append(metric)

    if failed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# rag-eval report
# ---------------------------------------------------------------------------


@cli.command("report")
@click.argument("report_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json", "markdown"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format for the report.",
)
def report_cmd(report_path: Path, fmt: str) -> None:
    """Display a saved evaluation report."""
    from rich.console import Console
    from rich.table import Table

    from rag_eval.report import format_report_summary, load_report

    console = Console()

    try:
        report = load_report(report_path)
    except Exception as exc:
        console.print(f"[bold red]Error loading report:[/bold red] {exc}")
        raise SystemExit(1) from exc

    if fmt == "json":
        console.print_json(report.model_dump_json(indent=2))

    elif fmt == "markdown":
        md_output = format_report_summary(report, fmt="markdown")
        click.echo(md_output)

    else:
        # Default: rich table with color-coded metrics
        table = Table(title="Evaluation Report", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Score", justify="right")

        for metric_name, value in sorted(report.aggregate_metrics.items()):
            color = _color_for_value(value)
            table.add_row(metric_name, f"[{color}]{value:.4f}[/{color}]")

        console.print()
        console.print(table)

        # Per-question breakdown if verbose
        if len(report.question_reports) > 0:
            detail_table = Table(title="Per-Question Breakdown", show_lines=True)
            detail_table.add_column("#", justify="right", style="dim")
            detail_table.add_column("Question", max_width=50)
            detail_table.add_column("Type", style="cyan")

            # Gather all metric names across questions
            all_metric_names: set[str] = set()
            for qr in report.question_reports:
                all_metric_names.update(qr.metrics.keys())

            for m_name in sorted(all_metric_names):
                detail_table.add_column(m_name, justify="right")

            for idx, qr in enumerate(report.question_reports, 1):
                row: list[str] = [
                    str(idx),
                    qr.question[:50],
                    qr.question_type.value,
                ]
                for m_name in sorted(all_metric_names):
                    mr = qr.metrics.get(m_name)
                    if mr is not None:
                        color = _color_for_value(mr.value)
                        row.append(f"[{color}]{mr.value:.4f}[/{color}]")
                    else:
                        row.append("[dim]—[/dim]")
                detail_table.add_row(*row)

            console.print()
            console.print(detail_table)
