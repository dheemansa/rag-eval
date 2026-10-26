"""JSON report I/O, comparison, and human-readable formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_eval.schema import Report


def save_report(report: Report, path: str | Path) -> None:
    """Write a :class:`Report` to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )


def load_report(path: str | Path) -> Report:
    """Load a :class:`Report` from a JSON file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return Report(**data)


def compare_reports(
    baseline: Report,
    candidate: Report,
) -> dict[str, dict[str, float]]:
    """Compare aggregate metrics between two reports.

    Returns a mapping of metric name → ``{baseline, candidate, delta}``.
    Metrics present in only one report are included with the missing side as
    ``0.0``.
    """
    all_keys = set(baseline.aggregate_metrics) | set(candidate.aggregate_metrics)
    diff: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        b_val = baseline.aggregate_metrics.get(key, 0.0)
        c_val = candidate.aggregate_metrics.get(key, 0.0)
        # Treat not-evaluated sentinels as 0 for delta display
        if b_val < 0:
            b_val = 0.0
        if c_val < 0:
            c_val = 0.0
        diff[key] = {
            "baseline": b_val,
            "candidate": c_val,
            "delta": c_val - b_val,
        }
    return diff


def format_report_summary(report: Report, fmt: str = "markdown") -> str:
    """Format aggregate metrics as a readable summary string.

    Parameters
    ----------
    report:
        The evaluation report.
    fmt:
        ``"markdown"`` or ``"text"``.
    """
    lines: list[str] = []
    if fmt == "markdown":
        lines.append("# Evaluation Report")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("| --- | ---: |")
        for name, value in sorted(report.aggregate_metrics.items()):
            display = "n/a" if value < 0 else f"{value:.4f}"
            lines.append(f"| {name} | {display} |")
        lines.append("")
        lines.append(f"_Questions: {len(report.question_reports)}_")
        if report.manifest.config_hash:
            lines.append(f"_Config hash: `{report.manifest.config_hash}`_")
    else:
        lines.append("Evaluation Report")
        lines.append("=" * 40)
        for name, value in sorted(report.aggregate_metrics.items()):
            display = "n/a" if value < 0 else f"{value:.4f}"
            lines.append(f"  {name}: {display}")
        lines.append(f"Questions: {len(report.question_reports)}")
    return "\n".join(lines)
