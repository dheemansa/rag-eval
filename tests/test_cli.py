"""Smoke tests for the CLI."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from demo.corpus import GOLDEN_PATH
from rag_eval.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "compare" in result.output


def test_cli_run_and_report(tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "report.json"
    result = runner.invoke(
        cli,
        [
            "run",
            "--dataset",
            str(GOLDEN_PATH),
            "--rag",
            "demo.baseline:rag_fn",
            "--judge",
            "none",
            "-o",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()

    result2 = runner.invoke(cli, ["report", str(out)])
    assert result2.exit_code == 0
    assert "token_f1" in result2.output or "recall@5" in result2.output


def test_cli_compare_fail_under(tmp_path: Path):
    runner = CliRunner()
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"

    for path, rag in [(base, "demo.baseline:rag_fn"), (cand, "demo.bad_retriever:rag_fn")]:
        result = runner.invoke(
            cli,
            [
                "run",
                "--dataset",
                str(GOLDEN_PATH),
                "--rag",
                rag,
                "--judge",
                "none",
                "-o",
                str(path),
            ],
        )
        assert result.exit_code == 0, result.output

    result = runner.invoke(
        cli,
        [
            "compare",
            str(base),
            str(cand),
            "--fail-under",
            "recall@5=0.8",
        ],
    )
    # Bad retriever should fail the gate
    assert result.exit_code == 1
