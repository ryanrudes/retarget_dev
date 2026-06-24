"""Smoke tests for the unified ``retarget`` Typer CLI.

The top-level app attaches the (source-checkout-only) example gallery as the
``examples`` command group, so these exercise the shipped wiring end to end:
help renders one command tree, the gallery's modes are listed, and name
validation rejects unknown captures. They deliberately stay on the cheap paths
(``--help``, ``list``, a validation error) and never trigger a matplotlib render.
"""

from __future__ import annotations

from typer.testing import CliRunner

from retarget.cli import app

runner = CliRunner()


def test_top_level_help_lists_examples() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "examples" in result.output


def test_examples_group_help_lists_modes() -> None:
    result = runner.invoke(app, ["examples", "--help"])
    assert result.exit_code == 0
    for mode in ("list", "timeline", "geom2d", "geom3d", "animate", "merged", "video"):
        assert mode in result.output


def test_examples_list_runs() -> None:
    result = runner.invoke(app, ["examples", "list"])
    assert result.exit_code == 0


def test_unknown_example_is_rejected() -> None:
    result = runner.invoke(app, ["examples", "timeline", "left_shoe_does_not_exist"])
    assert result.exit_code != 0
