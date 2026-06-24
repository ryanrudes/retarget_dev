"""Top-level ``retarget`` command-line entry point.

The library is import-first; this CLI exists only to drive bundled tooling. Its
one substantive command group, ``examples``, is the motion-example gallery that
lives in the ``examples/`` tree next to ``src/`` and is therefore only available
from a source checkout (it is not shipped in the wheel).

The gallery is its own ``typer.Typer`` app. We attach it here with
``add_typer`` so the whole thing behaves as a single CLI -- ``retarget --help``,
``retarget examples --help``, nested subcommands, and shell completion all work
across one command tree. When the gallery is absent (an installed wheel with no
source checkout) the ``examples`` command degrades to a short "source checkout
only" message instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="retarget",
    help="Motion-retargeting research toolkit.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _root() -> None:
    """Anchor the app as a command group.

    Without an explicit callback, Typer collapses a single-command app into that
    one command; the callback keeps ``examples`` a real subcommand.
    """


def _load_examples_app() -> typer.Typer | None:
    """Import the gallery's ``typer.Typer`` app from the source checkout, if present."""
    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    cli_path = examples_dir / "cli.py"
    if not cli_path.is_file():
        return None
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    spec = importlib.util.spec_from_file_location("retarget_examples_cli", cli_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gallery_app = getattr(module, "app", None)
    return gallery_app if isinstance(gallery_app, typer.Typer) else None


_examples_app = _load_examples_app()
if _examples_app is not None:
    app.add_typer(_examples_app, name="examples")
else:

    @app.command(
        "examples",
        # Swallow whatever follows `examples` so the message is shown regardless
        # of which gallery subcommand/flags were typed.
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
        add_help_option=False,
    )
    def _examples_unavailable(ctx: typer.Context) -> None:
        """Render the bundled motion examples (source checkout only)."""
        typer.echo(
            "retarget examples: the example gallery is only available from a source checkout",
            err=True,
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
