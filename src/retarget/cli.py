"""Top-level ``retarget`` command-line entry point.

The library is import-first; this CLI exists only to drive bundled tooling.
Subcommands are deliberately thin. ``retarget examples`` runs the example
gallery, which lives in the ``examples/`` tree next to ``src/`` and is therefore
only available from a source checkout (it is not shipped in the wheel). It is
loaded lazily so the installed package never hard-depends on it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, cast

from rich_argparse import RichHelpFormatter


def _load_examples_main() -> Callable[..., int] | None:
    """Import ``examples/cli.py`` from the source checkout, if present."""
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
    return cast("Callable[..., int]", module.main)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Dispatch `examples` manually, before argparse, so EVERYTHING after it
    # (including --help and flags) is forwarded verbatim to the gallery CLI,
    # which owns its own argument parsing.
    if argv and argv[0] == "examples":
        examples_main = _load_examples_main()
        if examples_main is None:
            print("retarget examples: the example gallery is only available from a source checkout", file=sys.stderr)
            return 1
        return examples_main(argv[1:], prog="retarget examples")

    # Otherwise let argparse handle top-level help, usage, and unknown commands.
    parser = argparse.ArgumentParser(
        prog="retarget",
        description="Motion-retargeting research toolkit.",
        formatter_class=RichHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.add_parser("examples", help="render the bundled motion examples (source checkout only)")

    parser.parse_args(argv)  # handles -h / unknown commands, then we show help
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
