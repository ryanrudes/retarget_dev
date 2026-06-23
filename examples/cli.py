#!/usr/bin/env python3
"""The ``retarget examples`` gallery: render a ``left_shoe_*`` capture.

One scene definition (``_shared/scene.py``) and one set of visualizers
(``_shared/*``) are reused across every ``left_shoe_*`` capture. Each example
folder contributes only its contact-detection *experiment* (``experiment.py``);
everything else -- building the scene from the folder name, plotting the contact
timeline, and the 2D / 3D geometry inspectors -- is driven from here.

Usage::

    retarget examples list
    retarget examples timeline left_shoe_slides_on_board
    retarget examples geom2d   left_shoe_slides_on_board
    retarget examples geom3d   left_shoe_slides_on_board
    retarget examples animate  left_shoe_slides_on_board
    retarget examples merged   left_shoe_slides_on_board left_shoe_hovers_in_air
    retarget examples merged   --all
    retarget examples video    left_shoe_slides_on_board
    retarget examples video    --all --fps 30

``merged`` stacks several experiments' contact timelines into one static figure.
``video`` renders a real-time movie (timeline with a moving time cursor next to
the animated 3D geometry) -- one example gives an individual video, several give
a merged one. ``--all`` selects every discovered example. Both modes title each
panel by its experiment folder name.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# Make `retarget` (src layout) and `_shared` importable regardless of cwd.
for _p in (SRC_DIR, EXAMPLES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rich.table import Table
from rich_argparse import RawDescriptionRichHelpFormatter

from _shared import inspect_geometry, inspect_geometry_3d, run as animation
from _shared.console import console, error_console
from _shared.contact_timeline import (
    TimelineSpec,
    plot_contact_timeline,
    plot_merged_timelines,
)
from _shared.scene import get_demo, unbagged_dir_for
from _shared.video import render_video


def discover_examples() -> list[str]:
    """Every ``left_shoe_*`` folder that declares an ``experiment.py``."""
    return sorted(
        p.name
        for p in EXAMPLES_DIR.iterdir()
        if p.is_dir() and p.name.startswith("left_shoe_") and (p / "experiment.py").is_file()
    )


def load_experiment(example: str) -> ModuleType:
    """Import ``<example>/experiment.py`` as a standalone module."""
    path = EXAMPLES_DIR / example / "experiment.py"
    spec = importlib.util.spec_from_file_location(f"_experiment_{example}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load experiment from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timeline_spec_for(example: str) -> TimelineSpec:
    """Build the example scene and run its contact-detection experiment."""
    demo = get_demo(unbagged_dir_for(example))
    experiment = load_experiment(example)
    spec: TimelineSpec = experiment.build_timeline(demo.tracks["mocap"])
    return spec


def run_timeline(example: str) -> None:
    spec = timeline_spec_for(example)
    plot_contact_timeline(spec, EXAMPLES_DIR / example / "contact_timeline.png")


def run_merged(examples: list[str]) -> None:
    # Each panel is titled by its experiment folder name, so the stacked figure
    # reads as a direct comparison across captures.
    named_specs = [(example, timeline_spec_for(example)) for example in examples]
    plot_merged_timelines(named_specs, EXAMPLES_DIR / "merged_timeline.png")


def run_video(examples: list[str], fps: int) -> None:
    # One experiment -> an individual video in its folder; several -> a merged
    # video at the examples root. Each panel is titled by its folder name.
    named_specs = [(example, timeline_spec_for(example)) for example in examples]
    if len(examples) == 1:
        output = EXAMPLES_DIR / examples[0] / "video"
    else:
        output = EXAMPLES_DIR / "merged_video"
    render_video(named_specs, output, fps=fps)


def run_geom2d(example: str) -> None:
    demo = get_demo(unbagged_dir_for(example))
    inspect_geometry.main(demo, EXAMPLES_DIR / example / "inspect_geometry.png")


def run_geom3d(example: str) -> None:
    demo = get_demo(unbagged_dir_for(example))
    inspect_geometry_3d.main(demo)


def run_animate(example: str) -> None:
    demo = get_demo(unbagged_dir_for(example))
    animation.animate(demo)


MODES = {
    "timeline": run_timeline,
    "geom2d": run_geom2d,
    "geom3d": run_geom3d,
    "animate": run_animate,
}
MULTI_MODES = ("merged", "video")

# Clean, terminal-facing help text. The module docstring above keeps the
# reStructuredText house style for code readers; argparse gets prose instead so
# `--help` doesn't show literal ``backticks`` and ``::`` markers.
_CLI_DESCRIPTION = "Render a left_shoe_* motion example: contact timeline, 2D/3D geometry, or video."

_CLI_EPILOG = (
    "examples:\n"
    "  retarget examples list\n"
    "  retarget examples timeline left_shoe_slides_on_board\n"
    "  retarget examples geom2d   left_shoe_slides_on_board\n"
    "  retarget examples geom3d   left_shoe_slides_on_board\n"
    "  retarget examples animate  left_shoe_slides_on_board\n"
    "  retarget examples merged   --all\n"
    "  retarget examples video    left_shoe_slides_on_board\n"
    "  retarget examples video    --all --fps 30\n"
    "\n"
    "modes:\n"
    "  list                 show the available examples\n"
    "  timeline             contact-state timeline (height / speed / state)\n"
    "  geom2d / geom3d      2D / 3D geometry inspection at frame 0\n"
    "  animate              animate the sole contact patch\n"
    "  merged               stack several timelines into one figure\n"
    "  video                real-time movie: timeline cursor + animated 3D geometry\n"
    "\n"
    "One example gives individual output; several names (or --all) give a merged one."
)


def _print_examples(available: list[str]) -> None:
    if not available:
        error_console.print("no examples found")
        return
    table = Table(title="examples", title_style="bold", header_style="bold cyan")
    table.add_column("#", justify="right", style="dim")
    table.add_column("example")
    for i, name in enumerate(available, 1):
        table.add_row(str(i), name)
    console.print(table)


def main(argv: list[str] | None = None, *, prog: str = "retarget examples") -> int:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=_CLI_DESCRIPTION,
        epilog=_CLI_EPILOG,
        formatter_class=RawDescriptionRichHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["list", *MULTI_MODES, *MODES],
        help="list examples; run one visualization; 'merged' stacks timelines; 'video' renders a real-time movie",
    )
    parser.add_argument(
        "examples",
        nargs="*",
        help="left_shoe_* folder name(s); 'merged'/'video' take one or more, other modes take one",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_examples",
        help="select every discovered example (merged/video only)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="frames per second for 'video' mode (default: 30)",
    )
    args = parser.parse_args(argv)

    available = discover_examples()

    if args.mode == "list":
        _print_examples(available)
        return 0

    if args.all_examples:
        if args.examples:
            parser.error("pass either --all or explicit example names, not both")
        if args.mode not in MULTI_MODES:
            parser.error(f"--all only applies to {' / '.join(MULTI_MODES)}")
        selected = available
    else:
        selected = args.examples

    unknown = [e for e in selected if e not in available]
    if unknown:
        parser.error(f"unknown example(s) {', '.join(unknown)}; choose from:\n  " + "\n  ".join(available))

    if args.mode in MULTI_MODES:
        if not selected:
            parser.error(f"the {args.mode!r} mode needs at least one example (or --all); choose from:\n  " + "\n  ".join(available))
        if args.mode == "merged":
            run_merged(selected)
        else:
            run_video(selected, args.fps)
        return 0

    if len(selected) != 1:
        parser.error(f"the {args.mode!r} mode takes exactly one example; choose one of:\n  " + "\n  ".join(available))

    MODES[args.mode](selected[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
