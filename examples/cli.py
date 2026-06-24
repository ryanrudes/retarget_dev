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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Annotated

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# Make `retarget` (src layout) and `_shared` importable regardless of cwd.
for _p in (SRC_DIR, EXAMPLES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import typer
from rich.table import Table

# Only the light, matplotlib-free helpers are imported eagerly. The plotting /
# video / geometry-inspector modules pull in matplotlib, so they are imported
# inside the command bodies that use them -- keeping `retarget --help` and
# `retarget examples --help` fast (no figure backend loaded just to list
# commands).
from _shared.console import console, error_console
from _shared.scene import get_demo, unbagged_dir_for

if TYPE_CHECKING:
    from _shared.contact_timeline import TimelineSpec

app = typer.Typer(
    name="retarget examples",
    help="Render a left_shoe_* motion example: contact timeline, 2D/3D geometry, or video.",
    no_args_is_help=True,
    add_completion=False,
)


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


def _require_known(example: str) -> str:
    """Validate a single example name, raising a choice-listing error if unknown."""
    available = discover_examples()
    if example not in available:
        raise typer.BadParameter(f"unknown example {example!r}; choose from:\n  " + "\n  ".join(available))
    return example


def _select_many(examples: list[str], all_examples: bool, mode: str) -> list[str]:
    """Resolve the example set for a merged/video run (explicit names or ``--all``)."""
    available = discover_examples()
    if all_examples:
        if examples:
            raise typer.BadParameter("pass either --all or explicit example names, not both")
        return available
    if not examples:
        raise typer.BadParameter(
            f"the {mode!r} mode needs at least one example (or --all); choose from:\n  " + "\n  ".join(available)
        )
    unknown = [e for e in examples if e not in available]
    if unknown:
        raise typer.BadParameter(f"unknown example(s) {', '.join(unknown)}; choose from:\n  " + "\n  ".join(available))
    return examples


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


# One required example-name argument, shared by the single-capture modes.
ExampleArg = Annotated[str, typer.Argument(help="left_shoe_* folder name", callback=_require_known)]


@app.command("list")
def list_examples() -> None:
    """Show the available examples."""
    _print_examples(discover_examples())


@app.command()
def timeline(example: ExampleArg) -> None:
    """Contact-state timeline (height / speed / state)."""
    from _shared.contact_timeline import plot_contact_timeline

    spec = timeline_spec_for(example)
    plot_contact_timeline(spec, EXAMPLES_DIR / example / "contact_timeline.png")


@app.command()
def geom2d(example: ExampleArg) -> None:
    """2D geometry inspection at frame 0."""
    from _shared import inspect_geometry

    demo = get_demo(unbagged_dir_for(example))
    inspect_geometry.main(demo, EXAMPLES_DIR / example / "inspect_geometry.png")


@app.command()
def geom3d(example: ExampleArg) -> None:
    """3D geometry inspection at frame 0."""
    from _shared import inspect_geometry_3d

    demo = get_demo(unbagged_dir_for(example))
    inspect_geometry_3d.main(demo)


@app.command()
def animate(example: ExampleArg) -> None:
    """Animate the sole contact patch."""
    from _shared import run as animation

    demo = get_demo(unbagged_dir_for(example))
    animation.animate(demo)


@app.command()
def merged(
    examples: Annotated[list[str] | None, typer.Argument(help="left_shoe_* folder name(s)")] = None,
    all_examples: Annotated[bool, typer.Option("--all", help="select every discovered example")] = False,
) -> None:
    """Stack several contact timelines into one figure."""
    from _shared.contact_timeline import plot_merged_timelines

    selected = _select_many(examples or [], all_examples, "merged")
    # Each panel is titled by its experiment folder name, so the stacked figure
    # reads as a direct comparison across captures.
    named_specs = [(example, timeline_spec_for(example)) for example in selected]
    plot_merged_timelines(named_specs, EXAMPLES_DIR / "merged_timeline.png")


@app.command()
def video(
    examples: Annotated[list[str] | None, typer.Argument(help="left_shoe_* folder name(s)")] = None,
    all_examples: Annotated[bool, typer.Option("--all", help="select every discovered example")] = False,
    fps: Annotated[int, typer.Option(help="frames per second")] = 30,
) -> None:
    """Render a real-time movie: timeline cursor + animated 3D geometry."""
    from _shared.video import render_video

    selected = _select_many(examples or [], all_examples, "video")
    # One experiment -> an individual video in its folder; several -> a merged
    # video at the examples root. Each panel is titled by its folder name.
    named_specs = [(example, timeline_spec_for(example)) for example in selected]
    output = EXAMPLES_DIR / selected[0] / "video" if len(selected) == 1 else EXAMPLES_DIR / "merged_video"
    render_video(named_specs, output, fps=fps)


# The canonical entry point is the unified ``retarget`` CLI, which attaches this
# module's ``app`` as the ``examples`` command group. This block only lets the
# gallery be run on its own from a checkout (``python examples/cli.py ...``).
if __name__ == "__main__":
    app()
