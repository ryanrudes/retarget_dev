"""Shared plotting for the contact-timeline examples.

Each ``left_shoe_*`` example owns its *contact-detection experiment* in its own
``experiment.py`` (which patch, against which supports, with which config) and
returns a :class:`TimelineSpec`. The presentation -- reading the categorical
state off the patch and rendering the height / speed / state timeline -- is
identical across examples and lives here.

The CLI builds the demo, hands the mocap to the example's experiment, and passes
the resulting :class:`TimelineSpec` to :func:`plot_contact_timeline`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch as LegendPatch

from retarget.contacts import CONTACT_CHANNELS, SupportFeatureChannels
from retarget.core import Patch, RectangularRegion
from retarget.demo import MocapTrack

from .console import console
from .schema import ExampleSubjects

# How each detection channel is labelled down the left edge of the timeline.
_CHANNEL_LABELS: Mapping[str, str] = {
    "clearance": "clearance",
    "normal_speed": "normal speed",
    "tangential_speed": "tangential speed",
    "quiet_activity": "quiet activity",
    "quiet_spread": "quiet spread",
    "angular_speed": "angular speed",
}
# Beyond ~3σ a single channel already collapses the contact confidence, so clip the heat
# there: everything from "3σ off" upward reads as fully "argues against contact".
_RESIDUAL_VMAX = 3.0


class CategoryStyle(NamedTuple):
    """How to draw one categorical support state on the timeline.

    ``key`` is the label value returned by ``patch.support_state(...)`` (e.g.
    ``"ground"``, ``"board"``, or the chosen ``unknown`` label); ``display`` is
    the legend text; ``color`` is any matplotlib color.
    """

    key: str
    display: str
    color: str


@dataclass(frozen=True)
class TimelineSpec:
    """The result of an example's contact-detection experiment, ready to plot.

    ``mocap`` is the gap-filled, contact-applied track; ``patch`` is the patch
    view whose categorical state is rendered. ``none_label``/``unknown_style``
    are passed through to ``patch.support_state(...)``; ``categories`` are the
    genuine contact supports (e.g. ground, board) drawn as filled bands.

    ``signals`` maps each support name to the per-frame
    :class:`~retarget.contacts.SupportFeatureChannels` behind its decision (from
    ``classify_feature_channels``). The timeline draws one heat strip per signal so the
    *why* of every contact is visible, not just the verdict.
    """

    mocap: MocapTrack[ExampleSubjects]
    patch: Patch[RectangularRegion]
    none_label: str
    unknown_style: CategoryStyle
    categories: Sequence[CategoryStyle]
    title: str
    signals: Mapping[str, SupportFeatureChannels]


@dataclass
class TimelinePanel:
    """Handles to a drawn timeline panel, for callers that animate it.

    ``axes`` are every stacked row (the per-signal heat strips plus the state bar), so a
    caller can drop a time cursor on each; ``t`` is the shared time axis (seconds);
    ``summary`` is the one-line coverage report.
    """

    axes: tuple[Axes, ...]
    t: npt.NDArray[np.float64]
    summary: str


def _ordered_supports(spec: TimelineSpec) -> list[str]:
    """Support names in a stable, readable order: declared categories first, then the rest."""
    ordered = [c.key for c in spec.categories if c.key in spec.signals]
    ordered += [name for name in spec.signals if name not in ordered]
    return ordered


def timeline_row_count(spec: TimelineSpec) -> int:
    """Total stacked rows: per support, one strip per channel plus confidence, then state."""
    per_support = len(CONTACT_CHANNELS) + 1  # channels + confidence
    return len(_ordered_supports(spec)) * per_support + 1  # + resolved-state bar


def _heat_strip(
    ax: Axes, t: npt.NDArray[np.float64], values: npt.NDArray[np.float64], *,
    cmap: str, vmin: float, vmax: float, label: str, label_color: str = "black",
) -> None:
    """Draw one signal as a horizontal heat band spanning the time axis."""
    ax.imshow(
        values[None, :], aspect="auto", extent=(float(t[0]), float(t[-1]), 0.0, 1.0),
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, color=label_color)


def _draw_timeline(target: Any, spec: TimelineSpec, title: str) -> TimelinePanel:
    """Draw one experiment's per-signal contact timeline into ``target``.

    Each support gets a block of heat strips -- one per detection channel (coloured by its
    standardized residual: green ≈ consistent with contact, red ≈ argues against) plus the
    fused confidence -- and a single resolved-state bar sits at the bottom. ``target`` is a
    ``Figure`` (single plot) or ``SubFigure`` (one panel of a merged plot).
    """
    patch = spec.patch
    t = np.asarray(spec.mocap.timestamps, dtype=np.float64)
    labels = patch.support_state(none=spec.none_label, unknown=spec.unknown_style.key)
    category_masks = [(c, labels == c.key) for c in spec.categories]
    unknown = labels == spec.unknown_style.key

    supports = _ordered_supports(spec)
    n_rows = timeline_row_count(spec)
    # Confidence/state bars read a touch taller than the thin channel strips.
    heights = ([1.0] * len(CONTACT_CHANNELS) + [1.3]) * len(supports) + [1.6]
    axes = target.subplots(n_rows, 1, sharex=True, gridspec_kw={"height_ratios": heights})
    axes = list(np.atleast_1d(axes))

    cat_color = {c.key: c.color for c in spec.categories}
    row = 0
    for name in supports:
        channels = spec.signals[name]
        header_color = cat_color.get(name, "black")
        for i, key in enumerate(CONTACT_CHANNELS):
            ax = axes[row]
            residual = np.abs(np.asarray(channels.residual[key], dtype=np.float64))
            _heat_strip(ax, t, residual, cmap="RdYlGn_r", vmin=0.0, vmax=_RESIDUAL_VMAX, label=_CHANNEL_LABELS[key])
            if i == 0:  # head each support block with its name, in the support's colour
                ax.set_title(f"vs {name}", loc="left", fontsize=8, fontweight="bold", color=header_color)
            row += 1
        # Fused contact confidence for this support: red (0) -> green (1).
        confidence = np.asarray(channels.confidence, dtype=np.float64)
        _heat_strip(axes[row], t, confidence, cmap="RdYlGn", vmin=0.0, vmax=1.0, label="confidence", label_color=header_color)
        row += 1

    # The resolved categorical state: which support actually won at each frame.
    ax_state = axes[row]
    for c, mask in category_masks:
        ax_state.fill_between(t, 0, 1, where=mask, step="post", color=c.color, alpha=0.9)
    ax_state.fill_between(t, 0, 1, where=unknown, step="post", color=spec.unknown_style.color, alpha=0.9)
    ax_state.set_yticks([])
    ax_state.set_ylim(0, 1)
    ax_state.set_xlim(float(t[0]), float(t[-1]))
    ax_state.set_ylabel("resolved", rotation=0, ha="right", va="center", fontsize=7, fontweight="bold")
    ax_state.set_xlabel("time (s)")
    # Only the supports actually resolved in THIS clip appear in the key (plus "lost" when
    # there is lost tracking). It sits at the bottom-left -- horizontal and clear of the
    # centred "time (s)" label -- and below the time axis, so it never steals width from the
    # timeline: a clip using fewer categories keeps the same physical length, just a shorter key.
    handles = [LegendPatch(facecolor=c.color, edgecolor="none", label=c.display) for c, mask in category_masks if bool(mask.any())]
    if bool(unknown.any()):
        handles.append(LegendPatch(facecolor=spec.unknown_style.color, edgecolor="none", label=spec.unknown_style.display))
    if handles:
        target.legend(handles=handles, loc="lower left", ncol=len(handles), fontsize=7, frameon=False)

    target.suptitle(
        f"{title}\nrows: detection signals (green = consistent with contact, red = argues against, "
        f"residual σ clipped at {_RESIDUAL_VMAX:g}); confidence 0–1; resolved state",
        fontsize=9,
    )

    air = np.ones(len(t), dtype=bool)
    for _, mask in category_masks:
        air &= ~mask
    air &= ~unknown
    coverage = " / ".join(f"{np.mean(mask):.0%} {c.display}" for c, mask in category_masks)
    summary = (
        f"{len(t)} frames, {coverage} / "
        f"{np.mean(air):.0%} {spec.none_label} / {np.mean(unknown):.0%} {spec.unknown_style.display}"
    )
    return TimelinePanel(axes=tuple(axes), t=t, summary=summary)


def _panel_height(spec: TimelineSpec) -> float:
    """Figure inches for one experiment's stack of thin heat strips plus headroom."""
    return 0.34 * timeline_row_count(spec) + 1.8


def plot_contact_timeline(spec: TimelineSpec, output_path: str | Path) -> None:
    """Render the per-signal contact timeline for ``spec.patch``."""
    fig = plt.figure(figsize=(17, _panel_height(spec)), layout="constrained")
    panel = _draw_timeline(fig, spec, spec.title)
    console.print(f"contact timeline: {panel.summary}")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    console.print(f"[green]saved[/] {output_path}")
    plt.show()


def plot_merged_timelines(
    named_specs: Sequence[tuple[str, TimelineSpec]], output_path: str | Path
) -> None:
    """Render several experiments' timelines stacked into one figure.

    Each ``(name, spec)`` pair becomes one panel headed by ``name`` (its
    experiment folder name), so the panels are directly comparable.
    """
    n = len(named_specs)
    height = sum(_panel_height(spec) for _, spec in named_specs)
    fig = plt.figure(figsize=(17, height), layout="constrained")
    # squeeze=False so a single experiment still yields an indexable column.
    subfigs = fig.subfigures(n, 1, squeeze=False).ravel()
    for subfig, (name, spec) in zip(subfigs, named_specs):
        panel = _draw_timeline(subfig, spec, name)
        console.print(f"[bold]{name}[/]: {panel.summary}")
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    console.print(f"[green]saved[/] {output_path}")
    plt.show()
