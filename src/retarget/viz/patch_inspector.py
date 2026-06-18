"""Static, top-down view of a patch's contact rectangle in its own frame.

`inspect_patch` answers "is the region I defined actually on the contact area, and
which way do width/height run?" -- it draws, looking straight down the patch normal:
the rectangle outline, the +X (forward) / +Y axes, the origin, and (if you pass the
segment-frame ``body_model``) the plane / origin / extent markers, each labeled and
projected into the patch plane.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from retarget.core.types import Vec3

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from retarget.core.schema.patch import Patch


def inspect_patch(
    patch: "Patch[Any]",
    marker_positions: Mapping[str, Vec3] | None = None,
    *,
    show: bool = True,
) -> "Figure":
    """Plot a bound rectangular ``patch`` top-down in its own frame.

    ``marker_positions`` is the segment-frame body model (e.g. the scene's
    ``read_marker_positions_from_vsk(...)`` mapping); when given, the plane markers
    (blue), origin markers (orange), and extent markers (purple) are drawn projected
    into the plane and labeled. Returns the matplotlib ``Figure``; pass ``show=False``
    to suppress the window (e.g. in tests / headless runs).
    """
    import matplotlib.pyplot as plt

    geometry = patch._geometry()  # segment -> patch
    rotation = np.asarray(geometry.rotation, dtype=np.float64)  # [x, y, normal]
    origin = np.asarray(geometry.translation, dtype=np.float64)

    region = patch.region
    if region is None:
        raise ValueError(f"Patch {patch.label!r} has no region to inspect")

    fig, ax = plt.subplots(figsize=(7, 7))

    # rectangle outline (already in patch local xy), closed
    boundary = np.asarray(region.boundary(), dtype=np.float64)
    loop = np.vstack([boundary, boundary[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color="tab:gray", lw=1.5, label="contact region")

    # origin + axes
    ax.plot(0.0, 0.0, "o", color="black", ms=6, label="origin")
    span = float(np.abs(boundary).max()) or 0.05
    ax.annotate("", xy=(0.4 * span, 0.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.text(0.42 * span, 0.0, "+x (forward)", color="tab:red", fontsize=8, va="center")
    ax.annotate("", xy=(0.0, 0.4 * span), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="tab:green"))
    ax.text(0.0, 0.42 * span, "+y", color="tab:green", fontsize=8, ha="center")

    # markers, projected into the plane, colored by role
    if marker_positions is not None:
        cal = patch.calibration
        roles: list[tuple[str, tuple[str, ...]]] = []
        if cal is not None:
            roles.append(("plane (tab:blue)", cal.markers))
            if cal.origin is not None:
                roles.append(("origin (tab:orange)", cal.origin.required_markers()))
            if cal.extent is not None:
                roles.append(("extent (tab:purple)", cal.extent.required_markers()))
        colors = {"plane (tab:blue)": "tab:blue",
                  "origin (tab:orange)": "tab:orange",
                  "extent (tab:purple)": "tab:purple"}
        for label, names in roles:
            pts = []
            for name in names:
                if name not in marker_positions:
                    continue
                local = (np.asarray(marker_positions[name], dtype=np.float64) - origin) @ rotation
                pts.append(local[:2])
                ax.text(local[0], local[1], f" {name}", fontsize=6, color=colors[label])
            if pts:
                arr = np.asarray(pts)
                ax.scatter(arr[:, 0], arr[:, 1], s=24, color=colors[label], label=label)

    ax.set_aspect("equal", "box")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"patch {patch.label!r}  (top-down, looking along -normal)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
