"""Static, top-down view of a patch's contact region in its own plane.

``inspect_patch`` answers "is the region I defined actually on the contact area?" -- it draws,
looking straight down the patch normal: the region outline (the patch's fungeom ``Face``
region, in plane-local 2-D) and, if you pass the segment-frame body model, the markers
projected into that plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fungeom import Point3

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from retarget.core.schema.patch import Patch
    from retarget.core.types import Vec3


def inspect_patch(
    patch: Patch,
    marker_positions: Mapping[str, Vec3] | None = None,
    *,
    show: bool = True,
) -> Figure:
    """Plot a bound geometry-authored ``patch`` top-down in its own plane.

    ``marker_positions`` is the segment-frame body model (e.g. the scene's
    ``read_marker_positions_from_vsk(...)`` mapping); when given, the markers are projected
    into the patch plane and labeled. Returns the matplotlib ``Figure``; pass ``show=False``
    to suppress the window (e.g. in tests / headless runs).
    """
    import matplotlib.pyplot as plt

    face = patch.face()
    plane = face.plane()
    vertices = face.region().vertices().resolve()
    boundary = np.array(
        [np.asarray(vertices.at(key).coord, dtype=np.float64) for key in vertices.roster],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    loop = np.vstack([boundary, boundary[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color="tab:gray", lw=1.5, label="contact region")

    span = float(np.abs(boundary).max()) or 0.05
    ax.plot(0.0, 0.0, "o", color="black", ms=6, label="plane origin")
    ax.annotate("", xy=(0.4 * span, 0.0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax.text(0.42 * span, 0.0, "+u", color="tab:red", fontsize=8, va="center")
    ax.annotate("", xy=(0.0, 0.4 * span), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="tab:green"))
    ax.text(0.0, 0.42 * span, "+v", color="tab:green", fontsize=8, ha="center")

    # markers projected into the patch plane (no per-resolver roles -- the open algebra has no
    # fixed resolver menu, so markers are drawn as one set).
    if marker_positions is not None:
        projected = []
        for name, position in marker_positions.items():
            coords = np.asarray(position, dtype=np.float64)
            local = np.asarray(
                plane.to_local(Point3.at(float(coords[0]), float(coords[1]), float(coords[2]))).resolve().coord,
                dtype=np.float64,
            )
            projected.append(local)
            ax.text(local[0], local[1], f" {name}", fontsize=6, color="tab:blue")
        if projected:
            arr = np.asarray(projected)
            ax.scatter(arr[:, 0], arr[:, 1], s=24, color="tab:blue", label="markers")

    ax.set_aspect("equal", "box")
    ax.grid(True, ls=":", alpha=0.4)
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    ax.set_title(f"patch {patch.label!r}  (top-down, looking along -normal)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
