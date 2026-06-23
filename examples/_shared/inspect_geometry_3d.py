"""3D sanity check of the sole / board patch planes vs the markers.

Stationary recording, so we read frame 0. Shows, in world coordinates:
  * foot markers + the fitted sole contact plane (with its outward normal),
  * board markers + the fitted board surface plane (with its outward normal),
  * the assumed ground plane at z=0.

The CLI builds the demo and calls :func:`main`.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from rich.table import Table

from retarget.demo import Demonstration

from .console import console
from .schema import ExampleTracks

FRAME = 0
NORMAL_SCALE = 0.04


def seg_marker_world(segment) -> dict[str, np.ndarray]:
    # modeled=True -> clean rigid-body world positions (observed are NaN/garbage when
    # a marker is briefly unobserved, e.g. at frame 0).
    return {
        name: np.asarray(segment.markers[name].positions(modeled=True), dtype=np.float64)[FRAME]
        for name in segment.markers.keys()
    }


def _equal_aspect(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    half_span = float((maxs - mins).max()) / 2.0
    half_span = max(half_span, 1e-3) * 1.1
    lo = center - half_span
    hi = center + half_span
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((1, 1, 1))


def main(demo: Demonstration[ExampleTracks]) -> None:
    mocap = demo.tracks["mocap"]
    shoe = mocap.subjects["left_foot"].segments["shoe"]
    board = mocap.subjects["balance_board"].segments["board"]
    sole = shoe.patches["sole"]
    surface = board.patches["surface"]

    foot = seg_marker_world(shoe)
    deck = seg_marker_world(board)

    sole_o = np.asarray(sole.points(), dtype=np.float64)[FRAME]
    sole_n = np.asarray(sole.normals(), dtype=np.float64)[FRAME]
    sole_c = np.asarray(sole.boundary_points(), dtype=np.float64)[FRAME]  # (4,3)
    surf_o = np.asarray(surface.points(), dtype=np.float64)[FRAME]
    surf_n = np.asarray(surface.normals(), dtype=np.float64)[FRAME]
    surf_c = np.asarray(surface.boundary_points(), dtype=np.float64)[FRAME]

    # ---- numbers ----
    foot_z = {k: v[2] for k, v in foot.items()}
    lowest_foot = min(foot_z, key=foot_z.get)
    lowest_all = min([*[v[2] for v in foot.values()], *[v[2] for v in deck.values()]])

    table = Table(title="world Z heights (m), frame 0", title_style="bold", header_style="bold cyan")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_column("note", style="dim")
    table.add_row("sole patch plane z", f"{sole_o[2]:+.4f}", f"outward normal z = {sole_n[2]:+.3f}")
    table.add_row("board surface plane z", f"{surf_o[2]:+.4f}", f"outward normal z = {surf_n[2]:+.3f}")
    table.add_row("sole plane - board plane", f"{sole_o[2]-surf_o[2]:+.4f}", "neg => sole sits BELOW the board surface")
    table.add_section()
    table.add_row("lowest foot marker", f"{foot_z[lowest_foot]:+.4f}", repr(lowest_foot))
    table.add_row("foot markers z range", f"{min(foot_z.values()):+.4f} .. {max(foot_z.values()):+.4f}", "")
    table.add_row("board markers z range", f"{min(v[2] for v in deck.values()):+.4f} .. {max(v[2] for v in deck.values()):+.4f}", "")
    table.add_row("lowest of ALL markers", f"{lowest_all:+.4f}", "")
    table.add_section()
    table.add_row("sole plane vs lowest foot marker", f"{sole_o[2]-foot_z[lowest_foot]:+.4f}", "physical sole bottom should be ~ here")
    console.print(table)

    # ---- figure: single 3D view ----
    foot_pts = np.stack(list(foot.values()))
    deck_pts = np.stack(list(deck.values()))
    all_pts = np.concatenate([foot_pts, deck_pts, sole_c, surf_c], axis=0)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(foot_pts[:, 0], foot_pts[:, 1], foot_pts[:, 2],
               c="tab:blue", s=35, label="foot markers")
    ax.scatter(deck_pts[:, 0], deck_pts[:, 1], deck_pts[:, 2],
               c="tab:purple", s=35, marker="s", label="board markers")

    sole_rect = Poly3DCollection([sole_c], facecolor="tab:blue", alpha=0.35, edgecolor="tab:blue", linewidths=2)
    surf_rect = Poly3DCollection([surf_c], facecolor="tab:purple", alpha=0.35, edgecolor="tab:purple", linewidths=2)
    ax.add_collection3d(sole_rect)
    ax.add_collection3d(surf_rect)

    for origin, normal, color in ((sole_o, sole_n, "tab:blue"), (surf_o, surf_n, "tab:purple")):
        ax.quiver(origin[0], origin[1], origin[2],
                  normal[0], normal[1], normal[2],
                  length=NORMAL_SCALE, normalize=True, color=color, arrow_length_ratio=0.35)

    # ground plane at z=0, spanning the scene footprint
    x_lo, x_hi = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_lo, y_hi = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
    ground = np.array([
        [x_lo, y_lo, 0.0],
        [x_hi, y_lo, 0.0],
        [x_hi, y_hi, 0.0],
        [x_lo, y_hi, 0.0],
    ])
    ground_rect = Poly3DCollection([ground], facecolor="tab:green", alpha=0.15, edgecolor="tab:green", linewidths=1.5, linestyle="--")
    ax.add_collection3d(ground_rect)

    _equal_aspect(ax, all_pts)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("world z (m)")
    ax.set_title("Roller-board geometry: sole & board planes vs markers and z=0")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    plt.show()
