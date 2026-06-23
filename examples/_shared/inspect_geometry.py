"""Side-view (world Z) sanity check of the sole / board patch planes vs the markers.

Stationary recording, so we read frame 0. Shows, in world coordinates:
  * raw foot markers + the fitted sole contact plane (with its outward normal),
  * raw board markers + the fitted board surface plane (with its outward normal),
  * the assumed ground plane at z=0.

The CLI builds the demo and calls :func:`main`; ``output_path`` is where the
side-view PNG is written.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from rich.table import Table

from retarget.demo import Demonstration

from .console import console
from .schema import ExampleTracks

FRAME = 0


def seg_marker_world(segment) -> dict[str, np.ndarray]:
    # modeled=True -> clean rigid-body world positions (observed are NaN/garbage when
    # a marker is briefly unobserved, e.g. at frame 0).
    return {
        name: np.asarray(segment.markers[name].positions(modeled=True), dtype=np.float64)[FRAME]
        for name in segment.markers.keys()
    }


def main(demo: Demonstration[ExampleTracks], output_path: str | Path) -> None:
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

    # ---- figure: two side views (X-Z and Y-Z) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (h, hi) in zip(axes, [(0, "x"), (1, "y")]):
        # markers
        ax.scatter([p[h] for p in foot.values()], [p[2] for p in foot.values()],
                   c="tab:blue", s=25, label="foot markers")
        ax.scatter([p[h] for p in deck.values()], [p[2] for p in deck.values()],
                   c="tab:purple", s=25, marker="s", label="board markers")
        # planes drawn as the span of their boundary corners
        ax.plot([sole_c[:, h].min(), sole_c[:, h].max()], [sole_o[2], sole_o[2]],
                color="tab:blue", lw=2.5, label="sole plane")
        ax.plot([surf_c[:, h].min(), surf_c[:, h].max()], [surf_o[2], surf_o[2]],
                color="tab:purple", lw=2.5, label="board surface plane")
        # outward normals
        ax.annotate("", xy=(sole_o[h] + 0.04 * sole_n[h], sole_o[2] + 0.04 * sole_n[2]),
                    xytext=(sole_o[h], sole_o[2]), arrowprops=dict(arrowstyle="->", color="tab:blue"))
        ax.annotate("", xy=(surf_o[h] + 0.04 * surf_n[h], surf_o[2] + 0.04 * surf_n[2]),
                    xytext=(surf_o[h], surf_o[2]), arrowprops=dict(arrowstyle="->", color="tab:purple"))
        # ground
        ax.axhline(0.0, color="tab:green", ls="--", lw=1.5, label="ground z=0")
        ax.set_xlabel(f"world {hi} (m)")
        ax.set_ylabel("world z (m)")
        ax.set_title(f"side view: {hi}-z")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Roller-board geometry: sole & board planes vs markers and z=0")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    console.print(f"[green]saved[/] {output_path}")
    plt.show()
