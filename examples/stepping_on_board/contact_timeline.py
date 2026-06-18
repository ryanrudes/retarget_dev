"""Plot a categorical contact timeline (air / ground / board / lost) for the sole.

The whole detection is declared once as a :class:`ContactPlan`: the sole patch
against an inferred floor AND the moving balance-board surface. ``apply_contact_plan``
runs every query, merges, and attaches the result to the mocap in one step -- there
are no loose ``SupportStateTrack`` objects to juggle. The categorical label is then
read straight off the patch with YOUR names via ``sole.support_state(...)``.

``fill_pose_gaps`` interpolates *short* pose gaps; long dropouts are left (and read
as "lost") rather than fabricated. Detection runs on the modeled rigid-body geometry
(``sole.points()`` is the segment pose, not raw markers).

    cd examples/stepping_on_board && PYTHONPATH=../../src:. python3 contact_timeline.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from retarget.contacts import (
    ContactDetectionConfig,
    ContactPlan,
    ContactQuery,
    apply_contact_plan,
    ground_plane,
    speed_limit,
)
from retarget.demo import fill_pose_gaps

from scene import get_demo


def main() -> None:
    demo = get_demo()
    mocap = demo.tracks["mocap"]

    # fill_pose_gaps interpolates untrustworthy pose gaps (too-few-markers OR a
    # garbage sample). A foot has a clear speed cap, so speed_limit(8 m/s) cleanly
    # flags teleports; they're skipped as anchors. Lower max_gap_time to leave long
    # gaps untrusted ("lost") instead of filling them.
    mocap = fill_pose_gaps(mocap, max_gap_time=10, jumps=speed_limit(8.0))

    # Declare the whole detection at once: the sole vs the floor AND the moving
    # board surface. apply_contact_plan detects + merges + attaches in one step;
    # the state then lives ON the mocap. Add more ContactQuery rows for the other
    # foot or the board-vs-ground and they all land on the one track.
    #
    # Use an explicit ground_plane here, NOT infer_ground(): the foot rests on the
    # floor (z~=0) AND the ~5cm-tall board (z~=0.05), and a single inferred plane
    # would fit through both and conflate them (board contacts would be masked by
    # ground). A fixed floor cleanly separates "ground" from "board".
    sole = mocap.subjects["left_foot"].segments["shoe"].patches["sole"]
    board = mocap.subjects["balance_board"].segments["board"].patches["surface"]
    plan = ContactPlan(
        queries=(
            ContactQuery(sole, against={"ground": ground_plane(0.0), "board": board}),
        ),
        config=ContactDetectionConfig(sensitivity=0.4, jumps=speed_limit(8.0)),
    )
    mocap = apply_contact_plan(mocap, plan)

    # Read the categorical state off a cheap patch view, with labels you choose.
    sole = mocap.subjects["left_foot"].segments["shoe"].patches["sole"]
    t = np.asarray(mocap.timestamps, dtype=np.float64)
    height = np.asarray(sole.points(), dtype=np.float64)[:, 2]  # world z of the sole point
    labels = sole.support_state(none="air", unknown="lost")  # "ground"/"board"/"air"/"lost"

    on_ground = labels == "ground"
    on_board = labels == "board"
    unknown = labels == "lost"
    interpolated = np.asarray(sole.pose_filled())  # frames synthesized by fill_pose_gaps

    fig, (ax_height, ax_state) = plt.subplots(
        2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    lo, hi = float(np.nanmin(height)), float(np.nanmax(height))
    # Measured pose (solid); interpolated pose (red dashed, so fabricated stretches
    # are honest); untrusted frames are blanked entirely.
    ax_height.plot(t, np.where(unknown, np.nan, height), color="tab:blue", lw=1.0, label="sole height (measured)")
    ax_height.plot(t, np.where(interpolated, height, np.nan), color="tab:red", lw=1.6, ls=(0, (2, 1)), label="interpolated")
    ax_height.fill_between(t, lo, hi, where=on_ground, step="post", color="tab:green", alpha=0.2, label="ground")
    ax_height.fill_between(t, lo, hi, where=on_board, step="post", color="tab:purple", alpha=0.2, label="board")
    ax_height.fill_between(t, lo, hi, where=unknown, step="post", color="tab:orange", alpha=0.3, label="lost (tracking)")
    ax_height.set_ylabel("sole height (m)")
    ax_height.set_title("Sole patch contact timeline (air / ground / board / lost)")
    # Keep the legend off the data.
    ax_height.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)

    ax_state.fill_between(t, 0, 1, where=on_ground, step="post", color="tab:green", alpha=0.5)
    ax_state.fill_between(t, 0, 1, where=on_board, step="post", color="tab:purple", alpha=0.5)
    ax_state.fill_between(t, 0, 1, where=unknown, step="post", color="tab:orange", alpha=0.6)
    ax_state.set_yticks([])
    ax_state.set_ylabel("state")
    ax_state.set_xlabel("time (s)")

    fig.tight_layout()

    air = ~on_ground & ~on_board & ~unknown
    print(
        f"sole contact timeline: {len(t)} frames, "
        f"{np.mean(on_ground):.0%} ground / "
        f"{np.mean(on_board):.0%} board / "
        f"{np.mean(air):.0%} air / "
        f"{np.mean(unknown):.0%} lost"
    )

    fig.savefig("contact_timeline.png", dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
