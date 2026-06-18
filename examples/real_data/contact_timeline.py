"""Plot a robust contact timeline (air / ground / lost) for the sole patch.

The workflow is the intended one: build the mocap as a chain of immutable
transforms (``fill_pose_gaps`` -> ``with_support_states``), so the detected
contact state lives ON the mocap and travels with the demo. Then read it back off
a cheap patch view (``sole.support_state(...)``) -- no loose tracks to juggle.

``classify`` flags untrustworthy frames itself (coverage below ``config.min_coverage``
or a ``config.jumps`` garbage sample); they surface under whatever ``unknown`` label
you ask for. ``fill_pose_gaps`` interpolates *short* pose gaps; long dropouts are
left (and read as unknown) rather than fabricated. Detection runs on the modeled
rigid-body geometry (``sole.points()`` is the segment pose, not raw markers).

    cd examples/real_data && PYTHONPATH=../../src:. python3 contact_timeline.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from retarget.contacts import (
    ContactDetectionConfig,
    ContactPlan,
    ContactQuery,
    apply_contact_plan,
    infer_ground,
    speed_limit,
)
from retarget.demo import fill_pose_gaps

from scene import get_demo


def main() -> None:
    demo = get_demo()
    mocap = demo.tracks["mocap"]

    # fill_pose_gaps interpolates untrustworthy pose gaps (too-few-markers OR a
    # garbage sample). A foot has a clear speed cap, so speed_limit(8 m/s) cleanly
    # flags teleports; they're skipped as anchors. max_gap_time bounds how long a
    # gap we'll interpolate across: only brief dropouts are repaired, so a sustained
    # loss (here ~0.66s) is left unfilled and surfaces as "lost" rather than being
    # silently bridged with a straight line. Raise it to fill more aggressively.
    mocap = fill_pose_gaps(mocap, max_gap_time=0.3, jumps=speed_limit(8.0))

    # Declare the detection once and apply it in one step. Scoping the query to the
    # whole mocap detects every geometry patch against an inferred floor, so we
    # needn't pre-fetch a patch; the state then lives ON the mocap.

    config = ContactDetectionConfig(sensitivity=0.4, jumps=speed_limit(8.0))

    queries = (
        ContactQuery(mocap, against={"ground": infer_ground()}),
    )

    plan = ContactPlan(queries=queries, config=config)
    mocap = apply_contact_plan(mocap, plan)

    # Now derive a cheap patch view from the finished mocap and read everything off
    # it: geometry, YOUR contact labels, and which frames were synthesized.
    sole = mocap.subjects["left_foot"].segments["shoe"].patches["sole"]
    t = np.asarray(mocap.timestamps, dtype=np.float64)
    height = np.asarray(sole.points(), dtype=np.float64)[:, 2]  # world z of the sole point
    labels = sole.support_state(none="air", unknown="lost")  # (T,) "air"/"ground"/"lost"

    on_ground = labels == "ground"
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
    ax_height.fill_between(t, lo, hi, where=unknown, step="post", color="tab:orange", alpha=0.3, label="unknown (tracking lost)")
    ax_height.set_ylabel("sole height (m)")
    ax_height.set_title("Sole patch contact timeline (air / ground / unknown)")
    # Keep the legend off the data.
    ax_height.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, borderaxespad=0.0)

    ax_state.fill_between(t, 0, 1, where=on_ground, step="post", color="tab:green", alpha=0.5)
    ax_state.fill_between(t, 0, 1, where=unknown, step="post", color="tab:orange", alpha=0.6)
    ax_state.set_yticks([])
    ax_state.set_ylabel("state")
    ax_state.set_xlabel("time (s)")

    fig.tight_layout()

    print(
        f"sole contact timeline: {len(t)} frames, "
        f"{np.mean(on_ground):.0%} ground / "
        f"{np.mean(~on_ground & ~unknown):.0%} air / "
        f"{np.mean(unknown):.0%} unknown"
    )

    fig.savefig("contact_timeline.png", dpi=120, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
