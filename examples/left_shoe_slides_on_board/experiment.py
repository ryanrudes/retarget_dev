"""Contact-detection experiment for this example: the sole vs floor AND board.

This file OWNS the experiment -- which patch is tested, against which supports,
and with which detection / gap-filling config. Presentation is shared (the CLI
renders the :class:`TimelineSpec` this returns via ``_shared.contact_timeline``),
so each example can tune its own detection without touching the plotting.

Run it with the example CLI::

    retarget examples timeline <this_example_folder>

The whole detection is declared once as a :class:`ContactPlan`: the sole patch
against an inferred floor AND the moving balance-board surface. ``apply_contact_plan``
runs every query, merges, and attaches the result to the mocap in one step -- there
are no loose ``SupportStateTrack`` objects to juggle. The categorical label is then
read straight off the patch with YOUR names via ``sole.support_state(...)``.

``fill_pose_gaps`` interpolates *short* pose gaps; long dropouts are left (and read
as "lost") rather than fabricated. Detection runs on the modeled rigid-body geometry
(``sole.points()`` is the segment pose, not raw markers).
"""

from __future__ import annotations

from retarget.contacts import (
    ContactDetectionConfig,
    ContactPlan,
    ContactQuery,
    apply_contact_plan,
    classify_feature_channels,
    ground_plane,
    speed_limit,
)
from retarget.demo import MocapTrack, fill_pose_gaps

from _shared.contact_timeline import CategoryStyle, TimelineSpec
from _shared.schema import ExampleSubjects


def build_timeline(mocap: MocapTrack[ExampleSubjects]) -> TimelineSpec:
    # fill_pose_gaps interpolates untrustworthy pose gaps (too-few-markers OR a
    # garbage sample). A foot has a clear speed cap, so speed_limit(8 m/s) cleanly
    # flags teleports; they're skipped as anchors. Lower max_gap_time to leave long
    # gaps untrusted ("lost") instead of filling them.
    mocap = fill_pose_gaps(mocap, max_gap_time=0.1, jumps=speed_limit(8.0))

    # Declare the whole detection at once: the sole vs the floor AND the moving
    # board surface. apply_contact_plan detects + merges + attaches in one step;
    # the state then lives ON the mocap. Add more ContactQuery rows for the other
    # foot or the board-vs-ground and they all land on the one track.
    #
    # Use an explicit ground_plane here, NOT infer_ground(): the foot rests on the
    # floor (z~=0) AND the ~5cm-tall board (z~=0.05), and a single inferred plane
    # would fit through both and conflate them. A fixed floor cleanly separates
    # "ground" from "board".
    #
    # No per-scene tuning: detection scores each support by how many measured-noise
    # σ its clearance/motion sit from "resting contact". The shoe is ~4.4 cm above
    # the floor (tens of σ -> ~0 confidence) and ~0 from the board (high confidence),
    # so the default most-confident resolver picks "board" for any confidence level.
    against = dict(
        ground = ground_plane(0.0),
        board = mocap.subjects["balance_board"].segments["board"].patches["surface"],
    )

    queries = (
        ContactQuery(
            scope=mocap.subjects["left_foot"].segments["shoe"].patches["sole"],
            against=against,
        ),
    )
    config = ContactDetectionConfig(jumps=speed_limit(8.0))
    plan = ContactPlan(queries=queries, config=config)
    mocap = apply_contact_plan(mocap, plan)

    sole = mocap.subjects["left_foot"].segments["shoe"].patches["sole"]

    # The per-frame signals behind each decision feed the timeline's per-signal heat strips.
    signals = classify_feature_channels(sole, against=against, config=config)[sole.target]

    # Read the categorical state off a cheap patch view, with labels you choose,
    # and hand the shared plotter the category styling for this experiment.
    return TimelineSpec(
        mocap=mocap,
        patch=sole,
        signals=signals,
        none_label="air",
        unknown_style=CategoryStyle("lost", "lost (tracking)", "tab:orange"),
        categories=(
            CategoryStyle("ground", "ground", "tab:green"),
            CategoryStyle("board", "board", "tab:purple"),
        ),
        title="Sole patch contact timeline (air / ground / board / lost)",
    )
