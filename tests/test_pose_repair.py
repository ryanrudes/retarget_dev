"""Tests for fill_pose_gaps (short-gap segment-pose interpolation)."""

from __future__ import annotations

import numpy as np

from retarget.core import RigidTransform, SceneState, SegmentKey, SegmentPoseTrajectory
from retarget.demo import fill_pose_gaps
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame

from conftest import SEGMENT, SUBJECT, demo_segment, make_demo_subjects

_MARKERS = ("heel", "toe", "mid")


def _track_with_gap(z_values: list[float], visible: list[set[str]]) -> MocapTrack:
    t = np.arange(len(z_values), dtype=np.float64) * 0.1
    poses = tuple(
        RigidTransform.from_translation(np.array([0.0, 0.0, float(z)])) for z in z_values
    )
    state = SceneState(
        segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)}
    )
    frames = tuple(
        ViconMarkersFrame(
            stamp_seconds=float(t[i]),
            markers=tuple(
                MarkerObservation(
                    marker_name=name,
                    subject_name=SUBJECT,
                    segment_name=SEGMENT,
                    position_world=np.zeros(3),
                    occluded=name not in visible[i],
                )
                for name in _MARKERS
            ),
        )
        for i in range(len(z_values))
    )
    return MocapTrack(
        subjects=make_demo_subjects(), state=state, timestamps=t, marker_frames=frames
    )


def test_fill_pose_gaps_interpolates_short_gap() -> None:
    full = {"heel", "toe", "mid"}
    visible = [full, full, set(), full, full]  # frame 2 fully occluded
    track = _track_with_gap([0.0, 1.0, 99.0, 3.0, 4.0], visible)

    assert list(demo_segment(track).valid()) == [True, True, False, True, True]

    repaired = fill_pose_gaps(track, max_gap_time=0.25)
    z = demo_segment(repaired).translations()[:, 2]
    # The garbage frame is replaced by the midpoint of its trusted neighbors.
    np.testing.assert_allclose(z, [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-6)


def test_fill_pose_gaps_leaves_long_gap_untouched() -> None:
    full = {"heel", "toe", "mid"}
    visible = [full, full, set(), full, full]
    track = _track_with_gap([0.0, 1.0, 99.0, 3.0, 4.0], visible)
    # Gap span is t[3]-t[1] = 0.2s, longer than max_gap_time -> left as-is.
    repaired = fill_pose_gaps(track, max_gap_time=0.05)
    z = demo_segment(repaired).translations()[:, 2]
    assert z[2] == 99.0


def test_fill_marks_filled_frames_as_valid() -> None:
    full = {"heel", "toe", "mid"}
    visible = [full, full, set(), full, full]
    track = _track_with_gap([0.0, 1.0, 99.0, 3.0, 4.0], visible)
    assert not demo_segment(track).valid()[2]  # before: frame 2 untrusted

    repaired = fill_pose_gaps(track, max_gap_time=0.25)
    segment = demo_segment(repaired)
    # A filled frame now counts as valid / fully covered (so it won't read "unknown").
    assert segment.valid().all()
    assert segment.pose_coverage()[2] == 1.0
    # ...and is reported as synthesized (not measured) via pose_filled().
    assert segment.pose_filled()[2]
    assert segment.patches["sole"].pose_filled()[2]
    assert not segment.pose_filled()[0]


def test_unfilled_long_gap_stays_invalid() -> None:
    full = {"heel", "toe", "mid"}
    visible = [full, full, set(), full, full]
    track = _track_with_gap([0.0, 1.0, 99.0, 3.0, 4.0], visible)
    # Gap span 0.2s > max_gap_time -> not filled, so the frame stays untrusted.
    repaired = fill_pose_gaps(track, max_gap_time=0.05)
    assert not demo_segment(repaired).valid()[2]
    assert demo_segment(repaired).pose_coverage()[2] < 0.5


def test_fill_pose_gaps_noop_when_all_valid() -> None:
    full = {"heel", "toe", "mid"}
    track = _track_with_gap([0.0, 1.0, 2.0, 3.0], [full, full, full, full])
    repaired = fill_pose_gaps(track, max_gap_time=1.0)
    z = demo_segment(repaired).translations()[:, 2]
    np.testing.assert_allclose(z, [0.0, 1.0, 2.0, 3.0])
