"""Tests for pose validity/coverage, the .where() resolver, and auto-unknown."""

from __future__ import annotations

import numpy as np

from retarget.contacts import classify_contacts, ground_plane, priority, speed_limit
from retarget.core import RigidTransform, SceneState, SegmentKey, SegmentPoseTrajectory
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame

from conftest import SEGMENT, SUBJECT, demo_segment, make_demo_subjects, make_mocap_track

_MARKERS = ("heel", "toe", "mid")


def _resting_track_with_dropout(n: int = 40, dt: float = 0.05, drop: range = range(15, 30)) -> MocapTrack:
    """A foot resting at z=0 the whole time, with markers occluded during ``drop``."""
    t = np.arange(n, dtype=np.float64) * dt
    poses = tuple(RigidTransform.from_translation(np.zeros(3)) for _ in range(n))
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
                    occluded=(i in drop and name != "heel"),
                )
                for name in _MARKERS
            ),
        )
        for i in range(n)
    )
    return MocapTrack(
        subjects=make_demo_subjects(), state=state, timestamps=t, marker_frames=frames
    )


def _resting_track_with_jump(n: int = 20, dt: float = 0.02, jump_at: int = 5) -> MocapTrack:
    """A foot resting at z=0 with a one-frame teleport to z=1 (fully covered)."""
    t = np.arange(n, dtype=np.float64) * dt
    z = np.zeros(n, dtype=np.float64)
    z[jump_at] = 1.0
    poses = tuple(
        RigidTransform.from_translation(np.array([0.0, 0.0, float(zz)])) for zz in z
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
                    occluded=False,
                )
                for name in _MARKERS
            ),
        )
        for i in range(n)
    )
    return MocapTrack(
        subjects=make_demo_subjects(), state=state, timestamps=t, marker_frames=frames
    )


def test_implausible_jump_mask_flags_teleport() -> None:
    from retarget.core.formats import implausible_jump_mask

    t = np.arange(6, dtype=np.float64) * 0.01
    pos = np.zeros((6, 3))
    pos[3] = [0.0, 0.0, 1.0]  # teleport at index 3 -> 100 m/s steps in and out
    mask = implausible_jump_mask(pos, t, max_speed=10.0)
    assert mask[2] and mask[3] and mask[4]
    assert not mask[0]
    assert not implausible_jump_mask(pos, t, None).any()  # disabled


def test_valid_flags_implausible_jump() -> None:
    track = _resting_track_with_jump()
    segment = demo_segment(track)
    assert segment.valid().all()  # fully covered, no jump detector -> all valid
    flagged = segment.valid(jumps=speed_limit(10.0))
    assert not flagged[5]  # the teleport is garbage despite full marker coverage


def test_valid_flags_rotation_glitch_via_patch_points() -> None:
    # The segment origin never moves, but one frame has a wild rotation that
    # flings the sole patch point -- caught only because valid() measures the
    # jump on the patch points, not the origin translation.
    from scipy.spatial.transform import Rotation

    n, dt = 10, 0.02
    t = np.arange(n, dtype=np.float64) * dt
    rotations = [np.eye(3) for _ in range(n)]
    rotations[5] = Rotation.from_euler("x", 90.0, degrees=True).as_matrix()
    poses = tuple(
        RigidTransform.from_rotation_translation(rotations[i], np.zeros(3)) for i in range(n)
    )
    state = SceneState(
        segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)}
    )
    frames = tuple(
        ViconMarkersFrame(
            stamp_seconds=float(t[i]),
            markers=tuple(
                MarkerObservation(
                    marker_name=name, subject_name=SUBJECT, segment_name=SEGMENT,
                    position_world=np.zeros(3), occluded=False,
                )
                for name in _MARKERS
            ),
        )
        for i in range(n)
    )
    track = MocapTrack(subjects=make_demo_subjects(), state=state, timestamps=t, marker_frames=frames)
    segment = demo_segment(track)
    assert segment.valid().all()  # no jump detector -> all valid
    assert not segment.valid(jumps=speed_limit(5.0))[5]


def test_classify_flags_jump_as_unknown() -> None:
    from retarget.contacts import ContactDetectionConfig

    track = _resting_track_with_jump()
    states = classify_contacts(
        demo_segment(track).patches.sole,
        against={"ground": ground_plane(0.0)},
        config=ContactDetectionConfig(jumps=speed_limit(10.0)),
    )
    track = track.with_support_states(states)
    labels = demo_segment(track).patches.sole.support_state(unknown="unknown")
    assert labels[5] == "unknown"


def test_pose_coverage_counts_visible_markers() -> None:
    track = make_mocap_track(num_timesteps=3)  # observes only "heel" of 3 markers
    segment = demo_segment(track)
    np.testing.assert_allclose(segment.pose_coverage(), np.full(3, 1.0 / 3.0))
    assert segment.valid(min_coverage=0.3).all()
    assert not segment.valid(min_coverage=0.5).any()


def test_patch_validity_matches_segment() -> None:
    track = make_mocap_track(num_timesteps=3)
    segment = demo_segment(track)
    sole = segment.patches.sole
    np.testing.assert_allclose(sole.pose_coverage(), segment.pose_coverage())
    np.testing.assert_array_equal(sole.valid(min_coverage=0.3), segment.valid(min_coverage=0.3))


def test_pose_coverage_all_ones_without_marker_frames() -> None:
    track = make_mocap_track(num_timesteps=4, include_markers=False)
    segment = demo_segment(track)
    np.testing.assert_allclose(segment.pose_coverage(), np.ones(4))
    assert segment.valid(min_coverage=0.9).all()  # unknown coverage -> assumed valid


def test_resolver_where_overrides_invalid_frames() -> None:
    contacts = {"ground": np.array([True, True, False, False], dtype=np.bool_)}
    invalid = np.array([False, False, True, False], dtype=np.bool_)
    resolver = priority(none_label="air").where(invalid, "unknown")
    labels = resolver(contacts, {})
    assert list(labels) == ["ground", "ground", "unknown", "air"]


def test_resolver_where_rejects_shape_mismatch() -> None:
    contacts = {"ground": np.array([True, False], dtype=np.bool_)}
    resolver = priority().where(np.array([True, False, True], dtype=np.bool_))
    try:
        resolver(contacts, {})
    except ValueError:
        return
    raise AssertionError("expected a ValueError for mismatched invalid-mask shape")


def test_classify_emits_unknown_on_sustained_dropout() -> None:
    track = _resting_track_with_dropout()
    states = classify_contacts(
        demo_segment(track).patches.sole, against={"ground": ground_plane(0.0)}
    )
    track = track.with_support_states(states)
    sole = demo_segment(track).patches.sole
    labels = sole.support_state(none="air", unknown="lost")

    # Resting on the floor with full coverage -> ground; the dropout -> your label.
    assert labels[2] == "ground"
    assert labels[22] == "lost"
    assert "lost" in set(labels)


def test_unknown_is_opt_in_by_naming() -> None:
    # Untrusted frames only become a label if you name one -- nothing magic appears.
    track = _resting_track_with_dropout()
    states = classify_contacts(
        demo_segment(track).patches.sole, against={"ground": ground_plane(0.0)}
    )
    track = track.with_support_states(states)
    sole = demo_segment(track).patches.sole
    assert "lost" not in set(sole.support_state(none="air"))  # not named -> not surfaced
    assert "lost" in set(sole.support_state(none="air", unknown="lost"))
