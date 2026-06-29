from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    Segments,
    SegmentTarget,
    Subject,
    Subjects,
)
from retarget.demo import MocapTrack


@dataclass(frozen=True, slots=True)
class _Markers(Markers):
    a: Marker


@dataclass(frozen=True, slots=True)
class _Patches(Patches):
    p: Patch


@dataclass(frozen=True, slots=True)
class _Segments(Segments):
    shoe: Segment[_Markers, _Patches]


@dataclass(frozen=True, slots=True)
class _Subjects(Subjects):
    left_shoe: Subject[_Segments]
    right_shoe: Subject[_Segments]


def _segment() -> Segment[_Markers, _Patches]:
    return Segment(markers=_Markers(a=Marker(mocap_name="a")), patches=_Patches(p=Patch(label="p")))


def _subjects() -> _Subjects:
    return _Subjects(
        left_shoe=Subject(segments=_Segments(shoe=_segment())),
        right_shoe=Subject(segments=_Segments(shoe=_segment())),
    )


def test_segment_key_is_string_based_and_hashable() -> None:
    key = SegmentKey("left_shoe", "shoe")
    assert key.subject == "left_shoe"
    assert key.segment == "shoe"
    assert {key: 1}[SegmentKey("left_shoe", "shoe")] == 1


def test_same_segment_name_across_subjects_does_not_collide() -> None:
    state = SceneState(
        segment_poses={
            SegmentKey("left_shoe", "shoe"): SegmentPoseTrajectory(
                poses=(RigidTransform.from_translation(np.array([0.0, 0.0, 0.1])),)
            ),
            SegmentKey("right_shoe", "shoe"): SegmentPoseTrajectory(
                poses=(RigidTransform.from_translation(np.array([0.0, 0.0, 0.9])),)
            ),
        }
    )
    track = MocapTrack(
        subjects=_subjects(), state=state, timestamps=np.array([0.0])
    )

    left = track.subjects["left_shoe"].segments["shoe"]
    right = track.subjects["right_shoe"].segments["shoe"]

    np.testing.assert_allclose(left.translations()[0], np.array([0.0, 0.0, 0.1]))
    np.testing.assert_allclose(right.translations()[0], np.array([0.0, 0.0, 0.9]))
    assert left.segment_target() == SegmentTarget("left_shoe", "shoe")
    assert right.segment_target() == SegmentTarget("right_shoe", "shoe")
