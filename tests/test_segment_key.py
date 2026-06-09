from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from retarget.core import MarkerSetSpec, SceneSpec, SegmentSpec, SubjectSpec, Z_UP_AXES
from retarget.core.enums import MarkerId, PatchId, SegmentId, SubjectId
from retarget.core.keys import SegmentKey
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.transform import RigidTransform
from retarget.core.views import SceneView, SubjectView


class _LeftShoeSubject(SubjectId):
    LEFT_SHOE = "left_shoe"


class _RightShoeSubject(SubjectId):
    RIGHT_SHOE = "right_shoe"


class _ShoeSegment(SegmentId):
    SHOE = "shoe"


class _TestMarkerId(MarkerId):
    A = "a"


class _TestPatchId(PatchId):
    P = "p"


SHOE_SEGMENT_SPEC = SegmentSpec(
    segment=_ShoeSegment.SHOE,
    marker_type=_TestMarkerId,
    patch_type=_TestPatchId,
    axis_convention=Z_UP_AXES,
    marker_set=MarkerSetSpec(marker_type=_TestMarkerId),
)


@dataclass(frozen=True, slots=True)
class _LeftShoeSubjectSpec(SubjectSpec):
    shoe: SegmentSpec[_TestMarkerId, _TestPatchId]

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        yield self.shoe


@dataclass(frozen=True, slots=True)
class _RightShoeSubjectSpec(SubjectSpec):
    shoe: SegmentSpec[_TestMarkerId, _TestPatchId]

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        yield self.shoe


@dataclass(frozen=True, slots=True)
class _TestSceneSpec(SceneSpec):
    left_shoe: _LeftShoeSubjectSpec
    right_shoe: _RightShoeSubjectSpec

    def iter_subjects(self) -> Iterable[SubjectSpec]:
        yield self.left_shoe
        yield self.right_shoe


TEST_SCENE_SPEC = _TestSceneSpec(
    left_shoe=_LeftShoeSubjectSpec(
        subject=_LeftShoeSubject.LEFT_SHOE,
        shoe=SHOE_SEGMENT_SPEC,
    ),
    right_shoe=_RightShoeSubjectSpec(
        subject=_RightShoeSubject.RIGHT_SHOE,
        shoe=SHOE_SEGMENT_SPEC,
    ),
)


def _trajectory_at_z(z: float) -> SegmentPoseTrajectory:
    return SegmentPoseTrajectory(
        poses=(
            RigidTransform.from_rotation_translation(
                rotation=np.eye(3),
                translation=np.array([0.0, 0.0, z]),
            ),
        ),
    )


def test_same_segment_id_across_subjects_does_not_collide() -> None:
    left_key = SegmentKey(_LeftShoeSubject.LEFT_SHOE, _ShoeSegment.SHOE)
    right_key = SegmentKey(_RightShoeSubject.RIGHT_SHOE, _ShoeSegment.SHOE)

    left_pose = _trajectory_at_z(0.1)
    right_pose = _trajectory_at_z(0.9)

    state = SceneState(
        segment_poses={
            left_key: left_pose,
            right_key: right_pose,
        }
    )

    assert len(state.segment_poses) == 2
    assert state.pose_for_key(left_key) is left_pose
    assert state.pose_for_key(right_key) is right_pose
    assert state.pose(_LeftShoeSubject.LEFT_SHOE, _ShoeSegment.SHOE) is left_pose
    assert state.pose(_RightShoeSubject.RIGHT_SHOE, _ShoeSegment.SHOE) is right_pose

    scene_view = SceneView(spec=TEST_SCENE_SPEC, state=state)
    left_subject_view = scene_view.subject(_LeftShoeSubject.LEFT_SHOE)
    right_subject_view = SubjectView(
        subject_spec=TEST_SCENE_SPEC.right_shoe,
        state=state,
    )
    left_segment_view = left_subject_view.segment(SHOE_SEGMENT_SPEC)
    right_segment_view_from_scene = scene_view.segment(
        _RightShoeSubject.RIGHT_SHOE,
        SHOE_SEGMENT_SPEC,
    )
    right_segment_view = right_subject_view.segment(SHOE_SEGMENT_SPEC)
    assert left_segment_view.trajectory is left_pose
    assert right_segment_view_from_scene.trajectory is right_pose
    assert right_segment_view.trajectory is right_pose
    np.testing.assert_allclose(
        left_segment_view.pose_at(0).translation,
        np.array([0.0, 0.0, 0.1]),
    )
    np.testing.assert_allclose(
        right_segment_view.pose_at(0).translation,
        np.array([0.0, 0.0, 0.9]),
    )


def test_subject_view_segment_accepts_segment_id() -> None:
    pose = _trajectory_at_z(0.0)
    state = SceneState(
        segment_poses={
            SegmentKey(_LeftShoeSubject.LEFT_SHOE, _ShoeSegment.SHOE): pose,
        }
    )
    scene_view = SceneView(spec=TEST_SCENE_SPEC, state=state)
    segment_view = scene_view.subject(_LeftShoeSubject.LEFT_SHOE).segment(
        _ShoeSegment.SHOE
    )
    assert segment_view.spec is SHOE_SEGMENT_SPEC
    assert segment_view.subject_id == _LeftShoeSubject.LEFT_SHOE


def test_subject_view_segment_accepts_segment_spec() -> None:
    pose = _trajectory_at_z(0.0)
    state = SceneState(
        segment_poses={
            SegmentKey(_LeftShoeSubject.LEFT_SHOE, _ShoeSegment.SHOE): pose,
        }
    )
    scene_view = SceneView(spec=TEST_SCENE_SPEC, state=state)
    segment_view = scene_view.subject(_LeftShoeSubject.LEFT_SHOE).segment(
        SHOE_SEGMENT_SPEC
    )
    assert segment_view.spec is SHOE_SEGMENT_SPEC
    assert segment_view.subject_id == _LeftShoeSubject.LEFT_SHOE


def test_scene_view_segment_accepts_segment_id() -> None:
    pose = _trajectory_at_z(0.0)
    state = SceneState(
        segment_poses={
            SegmentKey(_RightShoeSubject.RIGHT_SHOE, _ShoeSegment.SHOE): pose,
        }
    )
    scene_view = SceneView(spec=TEST_SCENE_SPEC, state=state)
    segment_view = scene_view.segment(
        _RightShoeSubject.RIGHT_SHOE,
        _ShoeSegment.SHOE,
    )
    assert segment_view.spec is SHOE_SEGMENT_SPEC
    assert segment_view.subject_id == _RightShoeSubject.RIGHT_SHOE
