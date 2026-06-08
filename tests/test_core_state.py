import numpy as np
import pytest

from retarget.core import (
    SubjectId,
    SegmentId,
    SegmentKey,
    RigidTransform,
    SegmentPoseTrajectory,
    SceneState,
)


class TestSubjectId(SubjectId):
    A = "a"
    B = "b"


class TestSegmentId(SegmentId):
    FOOT = "foot"


def test_scene_state_allows_same_segment_id_in_different_subjects() -> None:
    pose_a = RigidTransform.identity()
    pose_b = RigidTransform.from_rotation_translation(
        rotation=np.eye(3),
        translation=np.array([1.0, 2.0, 3.0]),
    )
    state = SceneState(
        segment_poses={
            SegmentKey(TestSubjectId.A, TestSegmentId.FOOT): SegmentPoseTrajectory(
                poses=(pose_a,),
            ),
            SegmentKey(TestSubjectId.B, TestSegmentId.FOOT): SegmentPoseTrajectory(
                poses=(pose_b,),
            ),
        }
    )
    resolved_a = state.pose(TestSubjectId.A, TestSegmentId.FOOT).at(0)
    resolved_b = state.pose(TestSubjectId.B, TestSegmentId.FOOT).at(0)
    np.testing.assert_allclose(
        resolved_a.translation,
        np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        resolved_b.translation,
        np.array([1.0, 2.0, 3.0]),
    )


def test_segment_pose_trajectory_rejects_empty_trajectory() -> None:
    with pytest.raises(ValueError, match="requires at least one pose"):
        SegmentPoseTrajectory(poses=())


def test_segment_pose_trajectory_checks_bounds() -> None:
    trajectory = SegmentPoseTrajectory(
        poses=(RigidTransform.identity(),),
    )
    with pytest.raises(IndexError, match="out of range"):
        trajectory.at(1)


def test_scene_state_rejects_inconsistent_trajectory_lengths() -> None:
    state = SceneState(
        segment_poses={
            SegmentKey(TestSubjectId.A, TestSegmentId.FOOT): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),),
            ),
            SegmentKey(TestSubjectId.B, TestSegmentId.FOOT): SegmentPoseTrajectory(
                poses=(
                    RigidTransform.identity(),
                    RigidTransform.identity(),
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="inconsistent trajectory lengths"):
        _ = state.num_timesteps
