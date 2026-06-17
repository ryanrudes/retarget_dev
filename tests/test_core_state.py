from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    RigidTransform,
    SceneState,
    SegmentKey,
    SegmentPoseTrajectory,
)


def test_scene_state_allows_same_segment_name_in_different_subjects() -> None:
    pose_a = RigidTransform.identity()
    pose_b = RigidTransform.from_rotation_translation(
        rotation=np.eye(3),
        translation=np.array([1.0, 2.0, 3.0]),
    )
    state = SceneState(
        segment_poses={
            SegmentKey("a", "foot"): SegmentPoseTrajectory(poses=(pose_a,)),
            SegmentKey("b", "foot"): SegmentPoseTrajectory(poses=(pose_b,)),
        }
    )
    np.testing.assert_allclose(
        state.pose("a", "foot").at(0).translation,
        np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        state.pose("b", "foot").at(0).translation,
        np.array([1.0, 2.0, 3.0]),
    )


def test_segment_pose_trajectory_allows_empty_trajectory() -> None:
    trajectory = SegmentPoseTrajectory(poses=())
    assert trajectory.num_timesteps == 0
    assert len(trajectory) == 0


def test_segment_pose_trajectory_checks_bounds() -> None:
    trajectory = SegmentPoseTrajectory(poses=(RigidTransform.identity(),))
    with pytest.raises(IndexError, match="out of range"):
        trajectory.at(1)


def test_scene_state_rejects_inconsistent_trajectory_lengths() -> None:
    state = SceneState(
        segment_poses={
            SegmentKey("a", "foot"): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),),
            ),
            SegmentKey("b", "foot"): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(), RigidTransform.identity()),
            ),
        }
    )
    with pytest.raises(ValueError, match="inconsistent trajectory lengths"):
        _ = state.num_timesteps
