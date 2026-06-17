from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from retarget.core.keys import SegmentKey
from retarget.core.transform import RigidTransform


@dataclass(frozen=True, slots=True)
class SegmentPoseTrajectory:
    """Time-varying pose trajectory for one concrete segment instance."""

    poses: tuple[RigidTransform, ...]

    def __len__(self) -> int:
        """Number of timesteps in the trajectory."""
        return len(self.poses)

    @property
    def num_timesteps(self) -> int:
        """Number of timesteps in the trajectory."""
        return len(self.poses)

    def at(self, timestep: int) -> RigidTransform:
        """Return the pose at one timestep."""
        if not 0 <= timestep < self.num_timesteps:
            raise IndexError(
                f"Timestep {timestep} out of range [0, {self.num_timesteps})"
            )
        return self.poses[timestep]


@dataclass(frozen=True, slots=True)
class SceneState:
    """Runtime pose state for a scene, keyed by compiled string segment names."""

    segment_poses: Mapping[SegmentKey, SegmentPoseTrajectory] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_poses",
            MappingProxyType(dict(self.segment_poses)),
        )

    def pose(self, subject: str, segment: str) -> SegmentPoseTrajectory:
        return self.segment_poses[SegmentKey(subject, segment)]

    def pose_for_key(self, key: SegmentKey) -> SegmentPoseTrajectory:
        return self.segment_poses[key]

    @property
    def num_timesteps(self) -> int:
        """
        Number of timesteps in the scene state.
        All segment trajectories are required to have the same length.
        """
        if not self.segment_poses:
            return 0
        lengths = {
            trajectory.num_timesteps
            for trajectory in self.segment_poses.values()
        }
        if len(lengths) != 1:
            raise ValueError(
                f"SceneState contains inconsistent trajectory lengths: {lengths}"
            )
        return next(iter(lengths))
