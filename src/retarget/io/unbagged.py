from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import orjson
from scipy.spatial.transform import Rotation

from retarget.core import MarkerId, SegmentSpec, SegmentView, SubjectId, Vec3
from retarget.core.enums import SegmentId
from retarget.core.keys import SegmentKey
from retarget.core.specs import SceneSpec
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.transform import RigidTransform


TF_FILENAME = "tf.json"
VICON_MARKERS_FILENAME = "vicon_markers.json"


@dataclass(frozen=True, slots=True)
class UnbaggedDirectory:
    """Directory produced by ros2 unbag with single-file JSON exports."""

    path: Path

    def __post_init__(self) -> None:
        if not self.path.is_dir():
            raise FileNotFoundError(f"Unbagged directory not found: {self.path}")

    @property
    def tf_path(self) -> Path:
        return self.path / TF_FILENAME

    @property
    def vicon_markers_path(self) -> Path:
        return self.path / VICON_MARKERS_FILENAME

    def load_tf_messages(self) -> dict[str, Mapping[str, Any]]:
        """Load TF messages keyed by export timestamp string."""
        return _load_json_object(self.tf_path)

    def load_vicon_marker_messages(self) -> dict[str, Mapping[str, Any]]:
        """Load Vicon marker messages keyed by export timestamp string."""
        return _load_json_object(self.vicon_markers_path)


@dataclass(frozen=True, slots=True)
class MarkerObservation:
    """One marker reading in the world frame."""

    marker_name: str
    subject_name: str
    segment_name: str
    position_world: Vec3
    occluded: bool


@dataclass(frozen=True, slots=True)
class ViconMarkersFrame:
    """All marker observations at one synchronized timestep."""

    stamp_seconds: float
    markers: tuple[MarkerObservation, ...]


def stamp_to_seconds(stamp: Mapping[str, int]) -> float:
    """Convert a ROS header stamp dict to seconds."""
    return float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9


def rigid_transform_from_ros_json(transform: Mapping[str, Any]) -> RigidTransform:
    """Build a RigidTransform from a ROS geometry_msgs/Transform JSON object."""
    translation = transform["translation"]
    rotation = transform["rotation"]
    return RigidTransform.from_rotation_translation(
        rotation=Rotation.from_quat(
            [
                rotation["x"],
                rotation["y"],
                rotation["z"],
                rotation["w"],
            ]
        ).as_matrix(),
        translation=np.array(
            [
                translation["x"],
                translation["y"],
                translation["z"],
            ]
        ),
    )


def parse_tf_child_frame(
    child_frame_id: str,
    *,
    tf_prefix: str = "vicon",
) -> tuple[str, str]:
    """
    Parse a Vicon TF child frame into subject and segment names.

    Expected format: ``{tf_prefix}/{subject_name}/{segment_name}``.
    """
    parts = child_frame_id.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"Expected TF child frame '{child_frame_id}' to have three "
            f"'/'-separated components."
        )
    prefix, subject_name, segment_name = parts
    if prefix != tf_prefix:
        raise ValueError(
            f"Expected TF child frame prefix '{tf_prefix}', got '{prefix}'."
        )
    return subject_name, segment_name


def tf_child_frame_id(
    subject: SubjectId,
    segment: SegmentId,
    *,
    tf_prefix: str = "vicon",
) -> str:
    """Build the TF child frame id for one subject/segment pair."""
    return f"{tf_prefix}/{subject.label}/{segment.label}"


def load_segment_pose_trajectories(
    export: UnbaggedDirectory,
    scene: SceneSpec,
    *,
    tf_prefix: str = "vicon",
) -> dict[SegmentKey, SegmentPoseTrajectory]:
    """Load world-from-segment pose trajectories for all scene segments."""
    messages = export.load_tf_messages()
    expected_keys = {
        SegmentKey(subject.subject, segment.segment)
        for subject in scene.iter_subjects()
        for segment in subject.iter_segments()
    }
    poses_by_key: dict[SegmentKey, list[RigidTransform]] = {
        key: [] for key in expected_keys
    }

    for timestamp_key in sorted(messages):
        message = messages[timestamp_key]
        subject_name, segment_name = parse_tf_child_frame(
            message["child_frame_id"],
            tf_prefix=tf_prefix,
        )
        pose = rigid_transform_from_ros_json(message["transform"])

        for key in expected_keys:
            if (
                key.subject.label == subject_name
                and key.segment.label == segment_name
            ):
                poses_by_key[key].append(pose)
                break
        else:
            raise KeyError(
                f"TF frame '{message['child_frame_id']}' at {timestamp_key} "
                f"does not match any segment in the scene spec."
            )

    trajectories: dict[SegmentKey, SegmentPoseTrajectory] = {}
    for key, poses in poses_by_key.items():
        if not poses:
            raise ValueError(
                f"No TF poses found for segment "
                f"{key.subject.label}/{key.segment.label}."
            )
        trajectories[key] = SegmentPoseTrajectory(poses=tuple(poses))

    return trajectories


def load_scene_state(
    export: UnbaggedDirectory,
    scene: SceneSpec,
    *,
    tf_prefix: str = "vicon",
) -> SceneState:
    """Load a SceneState from an unbagged export directory."""
    return SceneState(
        segment_poses=load_segment_pose_trajectories(
            export,
            scene,
            tf_prefix=tf_prefix,
        )
    )


def iter_vicon_marker_frames(
    export: UnbaggedDirectory,
) -> Iterable[ViconMarkersFrame]:
    """Iterate marker observations in export timestamp order."""
    messages = export.load_vicon_marker_messages()
    for timestamp_key in sorted(messages):
        message = messages[timestamp_key]
        stamp_seconds = stamp_to_seconds(message["header"]["stamp"])
        markers = tuple(
            MarkerObservation(
                marker_name=marker["marker_name"],
                subject_name=marker["subject_name"],
                segment_name=marker["segment_name"],
                position_world=np.array(
                    [
                        marker["translation"]["x"],
                        marker["translation"]["y"],
                        marker["translation"]["z"],
                    ]
                ),
                occluded=bool(marker["occluded"]),
            )
            for marker in message["markers"]
        )
        yield ViconMarkersFrame(
            stamp_seconds=stamp_seconds,
            markers=markers,
        )


def marker_positions_by_name(
    frame: ViconMarkersFrame,
    *,
    subject_name: str | None = None,
    segment_name: str | None = None,
    include_occluded: bool = False,
) -> dict[str, Vec3]:
    """Index one marker frame by marker name."""
    positions: dict[str, Vec3] = {}
    for marker in frame.markers:
        if subject_name is not None and marker.subject_name != subject_name:
            continue
        if segment_name is not None and marker.segment_name != segment_name:
            continue
        if marker.occluded and not include_occluded:
            continue
        positions[marker.marker_name] = marker.position_world
    return positions


@overload
def marker_position[M: MarkerId](
    marker_frame: ViconMarkersFrame,
    *,
    segment: SegmentView[M, Any],
    marker: M,
) -> Vec3 | None: ...


@overload
def marker_position[M: MarkerId](
    marker_frame: ViconMarkersFrame,
    *,
    subject: SubjectId,
    segment: SegmentSpec[M, Any],
    marker: M,
) -> Vec3 | None: ...


def marker_position(
    marker_frame: ViconMarkersFrame,
    *,
    marker: MarkerId,
    segment: SegmentView[Any, Any] | SegmentSpec[Any, Any],
    subject: SubjectId | None = None,
) -> Vec3 | None:
    """
    Return one observed marker position from a Vicon marker frame.

    Preferred usage after resolving a view::

        marker_position(marker_frame, segment=segment_view, marker=marker)

    Alternative usage without a view::

        marker_position(
            marker_frame,
            subject=subject_id,
            segment=segment_spec,
            marker=marker,
        )
    """
    if isinstance(segment, SegmentView):
        subject_name = segment.subject_id.label
        segment_name = segment.spec.segment.label
    else:
        if subject is None:
            raise TypeError(
                "subject must be provided when segment is a SegmentSpec"
            )
        subject_name = subject.label
        segment_name = segment.segment.label
    positions = marker_positions_by_name(
        marker_frame,
        subject_name=subject_name,
        segment_name=segment_name,
    )
    return positions.get(marker.label)


def _load_json_object(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected unbagged JSON file: {path}")
    return orjson.loads(path.read_bytes())
