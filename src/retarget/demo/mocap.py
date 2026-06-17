"""Mocap demonstration track over a typed scene schema.

A :class:`MocapTrack` holds runtime pose state, raw marker observations, and an
optional attached contact track, plus the authored ``Subjects`` schema. On
construction it binds the schema to that data, so the public query surface is
the schema itself::

    mocap.subjects["left_shoe"].segments["shoe"].translations()
    mocap.subjects["left_shoe"].segments["shoe"].markers["heel"].positions()
    mocap.subjects["left_shoe"].segments["shoe"].patches["sole"].points()

Resampling policy:

- segment translations: linear interpolation;
- segment rotations: discrete nearest/previous sampling;
- attached contacts: delegated to :meth:`ContactTrack.resample_to`;
- raw marker frames: intentionally dropped on resampled output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from retarget.core.keys import SegmentKey
from retarget.core.schema import (
    Segment,
    Subject,
    Subjects,
    _SegmentRuntime,
    bind_subjects_runtime,
)
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.targets import PatchTarget
from retarget.core.transform import RigidTransform
from retarget.demo.contact import ContactTrack
from retarget.demo.resampling import (
    ResampleMethod,
    resample_indices,
    validate_resample_timestamps,
)
from retarget.demo.tracks import Track, indices_for_time_range
from retarget.io import (
    UnbaggedDirectory,
    ViconMarkersFrame,
    iter_vicon_marker_frames,
    load_scene_state,
)


def _validate_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


@dataclass(frozen=True, slots=True)
class MocapTrack[SubjectsT: Subjects = Subjects](Track):
    """Time-indexed mocap track over a typed scene schema and runtime state.

    ``subjects`` is the authored schema; after construction it is replaced with
    a runtime-bound copy whose query methods answer time-series questions.
    """

    subjects: SubjectsT
    state: SceneState
    timestamps: np.ndarray
    marker_frames: tuple[ViconMarkersFrame, ...] | None = None
    contacts: ContactTrack | None = None
    nominal_hz_override: float | None = None
    rebase_time: InitVar[bool] = False

    def __post_init__(self, rebase_time: bool = False) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        contacts = self.contacts
        if rebase_time and len(timestamps) > 0:
            offset = timestamps[0]
            timestamps = timestamps - offset
            if contacts is not None:
                contacts = ContactTrack(
                    contacts=contacts.contacts,
                    timestamps=contacts.timestamps - offset,
                    confidences=contacts.confidences,
                    nominal_hz_override=contacts.nominal_hz_override,
                )
                object.__setattr__(self, "contacts", contacts)
        _validate_timestamps(timestamps)
        object.__setattr__(self, "timestamps", timestamps)

        if len(timestamps) != self.state.num_timesteps:
            raise ValueError(
                f"len(timestamps)={len(timestamps)} does not match "
                f"state.num_timesteps={self.state.num_timesteps}"
            )
        if self.marker_frames is not None and len(self.marker_frames) != len(timestamps):
            raise ValueError(
                f"len(marker_frames)={len(self.marker_frames)} does not match "
                f"len(timestamps)={len(timestamps)}"
            )
        if self.contacts is not None:
            if len(self.contacts.timestamps) != len(timestamps):
                raise ValueError(
                    "ContactTrack timestamp count must match MocapTrack timestamp count"
                )
            if len(timestamps) and not np.allclose(self.contacts.timestamps, timestamps):
                raise ValueError(
                    "ContactTrack timestamps must match MocapTrack timestamps"
                )

        runtimes = _build_segment_runtimes(
            self.subjects,
            self.state,
            timestamps,
            self.marker_frames,
            self.contacts,
        )
        object.__setattr__(
            self, "subjects", bind_subjects_runtime(self.subjects, runtimes)
        )

    def __len__(self) -> int:
        return len(self.timestamps)

    @classmethod
    def from_unbagged[S: Subjects](
        cls,
        root: Path | UnbaggedDirectory,
        subjects: S,
        *,
        tf_prefix: str = "vicon",
        rebase_time: bool = False,
    ) -> MocapTrack[S]:
        """Load a mocap track from an unbagged export directory and authored scene.

        Uses a method-local type parameter (not the class ``SubjectsT``) so the
        returned track type is inferred from ``subjects`` at the call site.
        """
        export = root if isinstance(root, UnbaggedDirectory) else UnbaggedDirectory(root)
        state = load_scene_state(export, subjects, tf_prefix=tf_prefix)
        marker_frames = tuple(iter_vicon_marker_frames(export))
        timestamps = np.asarray(
            [frame.stamp_seconds for frame in marker_frames],
            dtype=np.float64,
        )
        if state.num_timesteps != len(marker_frames):
            raise ValueError(
                "SceneState timestep count does not match marker frame count: "
                f"{state.num_timesteps} != {len(marker_frames)}"
            )
        return MocapTrack(
            subjects=subjects,
            state=state,
            timestamps=timestamps,
            marker_frames=marker_frames,
            rebase_time=rebase_time,
        )

    # -- segment-keyed array access (low level) --------------------------------

    def segment_translations(self, key: SegmentKey) -> np.ndarray:
        """Full-track segment translations with shape ``(T, 3)``."""
        return _trajectory_arrays(self.state.pose_for_key(key))[0]

    def segment_rotations(self, key: SegmentKey) -> np.ndarray:
        """Full-track segment rotation matrices with shape ``(T, 3, 3)``."""
        return _trajectory_arrays(self.state.pose_for_key(key))[1]

    # -- timeline operations ---------------------------------------------------

    def slice_time(self, start: float, stop: float) -> MocapTrack[SubjectsT]:
        return self._select(indices_for_time_range(self.timestamps, start, stop))

    def _select(self, indices: tuple[int, ...]) -> MocapTrack[SubjectsT]:
        idx = list(indices)
        sliced_state = SceneState(
            segment_poses={
                key: SegmentPoseTrajectory(tuple(traj.poses[i] for i in idx))
                for key, traj in self.state.segment_poses.items()
            }
        )
        sliced_timestamps = self.timestamps[idx]
        sliced_frames = (
            None
            if self.marker_frames is None
            else tuple(self.marker_frames[i] for i in idx)
        )
        sliced_contacts = (
            None if self.contacts is None else self.contacts.select_indices(indices)
        )
        return MocapTrack(
            subjects=self.subjects,
            state=sliced_state,
            timestamps=sliced_timestamps,
            marker_frames=sliced_frames,
            contacts=sliced_contacts,
            nominal_hz_override=self.nominal_hz_override,
        )

    def resample_to(
        self,
        timestamps: np.ndarray,
        *,
        output_timestamps: np.ndarray | None = None,
        method: ResampleMethod | str = ResampleMethod.NEAREST,
    ) -> MocapTrack[SubjectsT]:
        """Return this track sampled at requested source-time timestamps.

        Translations are linearly interpolated; ``method`` is the discrete
        sampling policy applied to the step-like quantities (segment rotations
        and any attached contact state).
        """
        sample = validate_resample_timestamps(timestamps)
        if len(sample) == 0:
            raise ValueError("cannot resample mocap track to empty timestamps")
        if output_timestamps is None:
            result_timestamps = sample
        else:
            result_timestamps = validate_resample_timestamps(output_timestamps)
            if len(result_timestamps) != len(sample):
                raise ValueError(
                    "output_timestamps must have the same length as timestamps"
                )

        source_timestamps = self.timestamps
        if len(source_timestamps) == 0:
            raise ValueError("cannot resample empty mocap source timestamps")

        rotation_indices = resample_indices(
            source_timestamps=source_timestamps,
            target_timestamps=sample,
            method=method,
        )
        new_state = SceneState(
            segment_poses={
                key: _resample_trajectory(
                    traj, source_timestamps, sample, rotation_indices
                )
                for key, traj in self.state.segment_poses.items()
            }
        )
        new_contacts = (
            None
            if self.contacts is None
            else self.contacts.resample_to(
                sample,
                output_timestamps=result_timestamps,
                method=method,
            )
        )
        return MocapTrack(
            subjects=self.subjects,
            state=new_state,
            timestamps=result_timestamps,
            marker_frames=None,
            contacts=new_contacts,
            nominal_hz_override=self.nominal_hz_override,
        )


def _trajectory_arrays(
    trajectory: SegmentPoseTrajectory,
) -> tuple[np.ndarray, np.ndarray]:
    if not trajectory.poses:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3, 3), dtype=np.float64),
        )
    translations = np.stack([pose.translation for pose in trajectory.poses]).astype(
        np.float64, copy=False
    )
    rotations = np.stack([pose.rotation for pose in trajectory.poses]).astype(
        np.float64, copy=False
    )
    return translations, rotations


def _resample_trajectory(
    trajectory: SegmentPoseTrajectory,
    source_timestamps: np.ndarray,
    sample_timestamps: np.ndarray,
    rotation_indices: np.ndarray,
) -> SegmentPoseTrajectory:
    translations, rotations = _trajectory_arrays(trajectory)
    resampled_translations = np.column_stack(
        [
            np.interp(sample_timestamps, source_timestamps, translations[:, axis])
            for axis in range(3)
        ]
    )
    return SegmentPoseTrajectory(
        tuple(
            RigidTransform.from_rotation_translation(
                rotation=rotations[int(rotation_index)],
                translation=resampled_translations[timestep],
            )
            for timestep, rotation_index in enumerate(rotation_indices)
        )
    )


def _build_segment_runtimes(
    subjects: Subjects,
    state: SceneState,
    timestamps: np.ndarray,
    marker_frames: tuple[ViconMarkersFrame, ...] | None,
    contacts: ContactTrack | None,
) -> dict[SegmentKey, _SegmentRuntime]:
    runtimes: dict[SegmentKey, _SegmentRuntime] = {}
    num_timesteps = len(timestamps)
    for subject_name, subject in cast(Mapping[str, Subject[Any]], subjects).items():
        subject_external = subject.mocap_name or subject_name
        segments = cast(Mapping[str, Segment[Any, Any]], subject.segments)
        for segment_name, segment in segments.items():
            key = SegmentKey(subject_name, segment_name)
            trajectory = state.segment_poses.get(key)
            if trajectory is None:
                continue
            translations, rotations = _trajectory_arrays(trajectory)
            observed = _observed_markers(
                segment,
                marker_frames,
                subject_external=subject_external,
                segment_external=segment.mocap_name or segment_name,
                num_timesteps=num_timesteps,
            )
            contact_state, contact_confidence = _segment_contacts(
                contacts, subject_name, segment_name, segment
            )
            runtimes[key] = _SegmentRuntime(
                timestamps=timestamps,
                translations=translations,
                rotations=rotations,
                observed_markers=observed,
                contacts=contact_state,
                confidences=contact_confidence,
            )
    return runtimes


def _observed_markers(
    segment: Segment[Any, Any],
    marker_frames: tuple[ViconMarkersFrame, ...] | None,
    *,
    subject_external: str,
    segment_external: str,
    num_timesteps: int,
) -> Mapping[str, np.ndarray]:
    if marker_frames is None:
        return {}
    observed = {
        marker.mocap_name: np.full((num_timesteps, 3), np.nan, dtype=np.float64)
        for marker in segment.markers.values()
    }
    for timestep, frame in enumerate(marker_frames):
        for obs in frame.markers:
            if obs.occluded:
                continue
            if obs.subject_name != subject_external:
                continue
            if obs.segment_name != segment_external:
                continue
            array = observed.get(obs.marker_name)
            if array is not None:
                array[timestep] = obs.position_world
    return observed


def _segment_contacts(
    contacts: ContactTrack | None,
    subject_name: str,
    segment_name: str,
    segment: Segment[Any, Any],
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    if contacts is None:
        return {}, {}
    state_map: dict[str, np.ndarray] = {}
    confidence_map: dict[str, np.ndarray] = {}
    for patch_name in segment.patches:
        target = PatchTarget(
            subject=subject_name, segment=segment_name, patch=patch_name
        )
        if target in contacts.contacts:
            state_map[patch_name] = np.asarray(contacts.contacts[target])
        if target in contacts.confidences:
            confidence_map[patch_name] = np.asarray(contacts.confidences[target])
    return state_map, confidence_map
