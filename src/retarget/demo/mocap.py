"""Mocap demonstration track and time-series query views.

Mocap resampling is split by data type:

- continuous segment translations can be linearly interpolated;
- rotations should not be linearly interpolated as raw matrices unless they are
  projected/re-normalized afterward;
- observed marker positions can be linearly interpolated only across visible
  samples, with missing/occluded values preserved as NaN;
- marker frame objects are raw observations and are not resampled directly;
- attached contact tracks use discrete contact resampling semantics.

The initial implementation should prefer conservative behavior over pretending
all mocap data is the same kind of array. Humanity tried that with spreadsheets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast, overload

import numpy as np

from retarget.core.enums import MarkerId, PatchId, PoseFormat, RotationFormat, SegmentId, SubjectId
from retarget.core.keys import SegmentKey
from retarget.core.handles import PatchHandle
from retarget.core.specs import SceneSpec, SegmentSpec, SubjectSpec
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.targets import PatchTarget
from retarget.core.transform import RigidTransform
from retarget.core.types import TimeEntityVec3, TimeMat3, TimeQuat, TimeVec3
from retarget.core.views import SceneView, SegmentView
from retarget.core import segment_external_name, subject_external_name
from retarget.demo._mocap_arrays import (
    MocapArrayCache,
    pose_arrays_to_format,
    rotation_matrices_to_format,
)
from retarget.demo._query_utils import (
    finite_difference_velocity,
    resolve_indices,
    slice_timestamps,
    speed_from_velocity,
    stack_entity_arrays,
)
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.resampling import (
    ResampleMethod,
    resample_indices,
    validate_resample_timestamps,
)
from retarget.demo.tracks import Track, TrackView
from retarget.io import ViconMarkersFrame


def _validate_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def _normalize_stringable_entity_input[E](
    entity: E | str | Sequence[E | str],
    entity_type: type[E],
) -> tuple[tuple[E | str, ...], bool]:
    if isinstance(entity, entity_type):
        return (entity,), False
    if isinstance(entity, str):
        return (entity,), False
    entities = tuple(entity)
    for item in entities:
        if not isinstance(item, entity_type) and not isinstance(item, str):
            raise TypeError(
                f"Expected {entity_type.__name__} or str; got {type(item).__name__}"
            )
    return entities, True


@dataclass(frozen=True, slots=True)
class MocapTrack(Track):
    """Time-indexed mocap track over a scene spec and runtime state."""

    scene_spec: SceneSpec
    state: SceneState
    timestamps: np.ndarray
    marker_frames: tuple[ViconMarkersFrame, ...] | None = None
    contacts: ContactTrack | None = None
    nominal_hz_override: float | None = None
    _array_cache: MocapArrayCache = field(
        default_factory=MocapArrayCache,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        object.__setattr__(self, "timestamps", timestamps)
        _validate_timestamps(timestamps)
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
            if not np.allclose(self.contacts.timestamps, timestamps):
                raise ValueError(
                    "ContactTrack timestamps must match MocapTrack timestamps"
                )

    def __len__(self) -> int:
        return len(self.timestamps)

    def with_timestamps(self, timestamps: np.ndarray) -> MocapTrack:
        if self.contacts is not None and not np.allclose(
            self.contacts.timestamps, timestamps
        ):
            raise ValueError(
                "Cannot change MocapTrack timestamps while contacts are attached; "
                "rebase or resample contacts explicitly first."
            )
        return MocapTrack(
            scene_spec=self.scene_spec,
            state=self.state,
            timestamps=timestamps,
            marker_frames=self.marker_frames,
            contacts=self.contacts,
            nominal_hz_override=self.nominal_hz_override,
        )

    def with_rebased_time(self) -> MocapTrack:
        if len(self.timestamps) == 0:
            return self
        return self.with_timestamps(self.timestamps - self.timestamps[0])

    def resample_to(
        self,
        timestamps: np.ndarray,
        *,
        output_timestamps: np.ndarray | None = None,
        rotation_method: ResampleMethod | str = ResampleMethod.NEAREST,
        contact_method: ResampleMethod | str = ResampleMethod.NEAREST,
    ) -> "MocapTrack":
        """Return this mocap track sampled at requested source-time timestamps.

        Segment translations are linearly interpolated. Segment rotations are
        sampled discretely using ``rotation_method``. Raw marker frames are not
        resampled and are intentionally dropped on the returned track.
        """
        return _resample_mocap_track(
            self,
            timestamps=timestamps,
            output_timestamps=output_timestamps,
            rotation_method=rotation_method,
            contact_method=contact_method,
        )

    @classmethod
    def _view_type(cls) -> type[MocapTrackView]:
        return MocapTrackView

    @property
    def scene(self) -> SceneView:
        return SceneView(spec=self.scene_spec, state=self.state)

    @property
    def subjects(self) -> Mapping[str, MocapSubjectTrackView]:
        return _mocap_subject_views(self)

    def subject(self, subject: SubjectId | str) -> MocapSubjectTrackView:
        subject_view = self.scene.subject(subject)
        return MocapSubjectTrackView(
            mocap=self,
            subject_id=subject_view.subject_id,
        )

    def segment(
        self,
        subject: SubjectId | str,
        segment: SegmentId | str | SegmentSpec[Any, Any],
    ) -> MocapSegmentTrackView[Any, Any]:
        return self.subject(subject).segment(segment)

    def segment_translations(self, key: SegmentKey) -> np.ndarray:
        """Return full-track segment translations with shape ``(T, 3)``."""
        cached = self._array_cache.translations.get(key)
        if cached is not None:
            return cached
        trajectory = self.state.pose_for_key(key)
        if len(trajectory.poses) == 0:
            arr = np.empty((0, 3), dtype=np.float64)
        else:
            arr = np.stack(
                [pose.translation for pose in trajectory.poses],
                axis=0,
            ).astype(np.float64, copy=False)
        self._array_cache.translations[key] = arr
        return arr

    def segment_rotations(self, key: SegmentKey) -> np.ndarray:
        """Return full-track segment rotation matrices with shape ``(T, 3, 3)``."""
        cached = self._array_cache.rotations.get(key)
        if cached is not None:
            return cached
        trajectory = self.state.pose_for_key(key)
        if len(trajectory.poses) == 0:
            arr = np.empty((0, 3, 3), dtype=np.float64)
        else:
            arr = np.stack(
                [pose.rotation for pose in trajectory.poses],
                axis=0,
            ).astype(np.float64, copy=False)
        self._array_cache.rotations[key] = arr
        return arr

    def observed_marker_positions_for_segment[M: MarkerId](
        self,
        subject: SubjectId | str,
        segment: SegmentSpec[M, Any],
    ) -> np.ndarray:
        """Return full-track observed marker positions with shape ``(T, M, 3)``."""
        subject_view = self.scene.subject(subject)
        key = SegmentKey(subject_view.subject_id, segment.segment)
        cached = self._array_cache.observed_markers.get(key)
        if cached is not None:
            return cached
        if self.marker_frames is None:
            raise ValueError(
                "Observed marker positions require marker_frames on the mocap track"
            )
        arr = np.full(
            (len(self.timestamps), segment.marker_type.size(), 3),
            np.nan,
            dtype=np.float64,
        )
        subject_name = subject_external_name(subject_view.subject_spec)
        segment_name = segment_external_name(segment)
        for timestep, frame in enumerate(self.marker_frames):
            for obs in frame.markers:
                if obs.subject_name != subject_name:
                    continue
                if obs.segment_name != segment_name:
                    continue
                if obs.occluded:
                    continue
                try:
                    marker = segment.marker_from_external_name(obs.marker_name)
                except ValueError:
                    continue
                arr[timestep, marker.index, :] = obs.position_world
        self._array_cache.observed_markers[key] = arr
        return arr


@dataclass(frozen=True, slots=True)
class MocapTrackView(TrackView[MocapTrack]):
    """Sliced view into a :class:`MocapTrack`."""

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def timestamps(self) -> np.ndarray:
        return slice_timestamps(self.source.timestamps, self.indices)

    @property
    def scene(self) -> SceneView:
        return self.source.scene

    @property
    def subjects(self) -> Mapping[str, MocapSubjectTrackView]:
        return _mocap_subject_views(self)

    def subject(self, subject: SubjectId | str) -> MocapSubjectTrackView:
        subject_view = self.scene.subject(subject)
        return MocapSubjectTrackView(
            mocap=self,
            subject_id=subject_view.subject_id,
        )

    def segment(
        self,
        subject: SubjectId | str,
        segment: SegmentId | str | SegmentSpec[Any, Any],
    ) -> MocapSegmentTrackView[Any, Any]:
        return self.subject(subject).segment(segment)

    def resample_to(
        self,
        timestamps: np.ndarray,
        *,
        output_timestamps: np.ndarray | None = None,
        rotation_method: ResampleMethod | str = ResampleMethod.NEAREST,
        contact_method: ResampleMethod | str = ResampleMethod.NEAREST,
    ) -> MocapTrack:
        """Return this mocap view sampled at requested view-time timestamps."""
        return _resample_mocap_track(
            self,
            timestamps=timestamps,
            output_timestamps=output_timestamps,
            rotation_method=rotation_method,
            contact_method=contact_method,
        )


def _resample_mocap_track(
    mocap: MocapTrack | MocapTrackView,
    *,
    timestamps: np.ndarray,
    output_timestamps: np.ndarray | None,
    rotation_method: ResampleMethod | str,
    contact_method: ResampleMethod | str,
) -> MocapTrack:
    sample_timestamps = validate_resample_timestamps(timestamps)
    if len(sample_timestamps) == 0:
        raise ValueError("cannot resample mocap track to empty timestamps")

    if output_timestamps is None:
        result_timestamps = sample_timestamps
    else:
        result_timestamps = validate_resample_timestamps(output_timestamps)
        if len(result_timestamps) != len(sample_timestamps):
            raise ValueError(
                "output_timestamps must have the same length as timestamps"
            )

    source = _mocap_source(mocap)
    source_timestamps = _mocap_timestamps(mocap)
    visible_indices = _mocap_indices(mocap)
    state = _resample_scene_state(
        state=source.state,
        source_timestamps=source_timestamps,
        visible_indices=visible_indices,
        sample_timestamps=sample_timestamps,
        rotation_method=rotation_method,
    )
    contacts = _resample_mocap_contacts(
        mocap,
        sample_timestamps=sample_timestamps,
        output_timestamps=result_timestamps,
        contact_method=contact_method,
    )
    return MocapTrack(
        scene_spec=source.scene_spec,
        state=state,
        timestamps=result_timestamps,
        marker_frames=None,
        contacts=contacts,
        nominal_hz_override=source.nominal_hz_override,
    )


def _resample_scene_state(
    *,
    state: SceneState,
    source_timestamps: np.ndarray,
    visible_indices: tuple[int, ...],
    sample_timestamps: np.ndarray,
    rotation_method: ResampleMethod | str,
) -> SceneState:
    if len(source_timestamps) == 0:
        raise ValueError("cannot resample empty mocap source timestamps")
    visible_index_array = np.array(visible_indices, dtype=np.intp)
    return SceneState(
        segment_poses={
            key: _resample_pose_trajectory(
                trajectory=trajectory,
                source_timestamps=source_timestamps,
                visible_indices=visible_index_array,
                sample_timestamps=sample_timestamps,
                rotation_method=rotation_method,
            )
            for key, trajectory in state.segment_poses.items()
        }
    )


def _resample_pose_trajectory(
    *,
    trajectory: SegmentPoseTrajectory,
    source_timestamps: np.ndarray,
    visible_indices: np.ndarray,
    sample_timestamps: np.ndarray,
    rotation_method: ResampleMethod | str,
) -> SegmentPoseTrajectory:
    visible_poses = tuple(trajectory.poses[int(index)] for index in visible_indices)
    translations = np.stack(
        [pose.translation for pose in visible_poses],
        axis=0,
    ).astype(np.float64, copy=False)
    rotations = tuple(pose.rotation for pose in visible_poses)
    resampled_translations = np.column_stack(
        [
            np.interp(sample_timestamps, source_timestamps, translations[:, axis])
            for axis in range(3)
        ]
    )
    rotation_indices = resample_indices(
        source_timestamps=source_timestamps,
        target_timestamps=sample_timestamps,
        method=rotation_method,
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


def _resample_mocap_contacts(
    mocap: MocapTrack | MocapTrackView,
    *,
    sample_timestamps: np.ndarray,
    output_timestamps: np.ndarray,
    contact_method: ResampleMethod | str,
) -> ContactTrack | None:
    contact_track = _contact_track(mocap)
    if contact_track is None:
        return None
    sliced = _slice_contact_track(contact_track, mocap)
    return sliced.resample_to(
        sample_timestamps,
        output_timestamps=output_timestamps,
        method=contact_method,
    )


@dataclass(frozen=True, slots=True)
class MocapSubjectTrackView:
    """Subject-scoped entry point into a mocap track or sliced view."""

    mocap: MocapTrack | MocapTrackView
    subject_id: SubjectId

    @property
    def subject_spec(self) -> SubjectSpec:
        return self.mocap.scene.subject(self.subject_id).subject_spec

    @property
    def segments(self) -> Mapping[str, MocapSegmentTrackView[Any, Any]]:
        return _mocap_segment_views(self)

    @overload
    def segment[M: MarkerId, P: PatchId](
        self,
        segment: SegmentSpec[M, P] | SegmentId | str,
    ) -> MocapSegmentTrackView[M, P]: ...

    @overload
    def segment(
        self,
        segment: SegmentId | str,
    ) -> MocapSegmentTrackView[Any, Any]: ...

    def segment(
        self,
        segment: SegmentId | str | SegmentSpec[Any, Any],
    ) -> MocapSegmentTrackView[Any, Any]:
        segment_view = self.mocap.scene.subject(self.subject_id).segment(segment)
        indices = _mocap_indices(self.mocap)
        return MocapSegmentTrackView(
            mocap=self.mocap,
            segment_view=segment_view,
            indices=indices,
        )


@dataclass(frozen=True, slots=True)
class MocapSegmentTrackView[M: MarkerId, P: PatchId]:
    """Time-series query surface for one segment within a mocap track."""

    mocap: MocapTrack | MocapTrackView
    segment_view: SegmentView[M, P]
    indices: tuple[int, ...]

    @property
    def timestamps(self) -> np.ndarray:
        return _mocap_timestamps(self.mocap)

    @property
    def markers(self) -> Mapping[str, _MocapMarkerAccessor[M, P]]:
        return _mocap_marker_accessors(self)

    @property
    def patches(self) -> Mapping[str, _MocapPatchAccessor[M, P]]:
        return _mocap_patch_accessors(self)

    def poses(
        self,
        *,
        format: PoseFormat = PoseFormat.RIGID_TRANSFORM,
    ) -> tuple[RigidTransform, ...] | np.ndarray:
        return pose_arrays_to_format(
            self.translations(),
            self._rotation_matrices(),
            format=format,
        )

    def translations(self) -> TimeVec3:
        if not self.indices:
            return np.empty((0, 3), dtype=np.float64)
        source = self._source_track()
        full = source.segment_translations(self._segment_key())
        return full[self._index_array()]

    def rotations(
        self,
        *,
        format: RotationFormat = RotationFormat.MATRIX,
    ) -> TimeMat3 | TimeQuat | np.ndarray:
        return rotation_matrices_to_format(
            self._rotation_matrices(),
            format=format,
        )

    def linear_velocities(self) -> TimeVec3:
        return finite_difference_velocity(self.translations(), self.timestamps)

    def speed(self) -> np.ndarray:
        return speed_from_velocity(self.linear_velocities())

    @overload
    def marker_positions(
        self,
        marker: M | str,
        *,
        modeled: bool = False,
    ) -> TimeVec3: ...

    @overload
    def marker_positions(
        self,
        marker: Sequence[M | str],
        *,
        modeled: bool = False,
        return_dict: Literal[False] = False,
    ) -> TimeEntityVec3: ...

    @overload
    def marker_positions(
        self,
        marker: Sequence[M | str],
        *,
        modeled: bool = False,
        return_dict: Literal[True],
    ) -> Mapping[M, TimeVec3]: ...

    def marker_positions(
        self,
        marker: M | str | Sequence[M | str],
        *,
        modeled: bool = False,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[M, TimeVec3]:
        markers, is_many = _normalize_stringable_entity_input(
            marker,
            self.segment_view.spec.marker_type,
        )
        coerced = tuple(self._coerce_marker(m) for m in markers)
        if modeled:
            world = self._modeled_marker_positions_many(coerced)
        else:
            world = self._observed_marker_positions_many(coerced)
        if not is_many and not return_dict:
            return world[:, 0, :]
        if return_dict:
            return dict(zip(coerced, (world[:, i, :] for i in range(len(coerced))), strict=True))
        return world

    def marker_velocities(
        self,
        marker: M | str | Sequence[M | str],
        *,
        modeled: bool = False,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[M, TimeVec3]:
        markers, is_many = _normalize_stringable_entity_input(
            marker,
            self.segment_view.spec.marker_type,
        )
        coerced = tuple(self._coerce_marker(m) for m in markers)
        positions = self.marker_positions(marker, modeled=modeled, return_dict=False)
        velocities = self._differentiate(positions)
        if not is_many and not return_dict:
            return velocities
        if return_dict:
            if velocities.ndim == 2:
                return {coerced[0]: velocities}
            return dict(
                zip(coerced, (velocities[:, i, :] for i in range(len(coerced))), strict=True)
            )
        return velocities

    def marker_speed(
        self,
        marker: M | str,
        *,
        modeled: bool = False,
    ) -> np.ndarray:
        return speed_from_velocity(
            self.marker_velocities(marker, modeled=modeled)
        )

    def patch_points(
        self,
        patch: P | str | Sequence[P | str],
        *,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[P, TimeVec3]:
        patches, is_many = _normalize_stringable_entity_input(
            patch,
            self.segment_view.spec.patch_type,
        )
        coerced = tuple(self._coerce_patch(p) for p in patches)
        world = self._patch_points_many(coerced)
        if not is_many and not return_dict:
            return world[:, 0, :]
        if return_dict:
            return dict(zip(coerced, (world[:, i, :] for i in range(len(coerced))), strict=True))
        return world

    def patch_normals(
        self,
        patch: P | str | Sequence[P | str],
        *,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[P, TimeVec3]:
        patches, is_many = _normalize_stringable_entity_input(
            patch,
            self.segment_view.spec.patch_type,
        )
        coerced = tuple(self._coerce_patch(p) for p in patches)
        world = self._patch_normals_many(coerced)
        if not is_many and not return_dict:
            return world[:, 0, :]
        if return_dict:
            return dict(zip(coerced, (world[:, i, :] for i in range(len(coerced))), strict=True))
        return world

    def patch_velocities(
        self,
        patch: P | str | Sequence[P | str],
        *,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[P, TimeVec3]:
        patches, is_many = _normalize_stringable_entity_input(
            patch,
            self.segment_view.spec.patch_type,
        )
        coerced = tuple(self._coerce_patch(p) for p in patches)
        positions = self.patch_points(patch, return_dict=False)
        velocities = self._differentiate(positions)
        if not is_many and not return_dict:
            return velocities
        if return_dict:
            if velocities.ndim == 2:
                return {coerced[0]: velocities}
            return dict(
                zip(coerced, (velocities[:, i, :] for i in range(len(coerced))), strict=True)
            )
        return velocities

    def patch_speed(self, patch: P | str) -> np.ndarray:
        return speed_from_velocity(self.patch_velocities(self._coerce_patch(patch)))

    def patch_contacts(
        self,
        patch: P | str | Sequence[P | str],
        *,
        return_dict: bool = False,
    ) -> np.ndarray | Mapping[P, np.ndarray]:
        contact_track = _contact_track(self.mocap)
        if contact_track is None:
            raise ValueError("No contact track is attached to this mocap track")
        patches, is_many = _normalize_stringable_entity_input(
            patch,
            self.segment_view.spec.patch_type,
        )
        coerced = tuple(self._coerce_patch(p) for p in patches)
        targets = [self._patch_target(p) for p in coerced]
        sliced = _slice_contact_track(contact_track, self.mocap)
        if not self.indices:
            arrays = [np.empty((0,), dtype=np.bool_) for _ in targets]
        else:
            arrays = [sliced.state(target) for target in targets]
        if not is_many and not return_dict:
            return arrays[0]
        if return_dict:
            return dict(zip(coerced, arrays, strict=True))
        return stack_entity_arrays(coerced, arrays, return_dict=False)

    def _source_track(self) -> MocapTrack:
        if isinstance(self.mocap, MocapTrack):
            return self.mocap
        return self.mocap.source

    def _segment_key(self) -> SegmentKey:
        return SegmentKey(
            self.segment_view.subject_id,
            self.segment_view.segment_id,
        )

    def _index_array(self) -> np.ndarray:
        return np.array(self.indices, dtype=np.intp)

    def _rotation_matrices(self) -> TimeMat3:
        if not self.indices:
            return np.empty((0, 3, 3), dtype=np.float64)
        source = self._source_track()
        full = source.segment_rotations(self._segment_key())
        return full[self._index_array()]

    def _differentiate(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return np.empty_like(values)
        if len(values) == 1:
            return np.zeros_like(values)
        return np.gradient(values, self.timestamps, axis=0)

    def _coerce_marker(self, marker: MarkerId | str) -> M:
        if isinstance(marker, self.segment_view.spec.marker_type):
            return marker
        if isinstance(marker, str):
            try:
                return self.segment_view.spec.marker_type(marker)
            except ValueError as exc:
                raise KeyError(
                    f"Segment {self.segment_view.spec.segment!r} has no marker {marker!r}"
                ) from exc
        raise TypeError(
            f"Expected marker of type {self.segment_view.spec.marker_type.__name__} "
            f"or str, got {type(marker).__name__}"
        )

    def _coerce_patch(self, patch: PatchId | str) -> P:
        if isinstance(patch, self.segment_view.spec.patch_type):
            return patch
        if isinstance(patch, str):
            try:
                return self.segment_view.spec.patch_type(patch)
            except ValueError as exc:
                raise KeyError(
                    f"Segment {self.segment_view.spec.segment!r} has no patch {patch!r}"
                ) from exc
        raise TypeError(
            f"Expected patch of type {self.segment_view.spec.patch_type.__name__} "
            f"or str, got {type(patch).__name__}"
        )

    def _modeled_marker_positions_many(
        self,
        markers: Sequence[M],
    ) -> TimeEntityVec3:
        num_markers = len(markers)
        if not self.indices:
            return np.empty((0, num_markers, 3), dtype=np.float64)
        if num_markers == 0:
            return np.empty((len(self.indices), 0, 3), dtype=np.float64)
        source = self._source_track()
        key = self._segment_key()
        rotations = source.segment_rotations(key)
        translations = source.segment_translations(key)
        local_positions = np.stack(
            [
                self.segment_view.spec.marker_position(marker)
                for marker in markers
            ],
            axis=0,
        )
        world_full = (
            np.einsum("tij,nj->tni", rotations, local_positions)
            + translations[:, None, :]
        )
        return world_full[self._index_array()]

    def _observed_marker_positions_many(
        self,
        markers: Sequence[M],
    ) -> TimeEntityVec3:
        num_markers = len(markers)
        if not self.indices:
            return np.empty((0, num_markers, 3), dtype=np.float64)
        if num_markers == 0:
            return np.empty((len(self.indices), 0, 3), dtype=np.float64)
        source = self._source_track()
        full = source.observed_marker_positions_for_segment(
            self.segment_view.subject_id,
            self.segment_view.spec,
        )
        marker_indices = [marker.index for marker in markers]
        return full[self._index_array()][:, marker_indices, :]

    def _marker_positions_one(self, marker: M, *, modeled: bool) -> TimeVec3:
        if not self.indices:
            return np.empty((0, 3), dtype=np.float64)
        if modeled:
            return self._modeled_marker_positions_many((marker,))[:, 0, :]
        return self._observed_marker_positions(marker)

    def _observed_marker_positions(self, marker: M) -> TimeVec3:
        return self._observed_marker_positions_many((marker,))[:, 0, :]

    def _patch_points_segment_many(self, patches: Sequence[P]) -> np.ndarray:
        if not patches:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack(
            [
                self.segment_view.patch(patch).transform_segment_patch.translation
                for patch in patches
            ],
            axis=0,
        )

    def _patch_normals_segment_many(self, patches: Sequence[P]) -> np.ndarray:
        if not patches:
            return np.empty((0, 3), dtype=np.float64)
        return np.stack(
            [
                self.segment_view.patch(patch).transform_segment_patch.rotation[:, 2]
                for patch in patches
            ],
            axis=0,
        )

    def _patch_points_many(self, patches: Sequence[P]) -> TimeEntityVec3:
        num_patches = len(patches)
        if not self.indices:
            return np.empty((0, num_patches, 3), dtype=np.float64)
        if num_patches == 0:
            return np.empty((len(self.indices), 0, 3), dtype=np.float64)
        source = self._source_track()
        key = self._segment_key()
        rotations = source.segment_rotations(key)
        translations = source.segment_translations(key)
        local_points = self._patch_points_segment_many(patches)
        # R: (T, 3, 3), P: (N, 3), t: (T, 3)
        world_full = (
            np.einsum("tij,nj->tni", rotations, local_points)
            + translations[:, None, :]
        )
        return world_full[self._index_array(), :, :]

    def _patch_normals_many(self, patches: Sequence[P]) -> TimeEntityVec3:
        num_patches = len(patches)
        if not self.indices:
            return np.empty((0, num_patches, 3), dtype=np.float64)
        if num_patches == 0:
            return np.empty((len(self.indices), 0, 3), dtype=np.float64)
        source = self._source_track()
        key = self._segment_key()
        rotations = source.segment_rotations(key)
        local_normals = self._patch_normals_segment_many(patches)
        # R: (T, 3, 3), N: (N, 3)
        world_full = np.einsum("tij,nj->tni", rotations, local_normals)
        return world_full[self._index_array(), :, :]

    def _patch_target(self, patch: P) -> PatchTarget[P]:
        handle: PatchHandle[P] = self.segment_view.spec.patch(patch)
        return PatchTarget(subject=self.segment_view.subject_id, handle=handle)


def _mocap_source(mocap: MocapTrack | MocapTrackView) -> MocapTrack:
    if isinstance(mocap, MocapTrackView):
        return mocap.source
    return mocap


def _mocap_indices(mocap: MocapTrack | MocapTrackView) -> tuple[int, ...]:
    if isinstance(mocap, MocapTrackView):
        return mocap.indices
    return resolve_indices(len(mocap.timestamps), None)


def _mocap_timestamps(mocap: MocapTrack | MocapTrackView) -> np.ndarray:
    source = _mocap_source(mocap)
    indices = _mocap_indices(mocap)
    return slice_timestamps(source.timestamps, indices)


def _contact_track(mocap: MocapTrack | MocapTrackView) -> ContactTrack | None:
    return _mocap_source(mocap).contacts


def _slice_contact_track(
    contact_track: ContactTrack,
    mocap: MocapTrack | MocapTrackView,
) -> ContactTrackView:
    indices = _mocap_indices(mocap)
    return ContactTrackView(source=contact_track, indices=indices)


@dataclass(frozen=True, slots=True)
class _MocapMarkerAccessor[M: MarkerId, P: PatchId]:
    """Single marker accessor over a mocap segment query view."""

    segment_view: MocapSegmentTrackView[M, P]
    marker_id: M

    def positions(self, *, modeled: bool = False) -> TimeVec3:
        return self.segment_view.marker_positions(self.marker_id, modeled=modeled)


@dataclass(frozen=True, slots=True)
class _MocapPatchAccessor[M: MarkerId, P: PatchId]:
    """Single patch accessor over a mocap segment query view."""

    segment_view: MocapSegmentTrackView[M, P]
    patch_id: P

    def points(self) -> TimeVec3:
        return self.segment_view.patch_points(self.patch_id)


def _mocap_subject_views(
    mocap: MocapTrack | MocapTrackView,
) -> Mapping[str, MocapSubjectTrackView]:
    scene_spec = mocap.scene.spec
    views = {
        subject_spec.subject.label: MocapSubjectTrackView(
            mocap=mocap,
            subject_id=subject_spec.subject,
        )
        for subject_spec in scene_spec.iter_subjects()
    }
    return cast(Mapping[str, MocapSubjectTrackView], MappingProxyType(views))


def _mocap_segment_views(
    subject_view: MocapSubjectTrackView,
) -> Mapping[str, MocapSegmentTrackView[Any, Any]]:
    subject_spec = subject_view.subject_spec
    views = {
        segment_spec.segment.label: subject_view.segment(segment_spec)
        for segment_spec in subject_spec.iter_segments()
    }
    return cast(Mapping[str, MocapSegmentTrackView[Any, Any]], MappingProxyType(views))


def _mocap_marker_accessors(
    segment_view: MocapSegmentTrackView[M, P],
) -> Mapping[str, _MocapMarkerAccessor[M, P]]:
    views = {
        marker.label: _MocapMarkerAccessor(
            segment_view=segment_view,
            marker_id=marker,
        )
        for marker in segment_view.segment_view.spec.marker_type
    }
    return cast(Mapping[str, _MocapMarkerAccessor[Any, Any]], MappingProxyType(views))


def _mocap_patch_accessors(
    segment_view: MocapSegmentTrackView[M, P],
) -> Mapping[str, _MocapPatchAccessor[M, P]]:
    views = {
        patch.label: _MocapPatchAccessor(
            segment_view=segment_view,
            patch_id=patch,
        )
        for patch in segment_view.segment_view.spec.patch_type
    }
    return cast(Mapping[str, _MocapPatchAccessor[Any, Any]], MappingProxyType(views))
