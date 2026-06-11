"""Mocap demonstration track and time-series query views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

import numpy as np

from retarget.core.enums import MarkerId, PatchId, PoseFormat, RotationFormat, SegmentId, SubjectId
from retarget.core.keys import SegmentKey
from retarget.core.handles import PatchHandle
from retarget.core.specs import SceneSpec, SegmentSpec
from retarget.core.state import SceneState
from retarget.core.targets import PatchTarget
from retarget.core.transform import RigidTransform
from retarget.core.types import TimeEntityVec3, TimeMat3, TimeQuat, TimeVec3
from retarget.core.views import SceneView, SegmentView
from retarget.demo._mocap_arrays import (
    MocapArrayCache,
    pose_arrays_to_format,
    rotation_matrices_to_format,
)
from retarget.demo._query_utils import (
    coerce_marker_id,
    coerce_patch_id,
    finite_difference_velocity,
    normalize_entity_input,
    resolve_indices,
    slice_timestamps,
    speed_from_velocity,
    stack_entity_arrays,
)
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.io import ViconMarkersFrame


def _validate_timestamps(timestamps: np.ndarray) -> None:
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


@dataclass(frozen=True, slots=True)
class MocapTrack:
    """Time-indexed mocap track over a scene spec and runtime state."""

    scene_spec: SceneSpec
    state: SceneState
    timestamps: np.ndarray
    marker_frames: tuple[ViconMarkersFrame, ...] | None = None
    contacts: ContactTrack | None = None
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
        )

    def with_rebased_time(self) -> MocapTrack:
        if len(self.timestamps) == 0:
            return self
        return self.with_timestamps(self.timestamps - self.timestamps[0])

    @property
    def scene(self) -> SceneView:
        return SceneView(spec=self.scene_spec, state=self.state)

    def subject(self, subject: SubjectId) -> MocapSubjectTrackView:
        return MocapSubjectTrackView(mocap=self, subject_id=subject)

    def segment(
        self,
        subject: SubjectId,
        segment: SegmentId | SegmentSpec[Any, Any],
    ) -> MocapSegmentTrackView[Any, Any]:
        return self.subject(subject).segment(segment)

    def slice_time(self, start: float, stop: float) -> MocapTrackView:
        indices = _indices_for_time_range(self.timestamps, start, stop)
        return MocapTrackView(source=self, indices=indices)

    def nearest_index(self, time: float) -> int:
        if len(self.timestamps) == 0:
            raise IndexError("cannot query nearest_index on an empty mocap track")
        return int(np.argmin(np.abs(self.timestamps - time)))

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
        subject: SubjectId,
        segment: SegmentSpec[M, Any],
    ) -> np.ndarray:
        """Return full-track observed marker positions with shape ``(T, M, 3)``."""
        key = SegmentKey(subject, segment.segment)
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
        for timestep, frame in enumerate(self.marker_frames):
            for obs in frame.markers:
                if obs.subject_name != subject.label:
                    continue
                if obs.segment_name != segment.segment.label:
                    continue
                if obs.occluded:
                    continue
                try:
                    marker = segment.marker_type(obs.marker_name)
                except ValueError:
                    continue
                arr[timestep, marker.index, :] = obs.position_world
        self._array_cache.observed_markers[key] = arr
        return arr


@dataclass(frozen=True, slots=True)
class MocapTrackView:
    """Sliced view into a :class:`MocapTrack`."""

    source: MocapTrack
    indices: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def timestamps(self) -> np.ndarray:
        return slice_timestamps(self.source.timestamps, self.indices)

    @property
    def scene(self) -> SceneView:
        return self.source.scene

    def subject(self, subject: SubjectId) -> MocapSubjectTrackView:
        return MocapSubjectTrackView(mocap=self, subject_id=subject)

    def segment(
        self,
        subject: SubjectId,
        segment: SegmentId | SegmentSpec[Any, Any],
    ) -> MocapSegmentTrackView[Any, Any]:
        return self.subject(subject).segment(segment)

    def slice_time(self, start: float, stop: float) -> MocapTrackView:
        sub_indices = _indices_for_time_range(self.timestamps, start, stop)
        remapped = tuple(self.indices[i] for i in sub_indices)
        return MocapTrackView(source=self.source, indices=remapped)

    def nearest_index(self, time: float) -> int:
        if len(self.timestamps) == 0:
            raise IndexError("cannot query nearest_index on an empty mocap view")
        return int(np.argmin(np.abs(self.timestamps - time)))


@dataclass(frozen=True, slots=True)
class MocapSubjectTrackView:
    """Subject-scoped entry point into a mocap track or sliced view."""

    mocap: MocapTrack | MocapTrackView
    subject_id: SubjectId

    @overload
    def segment[M: MarkerId, P: PatchId](
        self,
        segment: SegmentSpec[M, P],
    ) -> MocapSegmentTrackView[M, P]: ...

    @overload
    def segment(
        self,
        segment: SegmentId,
    ) -> MocapSegmentTrackView[Any, Any]: ...

    def segment(
        self,
        segment: SegmentId | SegmentSpec[Any, Any],
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
        marker: M,
        *,
        modeled: bool = False,
    ) -> TimeVec3: ...

    @overload
    def marker_positions(
        self,
        marker: Sequence[M],
        *,
        modeled: bool = False,
        return_dict: Literal[False] = False,
    ) -> TimeEntityVec3: ...

    @overload
    def marker_positions(
        self,
        marker: Sequence[M],
        *,
        modeled: bool = False,
        return_dict: Literal[True],
    ) -> Mapping[M, TimeVec3]: ...

    def marker_positions(
        self,
        marker: M | Sequence[M],
        *,
        modeled: bool = False,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[M, TimeVec3]:
        markers, is_many = normalize_entity_input(marker, self.segment_view.spec.marker_type)
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
        marker: M | Sequence[M],
        *,
        modeled: bool = False,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[M, TimeVec3]:
        markers, is_many = normalize_entity_input(marker, self.segment_view.spec.marker_type)
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
        marker: M,
        *,
        modeled: bool = False,
    ) -> np.ndarray:
        return speed_from_velocity(
            self.marker_velocities(marker, modeled=modeled)
        )

    def patch_points(
        self,
        patch: P | Sequence[P],
        *,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[P, TimeVec3]:
        patches, is_many = normalize_entity_input(patch, self.segment_view.spec.patch_type)
        coerced = tuple(self._coerce_patch(p) for p in patches)
        world = self._patch_points_many(coerced)
        if not is_many and not return_dict:
            return world[:, 0, :]
        if return_dict:
            return dict(zip(coerced, (world[:, i, :] for i in range(len(coerced))), strict=True))
        return world

    def patch_normals(
        self,
        patch: P | Sequence[P],
        *,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[P, TimeVec3]:
        patches, is_many = normalize_entity_input(patch, self.segment_view.spec.patch_type)
        coerced = tuple(self._coerce_patch(p) for p in patches)
        world = self._patch_normals_many(coerced)
        if not is_many and not return_dict:
            return world[:, 0, :]
        if return_dict:
            return dict(zip(coerced, (world[:, i, :] for i in range(len(coerced))), strict=True))
        return world

    def patch_velocities(
        self,
        patch: P | Sequence[P],
        *,
        return_dict: bool = False,
    ) -> TimeVec3 | TimeEntityVec3 | Mapping[P, TimeVec3]:
        patches, is_many = normalize_entity_input(patch, self.segment_view.spec.patch_type)
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

    def patch_speed(self, patch: P) -> np.ndarray:
        return speed_from_velocity(self.patch_velocities(self._coerce_patch(patch)))

    def patch_contacts(
        self,
        patch: P | Sequence[P],
        *,
        return_dict: bool = False,
    ) -> np.ndarray | Mapping[P, np.ndarray]:
        contact_track = _contact_track(self.mocap)
        if contact_track is None:
            raise ValueError("No contact track is attached to this mocap track")
        patches, is_many = normalize_entity_input(patch, self.segment_view.spec.patch_type)
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

    def _coerce_marker(self, marker: MarkerId) -> M:
        return coerce_marker_id(marker, self.segment_view.spec.marker_type)

    def _coerce_patch(self, patch: PatchId) -> P:
        return coerce_patch_id(patch, self.segment_view.spec.patch_type)

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


def _indices_for_time_range(
    timestamps: np.ndarray,
    start: float,
    stop: float,
) -> tuple[int, ...]:
    mask = (timestamps >= start) & (timestamps < stop)
    return tuple(int(index) for index in np.nonzero(mask)[0])


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
