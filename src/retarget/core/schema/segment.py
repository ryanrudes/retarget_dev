"""Segment schema declaration and per-segment runtime data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

from retarget.core.enums import PoseFormat, RotationFormat
from retarget.core.formats import (
    finite_difference_velocity,
    pose_arrays_to_format,
    rotation_matrices_to_format,
    speed_from_velocity,
)
from retarget.core.targets import MarkerTarget, PatchTarget, SegmentTarget
from retarget.core.transform import RigidTransform
from retarget.core.types import TimeMat3, TimeQuat, TimeVec3

if TYPE_CHECKING:
    from retarget.core.schema.marker import Marker
    from retarget.core.schema.patch import Patch

class Segments(TypedDict):
    """Base class for typed segment schema declarations."""


@dataclass(slots=True, eq=False)
class _SegmentRuntime:
    """Per-segment time-series data attached to a bound segment subtree.

    Arrays are already materialized at the bound track's (possibly sliced) time
    resolution. ``observed_markers`` is keyed by marker ``mocap_name`` and always
    contains an entry for every declared marker (NaN rows where unobserved).
    ``contacts``/``confidences`` are keyed by authored patch name.
    """

    timestamps: np.ndarray
    translations: np.ndarray
    rotations: np.ndarray
    observed_markers: Mapping[str, np.ndarray]
    contacts: Mapping[str, np.ndarray]
    confidences: Mapping[str, np.ndarray]


@dataclass(slots=True, eq=False)
class _SegmentBinding:
    subject: str
    segment: str
    runtime: _SegmentRuntime | None


_UNBOUND_MESSAGE = (
    "This {what} is not bound to loaded data. Bind a scene with bind_scene(...) "
    "for static/geometry access, or query it through a loaded MocapTrack for "
    "time-series access."
)

from retarget.core.schema.marker import Markers  # noqa: E402
from retarget.core.schema.patch import Patches  # noqa: E402


@dataclass(frozen=True, slots=True)
class Segment[MarkersT: Markers, PatchesT: Patches]:
    """A rigid segment: typed marker/patch vocabularies plus bound pose queries."""

    markers: MarkersT
    patches: PatchesT
    mocap_name: str | None = None
    _binding: _SegmentBinding | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def translations(self) -> TimeVec3:
        """Segment-origin world translations with shape ``(T, 3)``."""
        return cast(TimeVec3, self._runtime().translations)

    def rotations(
        self,
        *,
        format: RotationFormat = RotationFormat.MATRIX,
    ) -> TimeMat3 | TimeQuat | np.ndarray:
        """Segment world rotations in the requested representation."""
        return rotation_matrices_to_format(self._runtime().rotations, format=format)

    def poses(
        self,
        *,
        format: PoseFormat = PoseFormat.RIGID_TRANSFORM,
    ) -> tuple[RigidTransform, ...] | np.ndarray:
        """Segment world poses in the requested representation."""
        runtime = self._runtime()
        return pose_arrays_to_format(
            runtime.translations, runtime.rotations, format=format
        )

    def linear_velocities(self) -> TimeVec3:
        """Segment-origin world velocities with shape ``(T, 3)``."""
        runtime = self._runtime()
        return cast(
            TimeVec3,
            finite_difference_velocity(runtime.translations, runtime.timestamps),
        )

    def speed(self) -> np.ndarray:
        """Segment-origin world speed with shape ``(T,)``."""
        return speed_from_velocity(self.linear_velocities())

    def marker_positions(
        self,
        *markers: str,
        modeled: bool = False,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch marker positions: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``."""
        if as_dict:
            return {
                name: self._marker(name).positions(modeled=modeled) for name in markers
            }
        return self._stack(
            [self._marker(name).positions(modeled=modeled) for name in markers]
        )

    def marker_velocities(
        self,
        *markers: str,
        modeled: bool = False,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch marker velocities: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``."""
        if as_dict:
            return {
                name: self._marker(name).velocities(modeled=modeled)
                for name in markers
            }
        return self._stack(
            [self._marker(name).velocities(modeled=modeled) for name in markers]
        )

    def patch_points(
        self,
        *patches: str,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch patch points: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``."""
        if as_dict:
            return {name: self._patch(name).points() for name in patches}
        return self._stack([self._patch(name).points() for name in patches])

    def patch_normals(
        self,
        *patches: str,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch patch normals: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``."""
        if as_dict:
            return {name: self._patch(name).normals() for name in patches}
        return self._stack([self._patch(name).normals() for name in patches])

    def patch_velocities(
        self,
        *patches: str,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch patch-point velocities: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``."""
        if as_dict:
            return {name: self._patch(name).velocities() for name in patches}
        return self._stack([self._patch(name).velocities() for name in patches])

    def patch_contacts(
        self,
        *patches: str,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Batch patch contact state: ``(T, N)`` stacked, or ``{name: (T,)}``."""
        if as_dict:
            return {name: self._patch(name).contacts() for name in patches}
        arrays = [self._patch(name).contacts() for name in patches]
        if not arrays:
            return np.empty((len(self._runtime().timestamps), 0), dtype=np.bool_)
        return np.stack(arrays, axis=1)

    def segment_target(self) -> SegmentTarget:
        """Stable scene-level identity for this segment."""
        binding = self._require_binding()
        return SegmentTarget(subject=binding.subject, segment=binding.segment)

    def marker_target(self, marker: str) -> MarkerTarget:
        """Stable scene-level identity for one marker on this segment."""
        binding = self._require_binding()
        if marker not in self.markers:
            raise KeyError(self._missing_marker_message(marker))
        return MarkerTarget(
            subject=binding.subject, segment=binding.segment, marker=marker
        )

    def patch_target(self, patch: str) -> PatchTarget:
        """Stable scene-level identity for one patch on this segment."""
        binding = self._require_binding()
        if patch not in self.patches:
            raise KeyError(self._missing_patch_message(patch))
        return PatchTarget(
            subject=binding.subject, segment=binding.segment, patch=patch
        )

    def _marker(self, name: str) -> Marker:
        from retarget.core.schema.marker import Marker as MarkerCls

        try:
            return cast(MarkerCls, cast(Any, self.markers)[name])
        except KeyError as exc:
            raise KeyError(self._missing_marker_message(name)) from exc

    def _patch(self, name: str) -> Patch:
        from retarget.core.schema.patch import Patch as PatchCls

        try:
            return cast(PatchCls, cast(Any, self.patches)[name])
        except KeyError as exc:
            raise KeyError(self._missing_patch_message(name)) from exc

    def _stack(self, arrays: list[TimeVec3]) -> np.ndarray:
        if not arrays:
            runtime = self._runtime()
            return np.empty((len(runtime.timestamps), 0, 3), dtype=np.float64)
        return np.stack(arrays, axis=1)

    def _missing_marker_message(self, name: str) -> str:
        label = self._binding.segment if self._binding is not None else "segment"
        return f"Segment {label!r} has no marker {name!r}"

    def _missing_patch_message(self, name: str) -> str:
        label = self._binding.segment if self._binding is not None else "segment"
        return f"Segment {label!r} has no patch {name!r}"

    def _require_binding(self) -> _SegmentBinding:
        if self._binding is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Segment"))
        return self._binding

    def _runtime(self) -> _SegmentRuntime:
        binding = self._require_binding()
        if binding.runtime is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Segment"))
        return binding.runtime
