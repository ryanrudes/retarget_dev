"""Segment schema declaration and per-segment runtime data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from retarget.core.enums import PoseFormat, RotationFormat
from retarget.core.formats import (
    JumpDetector,
    finite_difference_velocity,
    pose_arrays_to_format,
    rotation_matrices_to_format,
    speed_from_velocity,
)
from retarget.core.schema.base import _Schema, _schema_values
from retarget.core.targets import SegmentTarget
from retarget.core.transform import RigidTransform
from retarget.core.types import FloatArray1D, TimeBool, TimeMat3, TimeQuat, TimeVec3

if TYPE_CHECKING:
    from retarget.core.schema.marker import Marker
    from retarget.core.schema.patch import Patch


@dataclass(frozen=True, slots=True)
class Segments(_Schema):
    """Base class for typed segment schema declarations."""


@dataclass(slots=True, eq=False)
class _SegmentRuntime:
    """Per-segment time-series data attached to a bound segment subtree.

    Arrays are already materialized at the bound track's (possibly sliced) time
    resolution. ``observed_markers`` is keyed by marker ``mocap_name`` and always
    contains an entry for every declared marker (NaN rows where unobserved).
    ``contacts``/``confidences`` are keyed by authored patch name.
    ``support_contacts``/``support_scores`` are keyed by authored patch name and
    then by support name (an attached :class:`SupportStateTrack`).
    """

    timestamps: np.ndarray
    translations: np.ndarray
    rotations: np.ndarray
    observed_markers: Mapping[str, np.ndarray]
    contacts: Mapping[str, np.ndarray]
    confidences: Mapping[str, np.ndarray]
    support_contacts: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    support_scores: Mapping[str, Mapping[str, np.ndarray]] = field(default_factory=dict)
    support_invalid: Mapping[str, np.ndarray] = field(default_factory=dict)
    pose_filled: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.bool_))


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
    _binding: _SegmentBinding | None = field(default=None, init=False, compare=False, repr=False)

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
        return pose_arrays_to_format(runtime.translations, runtime.rotations, format=format)

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

    def pose_coverage(self) -> FloatArray1D:
        """Fraction of this segment's markers observed per frame, shape ``(T,)``.

        A proxy for how well the segment pose is supported by data: the Vicon
        body solve degrades as markers drop out. Frames whose pose was synthesized
        by :func:`~retarget.demo.fill_pose_gaps` count as fully covered. Returns
        all ones when no marker observations are attached (coverage unknown).
        """
        runtime = self._runtime()
        num = len(runtime.timestamps)
        filled = _pose_filled_mask(runtime, num)
        if not runtime.observed_markers:
            return cast(FloatArray1D, np.ones(num, dtype=np.float64))
        coverage = _visible_marker_count(runtime.observed_markers, num).astype(np.float64) / len(
            runtime.observed_markers
        )
        return cast(FloatArray1D, np.where(filled, 1.0, coverage))

    def valid(
        self,
        *,
        min_coverage: float = 0.5,
        jumps: JumpDetector | None = None,
    ) -> TimeBool:
        """Per-frame mask of trustworthy segment pose, shape ``(T,)``.

        A frame is trustworthy when its marker coverage is at least
        ``min_coverage`` (or its pose was synthesized by
        :func:`~retarget.demo.fill_pose_gaps`) *and* it is not a garbage sample
        per the optional ``jumps`` detector. Jumps are measured on the patch
        points, so a rotation glitch is caught even though the origin barely
        moves. ``jumps`` is opt-in (``None`` = coverage only); pass
        :func:`~retarget.core.formats.speed_limit` or
        :func:`~retarget.core.formats.robust_jumps`.
        """
        coverage = np.asarray(self.pose_coverage())
        enough = coverage >= min_coverage
        return cast(TimeBool, enough & ~self._pose_jump_mask(jumps))

    def _pose_jump_mask(self, jumps: JumpDetector | None) -> np.ndarray:
        """Garbage mask over this segment's geometry patch points (rotation-aware)."""
        runtime = self._runtime()
        num = len(runtime.timestamps)
        if jumps is None:
            return np.zeros(num, dtype=np.bool_)
        geometry_patches: list[Patch] = [
            patch for patch in _schema_values(self.patches) if patch.has_geometry()
        ]
        if not geometry_patches:
            return np.asarray(jumps(runtime.translations, runtime.timestamps))
        mask = np.zeros(num, dtype=np.bool_)
        for patch in geometry_patches:
            mask |= np.asarray(jumps(patch.points(), runtime.timestamps))
        return mask

    def pose_filled(self) -> TimeBool:
        """Per-frame mask of poses synthesized by :func:`~retarget.demo.fill_pose_gaps`.

        All False when no gaps were filled. Use it to render or exclude
        interpolated (non-measured) frames.
        """
        runtime = self._runtime()
        return cast(TimeBool, _pose_filled_mask(runtime, len(runtime.timestamps)))

    def marker_positions(
        self,
        markers: Sequence[Marker],
        *,
        modeled: bool = False,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch marker positions: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``.

        ``markers`` are bound marker symbols (e.g. ``segment.markers.heel``); the
        ``as_dict`` keys are each symbol's own authored marker name.
        """
        if as_dict:
            return {m.target.marker: m.positions(modeled=modeled) for m in markers}
        return self._stack([m.positions(modeled=modeled) for m in markers])

    def patch_points(
        self,
        patches: Sequence[Patch],
        *,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch patch points: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``.

        ``patches`` are bound patch symbols (e.g. ``segment.patches.sole``); the
        ``as_dict`` keys are each symbol's own authored patch name.
        """
        if as_dict:
            return {p.target.patch: p.points() for p in patches}
        return self._stack([p.points() for p in patches])

    def patch_normals(
        self,
        patches: Sequence[Patch],
        *,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, TimeVec3]:
        """Batch patch normals: ``(T, N, 3)`` stacked, or ``{name: (T, 3)}``.

        ``patches`` are bound patch symbols (e.g. ``segment.patches.sole``); the
        ``as_dict`` keys are each symbol's own authored patch name.
        """
        if as_dict:
            return {p.target.patch: p.normals() for p in patches}
        return self._stack([p.normals() for p in patches])

    def patch_contacts(
        self,
        patches: Sequence[Patch],
        *,
        as_dict: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Batch patch contact state: ``(T, N)`` stacked, or ``{name: (T,)}``.

        ``patches`` are bound patch symbols (e.g. ``segment.patches.sole``); the
        ``as_dict`` keys are each symbol's own authored patch name.
        """
        if as_dict:
            return {p.target.patch: p.contacts() for p in patches}
        arrays = [p.contacts() for p in patches]
        if not arrays:
            return np.empty((len(self._runtime().timestamps), 0), dtype=np.bool_)
        return np.stack(arrays, axis=1)

    def segment_target(self) -> SegmentTarget:
        """Stable scene-level identity for this segment."""
        binding = self._require_binding()
        return SegmentTarget(subject=binding.subject, segment=binding.segment)

    def _stack(self, arrays: list[TimeVec3]) -> np.ndarray:
        if not arrays:
            runtime = self._runtime()
            return np.empty((len(runtime.timestamps), 0, 3), dtype=np.float64)
        return np.stack(arrays, axis=1)

    def _require_binding(self) -> _SegmentBinding:
        if self._binding is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Segment"))
        return self._binding

    def _runtime(self) -> _SegmentRuntime:
        binding = self._require_binding()
        if binding.runtime is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Segment"))
        return binding.runtime


def _visible_marker_count(
    observed_markers: Mapping[str, np.ndarray],
    num_timesteps: int,
) -> np.ndarray:
    """Per-frame count of markers with a fully-finite (non-occluded) position."""
    counts = np.zeros(num_timesteps, dtype=np.int64)
    for array in observed_markers.values():
        counts += np.isfinite(np.asarray(array, dtype=np.float64)).all(axis=1)
    return counts


def _pose_filled_mask(runtime: _SegmentRuntime, num_timesteps: int) -> np.ndarray:
    """Per-frame mask of poses synthesized by gap filling (all False if unset)."""
    filled = np.asarray(runtime.pose_filled, dtype=np.bool_)
    if filled.shape != (num_timesteps,):
        return np.zeros(num_timesteps, dtype=np.bool_)
    return filled
