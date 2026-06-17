"""Typed scene authoring schema and the bound runtime query surface.

This module defines a single set of dual-purpose types:

* ``Markers`` / ``Patches`` / ``Segments`` / ``Subjects`` are ``TypedDict`` bases
  the user subclasses to declare the *shape* of a scene. Because each subclass is
  a concrete ``TypedDict``, literal-key access projects to the declared field
  type, which is what gives the deep query chain perfect static typing.
* ``Marker`` / ``Patch`` / ``Segment`` / ``Subject`` are frozen dataclasses the
  user instantiates to author concrete scene data. The same instances double as
  the runtime query surface: once a :class:`~retarget.demo.mocap.MocapTrack`
  binds them to loaded data, ``marker.positions()``/``segment.translations()``/
  ``patch.points()`` answer time-series queries.

Authoring objects carry a private, non-init ``_binding`` that links them to
loaded track data. It is ``None`` while authoring and ignored by equality/repr,
so the public constructors stay pure authoring (``Marker(vicon_name=...)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import numpy as np

from retarget.core.contact_region import ContactRegion, RectangularRegion
from retarget.core.enums import PoseFormat, RotationFormat
from retarget.core.formats import (
    finite_difference_velocity,
    pose_arrays_to_format,
    rotation_matrices_to_format,
    speed_from_velocity,
)
from retarget.core.keys import SegmentKey
from retarget.core.targets import MarkerTarget, PatchTarget, SegmentTarget
from retarget.core.transform import RigidTransform
from retarget.core.types import TimeMat3, TimeQuat, TimeVec3, Vec3


# ---------------------------------------------------------------------------
# Typed schema declaration bases
# ---------------------------------------------------------------------------


class Markers(TypedDict):
    """Base class for typed marker schema declarations."""


class Patches(TypedDict):
    """Base class for typed patch schema declarations."""


class Segments(TypedDict):
    """Base class for typed segment schema declarations."""


class Subjects(TypedDict):
    """Base class for typed subject schema declarations."""


# ---------------------------------------------------------------------------
# Runtime binding (private)
# ---------------------------------------------------------------------------


@dataclass(slots=True, eq=False)
class _SegmentRuntime:
    """Per-segment time-series data attached to a bound segment subtree.

    Arrays are already materialized at the bound track's (possibly sliced) time
    resolution. ``observed_markers`` is keyed by marker ``vicon_name`` and always
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
class _MarkerBinding:
    subject: str
    segment: str
    marker: str
    runtime: _SegmentRuntime | None


@dataclass(slots=True, eq=False)
class _PatchBinding:
    subject: str
    segment: str
    patch: str
    runtime: _SegmentRuntime | None


@dataclass(slots=True, eq=False)
class _SegmentBinding:
    subject: str
    segment: str
    runtime: _SegmentRuntime | None


@dataclass(slots=True, eq=False)
class _SubjectBinding:
    subject: str


_UNBOUND_MESSAGE = (
    "This {what} is not bound to loaded data. Bind a scene with bind_scene(...) "
    "for static/geometry access, or query it through a loaded MocapTrack for "
    "time-series access."
)


# ---------------------------------------------------------------------------
# Domain dataclasses (authoring + bound runtime surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Marker:
    """A marker: authoring metadata plus bound time-series queries."""

    vicon_name: str
    position_segment: Vec3 | None = field(default=None, compare=False)
    _binding: _MarkerBinding | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def positions(self, *, modeled: bool = False) -> TimeVec3:
        """World-frame marker positions with shape ``(T, 3)``.

        ``modeled=False`` returns observed positions (NaN where unobserved);
        ``modeled=True`` rigidly transforms the segment-frame position.
        """
        runtime = self._runtime()
        if modeled:
            if self.position_segment is None:
                raise ValueError(
                    f"Marker {self._binding.marker!r} has no segment-frame "  # type: ignore[union-attr]
                    "position for modeled queries"
                )
            local = np.asarray(self.position_segment, dtype=np.float64)
            world = np.einsum("tij,j->ti", runtime.rotations, local)
            return cast(TimeVec3, world + runtime.translations)
        try:
            return cast(TimeVec3, runtime.observed_markers[self.vicon_name])
        except KeyError as exc:
            raise ValueError(
                "Observed marker positions require marker_frames on the mocap track"
            ) from exc

    def velocities(self, *, modeled: bool = False) -> TimeVec3:
        """World-frame marker velocities with shape ``(T, 3)``."""
        runtime = self._runtime()
        return cast(
            TimeVec3,
            finite_difference_velocity(
                self.positions(modeled=modeled), runtime.timestamps
            ),
        )

    def speed(self, *, modeled: bool = False) -> np.ndarray:
        """World-frame marker speed with shape ``(T,)``."""
        return speed_from_velocity(self.velocities(modeled=modeled))

    @property
    def target(self) -> MarkerTarget:
        """Stable scene-level identity for this marker."""
        binding = self._require_binding()
        return MarkerTarget(
            subject=binding.subject,
            segment=binding.segment,
            marker=binding.marker,
        )

    def _require_binding(self) -> _MarkerBinding:
        if self._binding is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Marker"))
        return self._binding

    def _runtime(self) -> _SegmentRuntime:
        binding = self._require_binding()
        if binding.runtime is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Marker"))
        return binding.runtime


@dataclass(frozen=True, slots=True)
class Patch:
    """A contact patch: authoring metadata plus bound time-series queries."""

    label: str
    transform_segment_patch: RigidTransform | None = field(default=None, compare=False)
    region: ContactRegion | None = field(default=None, compare=False)
    frame: str | None = None
    _binding: _PatchBinding | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @classmethod
    def rectangular(
        cls,
        label: str,
        *,
        transform_segment_patch: RigidTransform,
        width: float,
        height: float,
        frame: str | None = None,
    ) -> "Patch":
        """Build a rectangular patch declaration with explicit geometry."""
        return cls(
            label=label,
            transform_segment_patch=transform_segment_patch,
            region=RectangularRegion(width=width, height=height),
            frame=frame,
        )

    def points(self) -> TimeVec3:
        """World-frame patch origin/contact points with shape ``(T, 3)``."""
        runtime = self._runtime()
        local = np.asarray(self._geometry().translation, dtype=np.float64)
        world = np.einsum("tij,j->ti", runtime.rotations, local)
        return cast(TimeVec3, world + runtime.translations)

    def normals(self) -> TimeVec3:
        """World-frame patch normals with shape ``(T, 3)``."""
        runtime = self._runtime()
        local_normal = np.asarray(self._geometry().rotation[:, 2], dtype=np.float64)
        return cast(TimeVec3, np.einsum("tij,j->ti", runtime.rotations, local_normal))

    def velocities(self) -> TimeVec3:
        """World-frame patch-point velocities with shape ``(T, 3)``."""
        runtime = self._runtime()
        return cast(
            TimeVec3, finite_difference_velocity(self.points(), runtime.timestamps)
        )

    def contacts(self) -> np.ndarray:
        """Boolean contact state with shape ``(T,)`` from an attached contact track."""
        runtime = self._runtime()
        name = self._require_binding().patch
        try:
            return runtime.contacts[name]
        except KeyError as exc:
            raise ValueError(
                f"No contact track is attached for patch {name!r}"
            ) from exc

    def confidence(self) -> np.ndarray:
        """Contact confidence with shape ``(T,)`` from an attached contact track."""
        runtime = self._runtime()
        name = self._require_binding().patch
        try:
            return runtime.confidences[name]
        except KeyError as exc:
            raise ValueError(
                f"No contact confidence is attached for patch {name!r}"
            ) from exc

    @property
    def target(self) -> PatchTarget:
        """Stable scene-level identity for this patch."""
        binding = self._require_binding()
        return PatchTarget(
            subject=binding.subject,
            segment=binding.segment,
            patch=binding.patch,
        )

    def has_geometry(self) -> bool:
        """Whether this patch carries calibrated geometry."""
        return self.transform_segment_patch is not None and self.region is not None

    def _geometry(self) -> RigidTransform:
        if self.transform_segment_patch is None:
            name = self._binding.patch if self._binding is not None else self.label
            raise ValueError(
                f"Patch {name!r} is declared but has no calibrated geometry"
            )
        return self.transform_segment_patch

    def _require_binding(self) -> _PatchBinding:
        if self._binding is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Patch"))
        return self._binding

    def _runtime(self) -> _SegmentRuntime:
        binding = self._require_binding()
        if binding.runtime is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Patch"))
        return binding.runtime


@dataclass(frozen=True, slots=True)
class Segment[MarkersT: Markers, PatchesT: Patches]:
    """A rigid segment: typed marker/patch vocabularies plus bound pose queries."""

    markers: MarkersT
    patches: PatchesT
    vicon_name: str | None = None
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
        try:
            return cast(Mapping[str, Marker], self.markers)[name]
        except KeyError as exc:
            raise KeyError(self._missing_marker_message(name)) from exc

    def _patch(self, name: str) -> Patch:
        try:
            return cast(Mapping[str, Patch], self.patches)[name]
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


@dataclass(frozen=True, slots=True)
class Subject[SegmentsT: Segments]:
    """A subject: a typed mapping of named segments."""

    segments: SegmentsT
    vicon_name: str | None = None
    _binding: _SubjectBinding | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def segment_external_name(self, segment: str) -> str:
        """External (Vicon) name used by IO for one of this subject's segments."""
        seg = cast(Mapping[str, Segment[Any, Any]], self.segments)[segment]
        return seg.vicon_name or segment

    @property
    def external_name(self) -> str | None:
        """External (Vicon) subject name, if any."""
        return self.vicon_name


# ---------------------------------------------------------------------------
# Building / binding
# ---------------------------------------------------------------------------


def bind_scene[SubjectsT: Subjects](subjects: SubjectsT) -> SubjectsT:
    """Path-bind an authored scene so targets/geometry are available.

    Returns a structure of the same ``SubjectsT`` type whose subjects/segments/
    markers/patches know their compiled names (enabling ``*_target(...)`` and
    geometry) but are not yet bound to time-series data.
    """
    _validate_subjects(subjects)
    return _bind_subjects(subjects, runtimes=None)


def bind_subjects_runtime[SubjectsT: Subjects](
    subjects: SubjectsT,
    runtimes: Mapping[SegmentKey, _SegmentRuntime],
) -> SubjectsT:
    """Bind an authored scene to per-segment runtime data (used by MocapTrack)."""
    return _bind_subjects(subjects, runtimes=runtimes)


def _bind_subjects[SubjectsT: Subjects](
    subjects: SubjectsT,
    *,
    runtimes: Mapping[SegmentKey, _SegmentRuntime] | None,
) -> SubjectsT:
    bound = {
        name: _bind_subject(subject, name=name, runtimes=runtimes)
        for name, subject in cast(Mapping[str, Subject[Any]], subjects).items()
    }
    return cast(SubjectsT, bound)


def _bind_subject(
    subject: Subject[Any],
    *,
    name: str,
    runtimes: Mapping[SegmentKey, _SegmentRuntime] | None,
) -> Subject[Any]:
    bound_segments = {
        segment_name: _bind_segment(
            segment,
            subject=name,
            segment_name=segment_name,
            runtime=(
                None
                if runtimes is None
                else runtimes.get(SegmentKey(name, segment_name))
            ),
        )
        for segment_name, segment in subject.segments.items()
    }
    bound = Subject(segments=cast(Any, bound_segments), vicon_name=subject.vicon_name)
    object.__setattr__(bound, "_binding", _SubjectBinding(subject=name))
    return bound


def _bind_segment(
    segment: Segment[Any, Any],
    *,
    subject: str,
    segment_name: str,
    runtime: _SegmentRuntime | None,
) -> Segment[Any, Any]:
    bound_markers = {
        marker_name: _bind_marker(
            marker,
            subject=subject,
            segment=segment_name,
            marker_name=marker_name,
            runtime=runtime,
        )
        for marker_name, marker in segment.markers.items()
    }
    bound_patches = {
        patch_name: _bind_patch(
            patch,
            subject=subject,
            segment=segment_name,
            patch_name=patch_name,
            runtime=runtime,
        )
        for patch_name, patch in segment.patches.items()
    }
    bound = Segment(
        markers=cast(Any, bound_markers),
        patches=cast(Any, bound_patches),
        vicon_name=segment.vicon_name,
    )
    object.__setattr__(
        bound,
        "_binding",
        _SegmentBinding(subject=subject, segment=segment_name, runtime=runtime),
    )
    return bound


def _bind_marker(
    marker: Marker,
    *,
    subject: str,
    segment: str,
    marker_name: str,
    runtime: _SegmentRuntime | None,
) -> Marker:
    bound = Marker(
        vicon_name=marker.vicon_name,
        position_segment=marker.position_segment,
    )
    object.__setattr__(
        bound,
        "_binding",
        _MarkerBinding(
            subject=subject, segment=segment, marker=marker_name, runtime=runtime
        ),
    )
    return bound


def _bind_patch(
    patch: Patch,
    *,
    subject: str,
    segment: str,
    patch_name: str,
    runtime: _SegmentRuntime | None,
) -> Patch:
    bound = Patch(
        label=patch.label,
        transform_segment_patch=patch.transform_segment_patch,
        region=patch.region,
        frame=patch.frame,
    )
    object.__setattr__(
        bound,
        "_binding",
        _PatchBinding(
            subject=subject, segment=segment, patch=patch_name, runtime=runtime
        ),
    )
    return bound


def _validate_subjects(subjects: Mapping[str, Any]) -> None:
    if not subjects:
        raise ValueError("A scene must declare at least one subject")
    for subject_name, subject in subjects.items():
        if not subject_name:
            raise ValueError("Subject names must be non-empty")
        segments = cast(Mapping[str, Segment[Any, Any]], subject.segments)
        for segment_name, segment in segments.items():
            _validate_segment(subject_name, segment_name, segment)


def _validate_segment(
    subject_name: str,
    segment_name: str,
    segment: Segment[Any, Any],
) -> None:
    seen_vicon: dict[str, str] = {}
    for marker_name, marker in segment.markers.items():
        previous = seen_vicon.get(marker.vicon_name)
        if previous is not None:
            raise ValueError(
                "Duplicate Marker.vicon_name within "
                f"{subject_name!r}/{segment_name!r}: {marker.vicon_name!r} used by "
                f"{previous!r} and {marker_name!r}"
            )
        seen_vicon[marker.vicon_name] = marker_name
