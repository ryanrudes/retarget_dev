from __future__ import annotations

import keyword
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, TypedDict, cast

from retarget.core.axes import Z_UP_AXES
from retarget.core.contact_region import ContactRegion, RectangularRegion
from retarget.core.enums import (
    MarkerId,
    NameId,
    PatchId,
    SegmentId,
    SubjectId,
)
from retarget.core.handles import MarkerHandle, PatchHandle
from retarget.core.keys import SegmentKey
from retarget.core.specs import (
    MarkerSetSpec,
    MarkerSpec,
    PatchDeclarationSpec,
    PatchSpec,
    SceneSpec,
    SegmentSpec,
    SubjectSpec,
)
from retarget.core.targets import MarkerTarget, PatchTarget, SegmentTarget
from retarget.core.transform import RigidTransform

_NON_IDENTIFIER = re.compile(r"[^0-9A-Za-z_]")


class Markers(TypedDict):
    """Base class for typed marker schema declarations."""


class Patches(TypedDict):
    """Base class for typed patch schema declarations."""


class Segments(TypedDict):
    """Base class for typed segment schema declarations."""


class Subjects(TypedDict):
    """Base class for typed subject schema declarations."""


@dataclass(frozen=True, slots=True)
class Marker:
    """Authoring-time marker metadata."""

    vicon_name: str


@dataclass(frozen=True, slots=True)
class Patch:
    """Authoring-time patch metadata."""

    label: str
    transform_segment_patch: RigidTransform | None = None
    region: ContactRegion | None = None
    frame: str | None = None

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


@dataclass(frozen=True, slots=True)
class Segment[MarkersT: Markers, PatchesT: Patches]:
    """Authoring-time segment declaration."""

    markers: MarkersT
    patches: PatchesT


@dataclass(frozen=True, slots=True)
class Subject[SegmentsT: Segments]:
    """Authoring-time subject declaration."""

    segments: SegmentsT


@dataclass(frozen=True, slots=True)
class GeneratedIds:
    """Private runtime ID classes generated from authored field names."""

    subjects: type[SubjectId]
    segments: Mapping[SubjectId, type[SegmentId]]
    markers: Mapping[SegmentKey, type[MarkerId]]
    patches: Mapping[SegmentKey, type[PatchId]]


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledSegmentSpec[M: MarkerId, P: PatchId](SegmentSpec[M, P]):
    """SegmentSpec compiled from an authored segment declaration."""

    subject: SubjectId
    marker_defs: Mapping[M, Marker]
    patch_defs: Mapping[P, Patch]
    _marker_from_vicon_name: Mapping[str, M] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        SegmentSpec.__post_init__(self)
        object.__setattr__(
            self,
            "marker_defs",
            MappingProxyType(dict(self.marker_defs)),
        )
        object.__setattr__(
            self,
            "patch_defs",
            MappingProxyType(dict(self.patch_defs)),
        )

        marker_from_vicon_name: dict[str, M] = {}
        for marker, marker_def in self.marker_defs.items():
            self._check_marker(marker)
            if marker_def.vicon_name in marker_from_vicon_name:
                previous = marker_from_vicon_name[marker_def.vicon_name]
                raise ValueError(
                    "Duplicate Marker.vicon_name values within a segment: "
                    f"{marker_def.vicon_name!r} used by {previous!r} and {marker!r}"
                )
            marker_from_vicon_name[marker_def.vicon_name] = marker

        for patch in self.patch_defs:
            self._check_patch(patch)

        object.__setattr__(
            self,
            "_marker_from_vicon_name",
            MappingProxyType(marker_from_vicon_name),
        )

    def marker_spec(self, marker: M) -> MarkerSpec[M]:
        marker_id = self._coerce_marker(marker)
        marker_def = self.marker_defs[marker_id]
        return MarkerSpec(
            marker=marker_id,
            role=self.marker_set.role(marker_id),
            vicon_name=marker_def.vicon_name,
        )

    def marker_external_name(self, marker: M) -> str:
        marker_id = self._coerce_marker(marker)
        return self.marker_defs[marker_id].vicon_name

    def marker_from_external_name(self, marker_name: str) -> M:
        try:
            return self._marker_from_vicon_name[marker_name]
        except KeyError:
            return SegmentSpec.marker_from_external_name(self, marker_name)

    def marker_from_vicon_name(self, marker_name: str) -> M:
        return self.marker_from_external_name(marker_name)

    def marker(self, marker: M | str) -> MarkerHandle[M]:
        marker_id = self._coerce_marker(marker)
        return SegmentSpec.marker(self, marker_id)

    def patch(self, patch: P | str) -> PatchHandle[P]:
        patch_id = self._coerce_patch(patch)
        return SegmentSpec.patch(self, patch_id)

    def segment_target(self) -> SegmentTarget:
        return SegmentTarget(subject=self.subject, segment=self.segment)

    def marker_target(self, marker: M | str) -> MarkerTarget[M]:
        return MarkerTarget(subject=self.subject, handle=self.marker(marker))

    def patch_target(self, patch: P | str) -> PatchTarget[P]:
        return PatchTarget(subject=self.subject, handle=self.patch(patch))

    def patch_label(self, patch: P | str) -> str:
        patch_id = self._coerce_patch(patch)
        return self.patch_declaration(patch_id).label

    def patch_frame(self, patch: P | str) -> str | None:
        patch_id = self._coerce_patch(patch)
        return self.patch_declaration(patch_id).frame

    def _coerce_marker(self, marker: M | str) -> M:
        if isinstance(marker, self.marker_type):
            return marker
        if isinstance(marker, str):
            try:
                return self.marker_type(marker)
            except ValueError as exc:
                raise KeyError(
                    f"Segment {self.segment!r} has no marker {marker!r}"
                ) from exc
        if isinstance(marker, NameId):
            try:
                return self.marker_type(marker)
            except ValueError as exc:
                raise KeyError(
                    f"Segment {self.segment!r} has no marker {marker!r}"
                ) from exc
        raise TypeError(
            f"Expected marker of type {self.marker_type.__name__} or str, "
            f"got {type(marker).__name__}"
        )

    def _coerce_patch(self, patch: P | str) -> P:
        if isinstance(patch, self.patch_type):
            return patch
        if isinstance(patch, str):
            try:
                return self.patch_type(patch)
            except ValueError as exc:
                raise KeyError(
                    f"Segment {self.segment!r} has no patch {patch!r}"
                ) from exc
        if isinstance(patch, NameId):
            try:
                return self.patch_type(patch)
            except ValueError as exc:
                raise KeyError(
                    f"Segment {self.segment!r} has no patch {patch!r}"
                ) from exc
        raise TypeError(
            f"Expected patch of type {self.patch_type.__name__} or str, "
            f"got {type(patch).__name__}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledSubjectSpec(SubjectSpec):
    """SubjectSpec compiled from an authored subject declaration."""

    segment_type: type[SegmentId]
    segments: Mapping[SegmentId, CompiledSegmentSpec[Any, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", MappingProxyType(dict(self.segments)))

    def iter_segments(self) -> tuple[CompiledSegmentSpec[Any, Any], ...]:
        return tuple(self.segments.values())

    def segment(self, segment: SegmentId | str) -> CompiledSegmentSpec[Any, Any]:
        segment_id = self._coerce_segment(segment)
        try:
            return self.segments[segment_id]
        except KeyError as exc:
            raise KeyError(
                f"Subject {self.subject!r} has no segment {segment!r}"
            ) from exc

    def _coerce_segment(self, segment: SegmentId | str) -> SegmentId:
        if isinstance(segment, self.segment_type):
            return segment
        if isinstance(segment, str):
            try:
                return self.segment_type(segment)
            except ValueError as exc:
                raise KeyError(
                    f"Subject {self.subject!r} has no segment {segment!r}"
                ) from exc
        if isinstance(segment, NameId):
            try:
                return self.segment_type(segment)
            except ValueError as exc:
                raise KeyError(
                    f"Subject {self.subject!r} has no segment {segment!r}"
                ) from exc
        raise TypeError(
            f"Expected segment of type {self.segment_type.__name__} or str, "
            f"got {type(segment).__name__}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledSceneSpec(SceneSpec):
    """SceneSpec compiled from authored subject/segment/marker schemas."""

    subject_type: type[SubjectId]
    subjects: Mapping[SubjectId, CompiledSubjectSpec]
    generated_ids: GeneratedIds

    def __post_init__(self) -> None:
        object.__setattr__(self, "subjects", MappingProxyType(dict(self.subjects)))

    def iter_subjects(self) -> tuple[CompiledSubjectSpec, ...]:
        return tuple(self.subjects.values())

    def subject(self, subject: SubjectId | str) -> CompiledSubjectSpec:
        subject_id = self._coerce_subject(subject)
        try:
            return self.subjects[subject_id]
        except KeyError as exc:
            raise KeyError(f"Scene has no subject {subject!r}") from exc

    def _coerce_subject(self, subject: SubjectId | str) -> SubjectId:
        if isinstance(subject, self.subject_type):
            return subject
        if isinstance(subject, str):
            try:
                return self.subject_type(subject)
            except ValueError as exc:
                raise KeyError(f"Scene has no subject {subject!r}") from exc
        if isinstance(subject, NameId):
            try:
                return self.subject_type(subject)
            except ValueError as exc:
                raise KeyError(f"Scene has no subject {subject!r}") from exc
        raise TypeError(
            f"Expected subject of type {self.subject_type.__name__} or str, "
            f"got {type(subject).__name__}"
        )


def build_scene[SubjectsT: Subjects](
    subjects: SubjectsT,
) -> CompiledSceneSpec:
    """Compile typed authoring declarations into runtime specs."""
    subject_items = tuple(subjects.items())
    subject_type = _runtime_id_type(
        "Subject",
        tuple(name for name, _ in subject_items),
        SubjectId,
    )

    compiled_subjects: dict[SubjectId, CompiledSubjectSpec] = {}
    segment_types: dict[SubjectId, type[SegmentId]] = {}
    marker_types: dict[SegmentKey, type[MarkerId]] = {}
    patch_types: dict[SegmentKey, type[PatchId]] = {}

    for subject_name, subject in subject_items:
        subject_id = subject_type(subject_name)
        segment_items = tuple(subject.segments.items())
        segment_type = _runtime_id_type(
            "Segment",
            tuple(name for name, _ in segment_items),
            SegmentId,
        )
        segment_types[subject_id] = segment_type

        compiled_segments: dict[SegmentId, CompiledSegmentSpec[Any, Any]] = {}
        for segment_name, segment in segment_items:
            segment_id = segment_type(segment_name)
            marker_items = tuple(segment.markers.items())
            patch_items = tuple(segment.patches.items())
            marker_type = _runtime_id_type(
                "Marker",
                tuple(name for name, _ in marker_items),
                MarkerId,
            )
            patch_type = _runtime_id_type(
                "Patch",
                tuple(name for name, _ in patch_items),
                PatchId,
            )
            key = SegmentKey(subject_id, segment_id)
            marker_types[key] = marker_type
            patch_types[key] = patch_type

            marker_defs = {
                marker_type(marker_name): marker_def
                for marker_name, marker_def in marker_items
            }
            patch_defs: dict[PatchId, Patch] = {}
            patch_declarations: dict[PatchId, PatchDeclarationSpec[Any]] = {}
            compiled_patches: dict[PatchId, PatchSpec[Any]] = {}
            for patch_name, patch_def in patch_items:
                patch_id = patch_type(patch_name)
                patch_defs[patch_id] = patch_def
                patch_declarations[patch_id] = PatchDeclarationSpec(
                    patch=patch_id,
                    label=patch_def.label,
                    frame=patch_def.frame,
                )
                if (
                    patch_def.transform_segment_patch is None
                    and patch_def.region is None
                ):
                    continue
                if (
                    patch_def.transform_segment_patch is None
                    or patch_def.region is None
                ):
                    raise ValueError(
                        "Patch geometry must provide both transform_segment_patch "
                        f"and region; subject={subject_id!r}, segment={segment_id!r}, "
                        f"patch={patch_id!r}"
                    )
                compiled_patches[patch_id] = PatchSpec(
                    patch=patch_id,
                    transform_segment_patch=patch_def.transform_segment_patch,
                    region=patch_def.region,
                    label=patch_def.label,
                    frame=patch_def.frame,
                )
            compiled_segments[segment_id] = CompiledSegmentSpec(
                subject=subject_id,
                segment=segment_id,
                marker_type=marker_type,
                patch_type=patch_type,
                axis_convention=Z_UP_AXES,
                marker_set=MarkerSetSpec(marker_type=marker_type),
                marker_positions_segment={},
                patch_calibrations={},
                patch_declarations=patch_declarations,
                patches=compiled_patches,
                marker_defs=marker_defs,
                patch_defs=patch_defs,
            )

        compiled_subjects[subject_id] = CompiledSubjectSpec(
            subject=subject_id,
            segment_type=segment_type,
            segments=compiled_segments,
        )

    generated_ids = GeneratedIds(
        subjects=subject_type,
        segments=MappingProxyType(segment_types),
        markers=MappingProxyType(marker_types),
        patches=MappingProxyType(patch_types),
    )
    return CompiledSceneSpec(
        subject_type=subject_type,
        subjects=compiled_subjects,
        generated_ids=generated_ids,
    )


def marker_external_name(segment: SegmentSpec[Any, Any], marker: MarkerId) -> str:
    """Return the external marker label for a runtime marker ID."""
    resolver = getattr(segment, "marker_external_name", None)
    if callable(resolver):
        return resolver(marker)
    resolver = getattr(segment, "marker_vicon_name", None)
    if callable(resolver):
        return resolver(marker)
    return marker.label


def marker_from_external_name(
    segment: SegmentSpec[Any, Any],
    marker_name: str,
) -> MarkerId:
    """Resolve an external marker label back to the runtime marker ID."""
    resolver = getattr(segment, "marker_from_external_name", None)
    if callable(resolver):
        return resolver(marker_name)
    resolver = getattr(segment, "marker_from_vicon_name", None)
    if callable(resolver):
        return resolver(marker_name)
    return segment.marker_type(marker_name)


def marker_from_vicon_name(segment: SegmentSpec[Any, Any], marker_name: str) -> MarkerId:
    """Backward-compatible alias for marker_from_external_name()."""
    return marker_from_external_name(segment, marker_name)


def _runtime_id_type[IdT: NameId](
    kind: str,
    authored_keys: tuple[str, ...],
    base: type[IdT],
) -> type[IdT]:
    member_map: dict[str, str] = {}
    for authored_key in authored_keys:
        member_name = _sanitize_identifier(authored_key)
        if member_name in member_map:
            previous = member_map[member_name]
            if previous != authored_key:
                raise ValueError(
                    f"{kind} keys collide after sanitization: "
                    f"{previous!r} and {authored_key!r} both map to {member_name!r}"
                )
            raise ValueError(
                f"Duplicate {kind.lower()} key after sanitization: {authored_key!r}"
            )
        member_map[member_name] = authored_key

    runtime_enum = Enum(
        f"Generated{kind}Id",
        member_map,
        type=base,
        module=__name__,
    )
    return cast(type[IdT], runtime_enum)


def _sanitize_identifier(name: str) -> str:
    candidate = _NON_IDENTIFIER.sub("_", name)
    if not candidate:
        candidate = "_"
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    if keyword.iskeyword(candidate):
        candidate = f"{candidate}_"
    if not candidate.isidentifier():
        raise ValueError(f"Cannot sanitize {name!r} into a valid identifier")
    return candidate
