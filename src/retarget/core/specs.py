from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from retarget.core.enums import MarkerId, MarkerRole, PatchId, SegmentId, SubjectId
from retarget.core.transform import RigidTransform
from retarget.core.contact_region import ContactRegion
from retarget.core.types import Vec3, Points3
from retarget.core.axes import SemanticAxis, AxisConvention
from retarget.core.translation import MarkerTranslation
from retarget.utils.geometry import fit_patch_frame
from retarget.core.handles import MarkerHandle, PatchHandle


@dataclass(frozen=True, slots=True)
class MarkerSpec[M: MarkerId]:
    """Specification for one marker in a marker vocabulary."""

    marker: M
    """The marker identifier."""

    role: MarkerRole = MarkerRole.TRACKING
    """The role of the marker in the tracking system."""

    @property
    def label(self) -> str:
        """The label of the marker in the vocabulary."""
        return self.marker.label

    @property
    def index(self) -> int:
        """The index of the marker in the vocabulary."""
        return self.marker.index


@dataclass(frozen=True, slots=True)
class MarkerSetSpec[M: MarkerId]:
    """
    Specification for all markers attached to a segment.

    The marker vocabulary defines which markers exist.
    This spec defines how those markers are used.
    """

    marker_type: type[M]
    """The type of the marker."""

    roles: Mapping[M, MarkerRole] = field(default_factory=dict)
    """The roles of the markers in the tracking system."""

    default_role: MarkerRole = MarkerRole.TRACKING
    """The default role of the markers in the tracking system."""

    def __post_init__(self) -> None:
        roles = dict(self.roles)

        for marker in roles:
            self._check_marker(marker)

        object.__setattr__(self, "roles", MappingProxyType(roles))

    def role(self, marker: M) -> MarkerRole:
        self._check_marker(marker)
        return self.roles.get(marker, self.default_role)

    def spec(self, marker: M) -> MarkerSpec[M]:
        return MarkerSpec(
            marker=marker,
            role=self.role(marker),
        )

    def tracking_markers(self) -> tuple[M, ...]:
        return tuple(
            marker
            for marker in self.marker_type
            if self.role(marker)
            in {
                MarkerRole.TRACKING,
                MarkerRole.TRACKING_AND_CALIBRATION,
            }
        )

    def calibration_markers(self) -> tuple[M, ...]:
        return tuple(
            marker
            for marker in self.marker_type
            if self.role(marker)
            in {
                MarkerRole.CALIBRATION,
                MarkerRole.TRACKING_AND_CALIBRATION,
            }
        )

    def _check_marker(self, marker: M) -> None:
        if not isinstance(marker, self.marker_type):
            raise TypeError(
                f"Expected marker of type {self.marker_type.__name__}, "
                f"got {type(marker).__name__}"
            )


@dataclass(frozen=True, slots=True)
class PatchSpec[P: PatchId]:
    """
    Persistent segment-local contact patch definition.

    transform_segment_patch maps patch-frame points into segment-frame points.
    """

    patch: P
    """The patch identifier."""

    transform_segment_patch: RigidTransform
    """The transform from the patch frame to the segment frame."""

    region: ContactRegion
    """The contact region of the patch."""


@dataclass(frozen=True, slots=True)
class PatchCalibrationSpec[M: MarkerId, P: PatchId]:
    """
    Calibration recipe for building a contact patch from segment-local markers.

    markers defines which markers are used to fit the patch plane.
    marker_translations optionally maps markers to marker-specific offsets
    applied before plane fitting. Markers omitted from marker_translations
    use zero offset.
    normal_offset is applied after fitting, along the fitted patch normal.
    Positive values move along the patch normal; negative values move opposite it.
    """

    patch: P
    markers: tuple[M, ...]
    region: ContactRegion
    marker_translations: Mapping[M, MarkerTranslation] = field(default_factory=dict)
    normal_offset: float = 0.0
    outward_axis: SemanticAxis = SemanticAxis.UP
    x_axis: SemanticAxis = SemanticAxis.FORWARD

    def __post_init__(self) -> None:
        markers = tuple(self.markers)
        marker_translations = dict(self.marker_translations)
        if len(markers) < 3:
            raise ValueError("Patch calibration requires at least three markers")
        duplicate_markers = {
            marker
            for marker in markers
            if markers.count(marker) > 1
        }
        if duplicate_markers:
            raise ValueError(
                "Patch calibration markers must be unique; "
                f"got duplicates: {duplicate_markers}"
            )
        unknown_translation_markers = set(marker_translations) - set(markers)
        if unknown_translation_markers:
            raise ValueError(
                "marker_translations contains markers that are not listed in "
                f"markers: {unknown_translation_markers}"
            )
        object.__setattr__(self, "markers", markers)
        object.__setattr__(
            self,
            "marker_translations",
            MappingProxyType(marker_translations),
        )

    def surface_points(
        self,
        marker_positions_segment: Mapping[M, Vec3],
        segment: SegmentSpec[M, P],
    ) -> Points3:
        points: list[Vec3] = []
        for marker in self.markers:
            marker_position = marker_positions_segment[marker]
            translation = self.marker_translations.get(marker)
            if translation is None:
                points.append(marker_position)
            else:
                points.append(
                    cast(Vec3, marker_position + translation.resolve(segment))
                )
        return cast(Points3, np.stack(points))

    def build_patch(
        self,
        marker_positions_segment: Mapping[M, Vec3],
        segment: SegmentSpec[M, P],
    ) -> PatchSpec[P]:
        surface_points_segment = self.surface_points(
            marker_positions_segment=marker_positions_segment,
            segment=segment,
        )

        transform_segment_patch = fit_patch_frame(
            surface_points_segment=surface_points_segment,
            outward_hint_segment=segment.axis(self.outward_axis),
            x_axis_hint_segment=segment.axis(self.x_axis),
        )

        if self.normal_offset != 0.0:
            normal_segment = transform_segment_patch.rotation[:, 2]
            offset_segment = self.normal_offset * normal_segment
            transform_segment_patch = RigidTransform.from_rotation_translation(
                rotation=transform_segment_patch.rotation,
                translation=cast(
                    Vec3,
                    transform_segment_patch.translation + offset_segment,
                ),
            )

        return PatchSpec(
            patch=self.patch,
            transform_segment_patch=transform_segment_patch,
            region=self.region,
        )


@dataclass(frozen=True, slots=True)
class SegmentSpec[M: MarkerId, P: PatchId]:
    """
    Strongly typed persistent specification for one rigid segment.

    M is the marker vocabulary for this segment.
    P is the patch vocabulary for this segment.

    SegmentSpec is segment-local. It should not contain world-frame pose,
    timestep-dependent state, or observations.
    """

    segment: SegmentId
    """The symbolic identifier for this segment."""

    marker_type: type[M]
    """The marker vocabulary used by this segment."""

    patch_type: type[P]
    """The patch vocabulary used by this segment."""

    axis_convention: AxisConvention
    """Semantic-to-coordinate axis convention for this segment."""

    marker_set: MarkerSetSpec[M]
    """Marker roles and marker vocabulary information."""

    marker_positions_segment: Mapping[M, Vec3] = field(default_factory=dict)
    """Segment-frame positions of markers."""

    patch_calibrations: Mapping[P, PatchCalibrationSpec[M, P]] = field(default_factory=dict)
    """Patch calibrations keyed by patch ID."""

    patches: Mapping[P, PatchSpec[P]] = field(default_factory=dict)
    """Built/calibrated patch specs keyed by patch ID."""

    def __post_init__(self) -> None:
        if self.marker_set.marker_type is not self.marker_type:
            raise TypeError(
                "marker_set.marker_type must match segment marker_type"
            )

        marker_positions_segment = dict(self.marker_positions_segment)
        patch_calibrations = dict(self.patch_calibrations)
        patches = dict(self.patches)

        for marker in marker_positions_segment:
            self._check_marker(marker)

        for patch in patch_calibrations:
            self._check_patch(patch)

        for patch in patches:
            self._check_patch(patch)

        for calibration in patch_calibrations.values():
            for marker in calibration.markers:
                self._check_marker(marker)
                if marker not in marker_positions_segment:
                    raise ValueError(
                        f"Missing segment-frame position for calibration marker {marker}"
                    )
            for marker in calibration.marker_translations:
                self._check_marker(marker)

        object.__setattr__(
            self,
            "marker_positions_segment",
            MappingProxyType(marker_positions_segment),
        )
        object.__setattr__(
            self,
            "patch_calibrations",
            MappingProxyType(patch_calibrations),
        )
        object.__setattr__(
            self,
            "patches",
            MappingProxyType(patches),
        )

    def axis(self, axis: SemanticAxis) -> Vec3:
        """Resolve a semantic axis into a concrete segment-frame vector."""
        return self.axis_convention.vector(axis)

    def marker(self, marker: M) -> MarkerHandle[M]:
        """Return a typed handle to a marker on this segment."""
        self._check_marker(marker)

        return MarkerHandle(
            segment=self.segment,
            marker=marker,
        )

    def patch(self, patch: P) -> PatchHandle[P]:
        """Return a typed handle to a patch on this segment."""
        self._check_patch(patch)

        return PatchHandle(
            segment=self.segment,
            patch=patch,
        )

    def marker_position(self, marker: M) -> Vec3:
        """Return the segment-frame position of a marker."""
        self._check_marker(marker)
        return self.marker_positions_segment[marker]

    def patch_spec(self, patch: P) -> PatchSpec[P]:
        """Return the segment-local patch spec."""
        self._check_patch(patch)
        return self.patches[patch]

    def calibration(self, patch: P) -> PatchCalibrationSpec[M, P]:
        """Return the calibration spec for a patch."""
        self._check_patch(patch)
        return self.patch_calibrations[patch]

    def build_patch(self, patch: P) -> PatchSpec[P]:
        """Build one patch from this segment's marker positions and calibration."""
        self._check_patch(patch)

        calibration = self.patch_calibrations[patch]

        return calibration.build_patch(
            marker_positions_segment=self.marker_positions_segment,
            segment=self,
        )

    def with_patch(self, patch: PatchSpec[P]) -> SegmentSpec[M, P]:
        """Return a copy of this segment spec with one patch added/replaced."""
        self._check_patch(patch.patch)

        patches = dict(self.patches)
        patches[patch.patch] = patch

        return SegmentSpec(
            segment=self.segment,
            marker_type=self.marker_type,
            patch_type=self.patch_type,
            axis_convention=self.axis_convention,
            marker_set=self.marker_set,
            marker_positions_segment=self.marker_positions_segment,
            patch_calibrations=self.patch_calibrations,
            patches=patches,
        )

    def with_built_patch(self, patch: P) -> SegmentSpec[M, P]:
        """Return a copy of this segment spec with one calibrated patch built."""
        return self.with_patch(self.build_patch(patch))

    def with_built_patches(self) -> SegmentSpec[M, P]:
        """Build all calibrated patches and return a copy with patches populated."""
        spec: SegmentSpec[M, P] = self

        for patch in self.patch_calibrations:
            spec = spec.with_built_patch(patch)

        return spec

    def _check_marker(self, marker: M) -> None:
        if not isinstance(marker, self.marker_type):
            raise TypeError(
                f"Expected marker of type {self.marker_type.__name__}, "
                f"got {type(marker).__name__}"
            )

    def _check_patch(self, patch: P) -> None:
        if not isinstance(patch, self.patch_type):
            raise TypeError(
                f"Expected patch of type {self.patch_type.__name__}, "
                f"got {type(patch).__name__}"
            )


@dataclass(frozen=True, slots=True)
class SubjectSpec(ABC):
    """
    Base class for subject specs.

    Concrete subclasses should expose typed segment fields, such as:

        left_foot: SegmentSpec[LeftFootMarkerId, FootPatchId]

    Generic code can use iter_segments().

    This is intentionally tiny. The base class exists for traversal, not for perfect typing.
    """

    subject: SubjectId
    """The symbolic subject identifier."""

    @abstractmethod
    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        """Iterate over this subject's segment specs."""
        raise NotImplementedError

    def segment(self, segment: SegmentId) -> SegmentSpec[Any, Any]:
        """Return the segment spec with the given subject-local segment ID."""
        for segment_spec in self.iter_segments():
            if segment_spec.segment == segment:
                return segment_spec
        raise KeyError(
            f"Subject {self.subject!r} has no segment {segment!r}"
        )


@dataclass(frozen=True, slots=True)
class SceneSpec(ABC):
    """
    Base class for scene specs.

    Concrete subclasses should expose typed subject fields.
    """

    @abstractmethod
    def iter_subjects(self) -> Iterable[SubjectSpec]:
        """Iterate over subject specs in this scene."""
        raise NotImplementedError

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        """Iterate over all segments in all subjects."""
        for subject in self.iter_subjects():
            yield from subject.iter_segments()

    def subject(self, subject: SubjectId) -> SubjectSpec:
        """Return the subject spec with the given subject ID."""
        for subject_spec in self.iter_subjects():
            if subject_spec.subject == subject:
                return subject_spec
        raise KeyError(f"Scene has no subject {subject!r}")