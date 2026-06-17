"""Patch schema declaration and bound time-series queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.calibration import calibrate_patch_transform
from retarget.core.contact_region import ContactRegion, RectangularRegion
from retarget.core.formats import finite_difference_velocity
from retarget.core.targets import PatchTarget
from retarget.core.transform import RigidTransform
from retarget.core.translation import MarkerTranslation
from retarget.core.types import TimeVec3, Vec3

if TYPE_CHECKING:
    from retarget.core.schema.segment import _SegmentRuntime


class Patches(TypedDict):
    """Base class for typed patch schema declarations."""


@dataclass(slots=True, eq=False)
class _PatchBinding:
    subject: str
    segment: str
    patch: str
    runtime: _SegmentRuntime | None


@dataclass(frozen=True, slots=True)
class PatchCalibration:
    """A deferred patch-frame fit from a segment's calibration markers.

    The ``transform_segment_patch`` is computed at bind time from the named
    markers' segment-frame positions (which may come from the subject
    ``body_model`` or per-marker ``position_segment``), so the patch frame need
    not be precomputed during authoring.
    """

    markers: tuple[str, ...]
    outward_axis: SemanticAxis
    forward_axis: SemanticAxis
    normal_offset: float = 0.0
    axis_convention: AxisConvention = Z_UP_AXES
    marker_translations: Mapping[str, MarkerTranslation] | None = field(
        default=None, compare=False
    )


@dataclass(frozen=True, slots=True)
class Patch:
    """A contact patch: authoring metadata plus bound time-series queries."""

    label: str
    transform_segment_patch: RigidTransform | None = field(default=None, compare=False)
    region: ContactRegion | None = field(default=None, compare=False)
    frame: str | None = None
    calibration: PatchCalibration | None = field(default=None, compare=False)
    _binding: _PatchBinding | None = field(
        default=None, init=False, compare=False, repr=False
    )

    @classmethod
    def rectangle(
        cls,
        label: str,
        *,
        markers: Sequence[str],
        width: float,
        height: float,
        outward_axis: SemanticAxis,
        forward_axis: SemanticAxis,
        normal_offset: float = 0.0,
        axis_convention: AxisConvention = Z_UP_AXES,
        marker_translations: Mapping[str, MarkerTranslation] | None = None,
        frame: str | None = None,
    ) -> Patch:
        """Build a rectangular patch whose frame is fit from calibration markers.

        ``markers`` names markers on the same segment; their segment-frame
        positions (from the subject ``body_model`` or per-marker
        ``position_segment``) are fit into a patch frame at bind time.
        ``outward_axis`` fixes the normal direction and ``forward_axis`` fixes
        the in-plane local +X direction. For the rarer case where a frame is
        already known, construct ``Patch(..., transform_segment_patch=...,
        region=RectangularRegion(...))`` directly.
        """
        return cls(
            label=label,
            region=RectangularRegion(width=width, height=height),
            frame=frame,
            calibration=PatchCalibration(
                markers=tuple(markers),
                outward_axis=outward_axis,
                forward_axis=forward_axis,
                normal_offset=normal_offset,
                axis_convention=axis_convention,
                marker_translations=marker_translations,
            ),
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
            from retarget.core.schema.segment import _UNBOUND_MESSAGE

            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Patch"))
        return self._binding

    def _runtime(self) -> _SegmentRuntime:
        from retarget.core.schema.segment import _UNBOUND_MESSAGE

        binding = self._require_binding()
        if binding.runtime is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Patch"))
        return binding.runtime


def resolve_patch_calibration(
    calibration: PatchCalibration,
    marker_positions_segment: Mapping[str, Vec3],
    *,
    subject: str,
    segment: str,
    patch_name: str,
) -> RigidTransform:
    missing = [m for m in calibration.markers if m not in marker_positions_segment]
    if missing:
        raise ValueError(
            f"Patch {patch_name!r} on {subject!r}/{segment!r} cannot calibrate: "
            f"no segment-frame positions for markers {missing}. Provide a subject "
            "body_model or a per-marker position_segment for them."
        )
    return calibrate_patch_transform(
        marker_positions_segment=marker_positions_segment,
        markers=calibration.markers,
        axis_convention=calibration.axis_convention,
        marker_translations=calibration.marker_translations,
        normal_offset=calibration.normal_offset,
        outward_axis=calibration.outward_axis,
        forward_axis=calibration.forward_axis,
    )
