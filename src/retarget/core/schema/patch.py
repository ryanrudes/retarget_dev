"""Patch schema declaration and bound time-series queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.calibration import calibrate_patch_transform
from retarget.core.contact_region import ContactRegion, RectangularRegion
from retarget.core.formats import JumpDetector, finite_difference_velocity
from retarget.core.support_resolve import ResolveFn, SupportResolver, as_resolver, priority
from retarget.core.targets import PatchTarget
from retarget.core.transform import RigidTransform
from retarget.core.translation import MarkerTranslation
from retarget.core.types import (
    FloatArray1D,
    LabelArray,
    TimeBool,
    TimeEntityVec3,
    TimeMat3,
    TimeVec3,
    Vec3,
)

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
class Patch[RegionT: ContactRegion | None = ContactRegion | None]:
    """A contact patch: authoring metadata plus bound time-series queries.

    ``RegionT`` is the static type of ``region``. Authoring a patch without a
    region (declaration-only) leaves it as ``ContactRegion | None``; constructing
    via :meth:`rectangle` (or passing a concrete ``region=...``) narrows it, so a
    schema declaring ``sole: Patch[RectangularRegion]`` types ``sole.region`` as
    ``RectangularRegion`` with no ``isinstance`` check.
    """

    label: str
    transform_segment_patch: RigidTransform | None = field(default=None, compare=False)
    region: RegionT = field(default=cast("RegionT", None), compare=False)
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
    ) -> Patch[RectangularRegion]:
        """Build a rectangular patch whose frame is fit from calibration markers.

        ``markers`` names markers on the same segment; their segment-frame
        positions (from the subject ``body_model`` or per-marker
        ``position_segment``) are fit into a patch frame at bind time.
        ``outward_axis`` fixes the normal direction and ``forward_axis`` fixes
        the in-plane local +X direction. For the rarer case where a frame is
        already known, construct ``Patch(..., transform_segment_patch=...,
        region=RectangularRegion(...))`` directly.
        """
        return Patch(
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

    def frames(self) -> TimeMat3:
        """World-frame patch orientation with shape ``(T, 3, 3)``.

        Columns are the patch local +X/+Y/+Z axes expressed in the world frame;
        the third column matches :meth:`normals`. Together with :meth:`points`
        this is the full oriented patch frame over time.
        """
        runtime = self._runtime()
        local_rotation = np.asarray(self._geometry().rotation, dtype=np.float64)
        return cast(
            TimeMat3, np.einsum("tij,jk->tik", runtime.rotations, local_rotation)
        )

    def boundary_points(self) -> TimeEntityVec3:
        """World-frame contact-region boundary polygon with shape ``(T, K, 3)``.

        The region boundary (e.g. the four corners of a ``RectangularRegion``) is
        expressed in the patch local xy plane, then transported into the world
        frame at every timestep, ready to draw as an oriented polygon.
        """
        region = self.region
        if region is None:
            name = self._binding.patch if self._binding is not None else self.label
            raise ValueError(
                f"Patch {name!r} has no contact region to take a boundary from"
            )
        local_xy = np.asarray(region.boundary(), dtype=np.float64)
        local_xyz = np.concatenate(
            [local_xy, np.zeros((local_xy.shape[0], 1), dtype=np.float64)], axis=1
        )
        geometry = self._geometry()
        rotation = np.asarray(geometry.rotation, dtype=np.float64)
        translation = np.asarray(geometry.translation, dtype=np.float64)
        boundary_segment = local_xyz @ rotation.T + translation
        runtime = self._runtime()
        world = np.einsum("tij,kj->tki", runtime.rotations, boundary_segment)
        return cast(TimeEntityVec3, world + runtime.translations[:, None, :])

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

    def support_contacts(self) -> dict[str, TimeBool]:
        """Per-named-support boolean contact arrays for this patch (multi-contact).

        Populated from an attached :class:`SupportStateTrack`; empty if none is
        attached. Several supports may be True at the same time.
        """
        runtime = self._runtime()
        name = self._require_binding().patch
        return cast("dict[str, TimeBool]", dict(runtime.support_contacts.get(name, {})))

    def support_scores(self) -> dict[str, FloatArray1D]:
        """Per-named-support contact scores for this patch (from an attached track)."""
        runtime = self._runtime()
        name = self._require_binding().patch
        return cast("dict[str, FloatArray1D]", dict(runtime.support_scores.get(name, {})))

    def support_state(
        self,
        *,
        resolve: SupportResolver | ResolveFn | None = None,
        none: str | None = None,
        unknown: str | None = None,
    ) -> LabelArray:
        """Per-frame categorical support label for this patch with shape ``(T,)``.

        Labels are yours: the values are the support names you passed to
        ``classify`` (``None`` where nothing is in contact, unless you pass a
        ``none`` label). Pass ``unknown="..."`` to surface frames the detector
        flagged as untrustworthy under that label (validity is computed for you);
        omit it and those frames just read as ``none``. ``resolve`` overrides the
        default :func:`~retarget.core.support_resolve.priority` reduction.
        """
        runtime = self._runtime()
        name = self._require_binding().patch
        contacts = self.support_contacts()
        scores = self.support_scores()
        if not contacts:
            return cast(LabelArray, np.full(len(runtime.timestamps), none, dtype=object))
        resolver = priority(none_label=none) if resolve is None else as_resolver(resolve)
        labels = np.asarray(resolver(contacts, scores), dtype=object)
        if unknown is not None:
            invalid = runtime.support_invalid.get(name)
            if invalid is not None:
                labels = labels.copy()
                labels[np.asarray(invalid, dtype=np.bool_)] = unknown
        return cast(LabelArray, labels)

    def valid(self, *, min_coverage: float = 0.5, jumps: JumpDetector | None = None) -> TimeBool:
        """Per-frame mask of trustworthy segment pose, shape ``(T,)``.

        Trustworthy where marker coverage is at least ``min_coverage`` (or the
        pose was filled) *and*, when a ``jumps`` detector is given, the patch
        point is not a garbage sample. Jumps are measured on the patch point, so
        a rotation glitch is caught even though the origin barely moves.
        """
        coverage = np.asarray(self.pose_coverage())
        enough = coverage >= min_coverage
        if jumps is None:
            jump = np.zeros(len(coverage), dtype=np.bool_)
        else:
            runtime = self._runtime()
            positions = self.points() if self.has_geometry() else runtime.translations
            jump = np.asarray(jumps(positions, runtime.timestamps))
        return cast(TimeBool, enough & ~jump)

    def pose_filled(self) -> TimeBool:
        """Per-frame mask of poses synthesized by :func:`~retarget.demo.fill_pose_gaps`.

        All False when no gaps were filled. Use it to render or exclude
        interpolated (non-measured) frames.
        """
        from retarget.core.schema.segment import _pose_filled_mask

        runtime = self._runtime()
        return cast(TimeBool, _pose_filled_mask(runtime, len(runtime.timestamps)))

    def pose_coverage(self) -> FloatArray1D:
        """Fraction of the segment's markers observed per frame, shape ``(T,)``.

        Frames whose pose was synthesized by :func:`~retarget.demo.fill_pose_gaps`
        count as fully covered.
        """
        from retarget.core.schema.segment import _pose_filled_mask, _visible_marker_count

        runtime = self._runtime()
        num = len(runtime.timestamps)
        filled = _pose_filled_mask(runtime, num)
        if not runtime.observed_markers:
            return cast(FloatArray1D, np.ones(num, dtype=np.float64))
        coverage = _visible_marker_count(runtime.observed_markers, num).astype(np.float64) / len(
            runtime.observed_markers
        )
        return cast(FloatArray1D, np.where(filled, 1.0, coverage))

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
