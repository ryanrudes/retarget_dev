"""Patch schema declaration and bound time-series queries.

A patch is an oriented contact surface + a bounded footprint, authored as a ``geometry``
callable that, given the segment's :class:`~retarget.core.geometry.SegmentGeometry`, returns a
fungeom :class:`~fungeom.Face` (an oriented ``Plane`` + a ``Region2``). At bind time the
binding evaluates the callable and lowers the Face to a segment-local rigid frame + boundary;
the query methods below transport those per-frame by the segment pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from retarget.core.formats import JumpDetector, finite_difference_velocity
from retarget.core.geometry import face_signal, sampling_at
from retarget.core.support_resolve import ResolveFn, SupportResolver, as_resolver, most_confident
from retarget.core.targets import PatchTarget
from retarget.core.types import (
    FloatArray1D,
    LabelArray,
    TimeBool,
    TimeEntityVec3,
    TimeMat3,
    TimeVec3,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fungeom import Face, FaceSignal, Sampling

    from retarget.core.geometry import SegmentGeometry
    from retarget.core.schema.segment import _SegmentRuntime


class Patches(TypedDict):
    """Base class for typed patch schema declarations."""


@dataclass(slots=True, eq=False)
class _PatchBinding:
    subject: str
    segment: str
    patch: str
    runtime: _SegmentRuntime | None
    # the bind-time segment-local fungeom Face (the patch geometry); transported per-frame
    # as a FaceSignal by the segment pose at query time.
    face: Face | None = None


@dataclass(frozen=True, slots=True)
class Patch:
    """A contact patch: an authored ``geometry`` callable plus bound time-series queries.

    ``geometry`` takes the segment's :class:`~retarget.core.geometry.SegmentGeometry` and
    returns a fungeom :class:`~fungeom.Face`::

        def sole(seg):
            return Face.on(seg.markers["a", "b", "c"].fit_plane(), Region2.hull(...))

        Patch(label="sole", geometry=sole)

    A declaration-only patch (no ``geometry``) is targetable but has no contact geometry.
    """

    label: str
    frame: str | None = None
    geometry: Callable[[SegmentGeometry], Face] | None = field(default=None, compare=False, repr=False)
    _binding: _PatchBinding | None = field(default=None, init=False, compare=False, repr=False)

    def points(self) -> TimeVec3:
        """World-frame patch origin/contact points with shape ``(T, 3)``.

        The patch frame origin (the footprint centroid) transported by the segment pose, read
        off the patch's :class:`~fungeom.FaceSignal` at the track timestamps.
        """
        runtime = self._runtime()
        if len(runtime.timestamps) == 0:
            return cast(TimeVec3, np.empty((0, 3), dtype=np.float64))
        frames = np.asarray(self._face_signal().frame().resolve_over(self._sampling()), dtype=np.float64)
        return cast(TimeVec3, frames[:, :3, 3])

    def normals(self) -> TimeVec3:
        """World-frame patch normals with shape ``(T, 3)``.

        The third column of the oriented patch frame (see :meth:`frames`), so it always agrees
        with :meth:`frames` and resolves vectorized off the same :class:`~fungeom.FaceSignal`.
        """
        runtime = self._runtime()
        if len(runtime.timestamps) == 0:
            return cast(TimeVec3, np.empty((0, 3), dtype=np.float64))
        frames = np.asarray(self._face_signal().frame().resolve_over(self._sampling()), dtype=np.float64)
        return cast(TimeVec3, frames[:, :3, 2])

    def frames(self) -> TimeMat3:
        """World-frame patch orientation with shape ``(T, 3, 3)``.

        Columns are the patch local +X/+Y/+Z axes expressed in the world frame;
        the third column matches :meth:`normals`. Together with :meth:`points`
        this is the full oriented patch frame over time.
        """
        runtime = self._runtime()
        if len(runtime.timestamps) == 0:
            return cast(TimeMat3, np.empty((0, 3, 3), dtype=np.float64))
        frames = np.asarray(self._face_signal().frame().resolve_over(self._sampling()), dtype=np.float64)
        return cast(TimeMat3, frames[:, :3, :3])

    def boundary_points(self) -> TimeEntityVec3:
        """World-frame footprint boundary polygon with shape ``(T, K, 3)``.

        The Face region's vertices, transported into the world frame at every timestep (via the
        patch :class:`~fungeom.FaceSignal`), ready to draw as an oriented polygon.
        """
        binding = self._require_binding()
        if binding.face is None:
            raise ValueError(f"Patch {binding.patch!r} has no contact region to take a boundary from")
        runtime = self._runtime()
        if len(runtime.timestamps) == 0:
            num_vertices = len(binding.face.boundary().resolve().roster)
            return cast(TimeEntityVec3, np.empty((0, num_vertices, 3), dtype=np.float64))
        vertices, _present = self._face_signal().boundary().resolve_over(self._sampling())
        return cast(TimeEntityVec3, np.asarray(vertices, dtype=np.float64))

    def velocities(self) -> TimeVec3:
        """World-frame patch-point velocities with shape ``(T, 3)``."""
        runtime = self._runtime()
        return cast(TimeVec3, finite_difference_velocity(self.points(), runtime.timestamps))

    def contacts(self) -> np.ndarray:
        """Boolean contact state with shape ``(T,)`` from an attached contact track."""
        runtime = self._runtime()
        name = self._require_binding().patch
        try:
            return runtime.contacts[name]
        except KeyError as exc:
            raise ValueError(f"No contact track is attached for patch {name!r}") from exc

    def confidence(self) -> np.ndarray:
        """Contact confidence with shape ``(T,)`` from an attached contact track."""
        runtime = self._runtime()
        name = self._require_binding().patch
        try:
            return runtime.confidences[name]
        except KeyError as exc:
            raise ValueError(f"No contact confidence is attached for patch {name!r}") from exc

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
        resolver = most_confident(none_label=none) if resolve is None else as_resolver(resolve)
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
        """Whether this patch was authored with a ``geometry`` callable."""
        return self.geometry is not None

    def has_region(self) -> bool:
        """Whether this patch has a footprint boundary (for region-aware contact sampling)."""
        return self._binding is not None and self._binding.face is not None

    def face(self) -> Face:
        """The bound segment-local fungeom ``Face`` for this patch.

        The oriented surface (``face.plane()``) + bounded footprint (``face.region()``) in the
        segment frame. Raises if this patch was not authored with a ``geometry`` callable.
        """
        binding = self._require_binding()
        if binding.face is None:
            raise ValueError(f"Patch {binding.patch!r} was not authored with a geometry callable")
        return binding.face

    def _face_signal(self) -> FaceSignal:
        binding = self._require_binding()
        if binding.face is None:
            raise ValueError(f"Patch {binding.patch!r} is declared but has no calibrated geometry")
        runtime = self._runtime()
        return face_signal(binding.face, runtime.timestamps, runtime.translations, runtime.rotations)

    def _sampling(self) -> Sampling:
        return sampling_at(self._runtime().timestamps)

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
