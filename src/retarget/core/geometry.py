"""The bind-time geometry view a patch callable receives, and the bridge that turns a
bound patch + segment pose into a *moving* fungeom ``FaceSignal``.

In the substrate model a patch is authored as a callable over the segment's geometry::

    def sole(seg):
        return Face.on(seg.markers["a", "b", "c"].fit_plane(), Region2.hull(...))

``SegmentGeometry`` is the ``seg`` that callable receives. ``seg.markers[...]`` are fungeom
``Point3`` of the segment's marker **rest positions** (``Marker.position_segment``), expressed
in the segment frame and grounded at identity to world — so the static fungeom algebra
(``fit_plane``/``offset``/``centroid``/…) resolves and yields *segment-local* geometry.

At bind time the binding evaluates a patch's geometry callable to a segment-local ``Face`` and
stores it. At query time :func:`face_signal` fixes that ``Face`` in the segment frame and
transports it by the segment pose as a fungeom :class:`~fungeom.FaceSignal`; the patch query
methods materialize ``frame()``/``plane()``/``boundary()`` over the track timestamps with
:func:`sampling_at`. Geometry lives in fungeom — retarget only assembles the pose carrier and
reads the signals back.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from fungeom import FaceSignal, Point3, Point3Bundle, Sampling, TransformSignal

from retarget.core.schema.base import _schema_items
from retarget.core.targets import SegmentTarget

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    from fungeom import Face

    from retarget.core import Segment
    from retarget.core.types import FloatArray, Vec3

__all__ = [
    "SegmentGeometry",
    "MarkerGeometry",
    "segment_geometry",
    "segment_geometry_at",
    "patch_face",
    "face_signal",
    "segment_pose_signal",
    "sampling_at",
]


@dataclass(frozen=True, slots=True, eq=False)
class MarkerGeometry:
    """A segment's marker rest positions as fungeom ``Point3``, keyed by authored name.

    Single name → ``Point3``; a tuple of names → ``Point3Bundle`` (the subset a fit/centroid
    composes over). Only markers carrying a ``position_segment`` are present.
    """

    _points: Mapping[str, Point3]

    @overload
    def __getitem__(self, key: str) -> Point3: ...
    @overload
    def __getitem__(self, key: tuple[str, ...]) -> Point3Bundle: ...
    def __getitem__(self, key: str | tuple[str, ...]) -> Point3 | Point3Bundle:
        if isinstance(key, tuple):
            members = {name: self._point(name) for name in key}
            # str is Hashable; the cast only works around Mapping key-type invariance.
            return Point3Bundle.from_map(cast("Mapping[Hashable, Point3]", members))
        return self._point(key)

    def all(self) -> Point3Bundle:
        """The whole marker cloud (all markers with a rest position) as one bundle."""
        return Point3Bundle.from_map(cast("Mapping[Hashable, Point3]", dict(self._points)))

    def names(self) -> tuple[str, ...]:
        """Authored names of the markers that have a rest position, in declared order."""
        return tuple(self._points)

    def __contains__(self, key: str) -> bool:
        return key in self._points

    def _point(self, name: str) -> Point3:
        try:
            return self._points[name]
        except KeyError:
            available = ", ".join(self._points) or "(none)"
            raise KeyError(f"marker {name!r} has no segment-frame position (available: {available})") from None


@dataclass(frozen=True, slots=True, eq=False)
class SegmentGeometry:
    """The ``seg`` a patch-region callable receives — see the module docstring."""

    markers: MarkerGeometry
    target: SegmentTarget

    @property
    def name(self) -> str:
        return self.target.segment


def _marker_points(positions: Mapping[str, Vec3 | None]) -> dict[str, Point3]:
    points: dict[str, Point3] = {}
    for name, local in positions.items():
        if local is None:
            continue
        coords = np.asarray(local, dtype=np.float64).reshape(3)  # fail loud on non-(3,) input
        points[str(name)] = Point3.at(float(coords[0]), float(coords[1]), float(coords[2]))
    return points


def segment_geometry(segment: Segment[Any, Any]) -> SegmentGeometry:
    """Build a :class:`SegmentGeometry` from a bound retarget ``Segment``.

    Reads each marker's ``position_segment`` (segment-frame rest position, set at bind time
    from authoring or the subject ``body_model``). Markers without one are omitted.
    """
    positions = {name: marker.position_segment for name, marker in _schema_items(segment.markers)}
    return SegmentGeometry(
        markers=MarkerGeometry(MappingProxyType(_marker_points(positions))),
        target=segment.segment_target(),
    )


def segment_geometry_at(target: SegmentTarget, positions: Mapping[str, Vec3]) -> SegmentGeometry:
    """Build a :class:`SegmentGeometry` from segment-frame marker positions + a target.

    The form the binding uses: it already holds ``marker_positions_segment`` and the
    subject/segment names, so it need not reconstruct a bound ``Segment``.
    """
    return SegmentGeometry(
        markers=MarkerGeometry(MappingProxyType(_marker_points(positions))),
        target=target,
    )


def patch_face(
    segment: Segment[Any, Any],
    geometry: Callable[[SegmentGeometry], Face],
) -> Face:
    """Evaluate a patch-region callable over a segment's bind-time geometry → a ``Face``.

    The callable composes fungeom geometry over ``seg.markers[...]`` and returns a
    :class:`~fungeom.Face` (oriented plane + bounded region) in segment-local coordinates —
    the open-algebra replacement for the closed ``calibrate_patch_transform`` strategy menu.
    """
    return geometry(segment_geometry(segment))


def segment_pose_signal(
    timestamps: FloatArray,
    translations: FloatArray,
    rotations: FloatArray,
) -> TransformSignal:
    """The segment pose over time as a fungeom :class:`~fungeom.TransformSignal`.

    Assembles the runtime's ``(T, 3, 3)`` rotations and ``(T, 3)`` translations into a dense
    ``(T, 4, 4)`` stack and wraps it with the vectorized ``TransformSignal.from_matrices``
    carrier, so resolving over the track timestamps stays O(T) in numpy.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    rotation = np.asarray(rotations, dtype=np.float64)
    translation = np.asarray(translations, dtype=np.float64)
    matrices = np.zeros((times.shape[0], 4, 4), dtype=np.float64)
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = translation
    matrices[:, 3, 3] = 1.0
    return TransformSignal.from_matrices(times, matrices)


def face_signal(
    face: Face,
    timestamps: FloatArray,
    translations: FloatArray,
    rotations: FloatArray,
) -> FaceSignal:
    """The patch as a *moving* fungeom :class:`~fungeom.FaceSignal`.

    The bind-time segment-local ``face`` fixed in the segment frame and transported by the
    segment pose over time. ``face_signal(...).frame()/plane()/boundary()/clearance(...)`` are
    the patch's world geometry as signals, materialized via :func:`sampling_at`.
    """
    return FaceSignal.of(face, segment_pose_signal(timestamps, translations, rotations))


def sampling_at(timestamps: FloatArray) -> Sampling:
    """A fungeom :class:`~fungeom.Sampling` at the track timestamps (for ``resolve_over``)."""
    return Sampling.at_times(np.asarray(timestamps, dtype=np.float64))
