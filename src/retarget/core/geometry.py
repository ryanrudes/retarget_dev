"""The bind-time geometry view a patch-region callable receives, and the lowering
of a fungeom ``Face`` to retarget's segment-local rigid frame + boundary.

In the substrate model a patch is authored as a callable over the segment's geometry::

    def sole(seg):
        return Face.on(seg.markers["a", "b", "c"].fit_plane(), Region2.hull(...))

``SegmentGeometry`` is the ``seg`` that callable receives. ``seg.markers[...]`` are fungeom
``Point3`` of the segment's marker **rest positions** (``Marker.position_segment``), expressed
in the segment frame and grounded at identity to world — so the static fungeom algebra
(``fit_plane``/``offset``/``centroid``/…) resolves and yields *segment-local* geometry.

At bind time the binding evaluates a patch's geometry callable and ``lower_face``s the
resulting ``Face`` to a segment-local rigid frame (rotation columns ``[x, y, normal]``,
translation = the region centroid) plus the region boundary as segment-local 3-D points.
The patch query methods then transport those per-frame by the segment pose — exactly as
``transform_segment_patch`` does today.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from fungeom import Direction3, Point2, Point3, Point3Bundle

from retarget.core.targets import SegmentTarget
from retarget.core.transform import RigidTransform

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
    "lower_face",
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
    positions = {name: marker.position_segment for name, marker in segment.markers.items()}
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


def lower_face(face: Face) -> tuple[RigidTransform, FloatArray]:
    """Lower a segment-local fungeom ``Face`` to a retarget rigid frame + boundary polygon.

    Returns ``(transform, boundary)`` where ``transform`` is the segment→patch
    :class:`~retarget.core.transform.RigidTransform` (rotation columns ``[x, y, normal]``,
    translation = the region centroid) and ``boundary`` is the region's vertices as
    segment-local ``(K, 3)`` points on the patch plane. The patch query methods transport
    these per-frame by the segment pose, reproducing the legacy ``transform_segment_patch``
    contract from the open-algebra Face.

    The in-plane +x is the plane's own uv x-axis (deterministic), so the frame is a stable
    orthonormal basis with the third column equal to the plane normal.
    """
    plane = face.plane()
    region = face.region()
    origin = plane.embed(region.centroid())
    base = np.asarray(plane.embed(Point2.at(0.0, 0.0)).resolve().coord, dtype=np.float64)
    along_x = np.asarray(plane.embed(Point2.at(1.0, 0.0)).resolve().coord, dtype=np.float64)
    tangent_vec = along_x - base
    tangent = Direction3.of(float(tangent_vec[0]), float(tangent_vec[1]), float(tangent_vec[2]))
    matrix = np.asarray(plane.frame(origin, tangent).resolve().matrix, dtype=np.float64)
    transform = RigidTransform.from_rotation_translation(rotation=matrix[:3, :3], translation=matrix[:3, 3])
    vertices = region.vertices().resolve()
    boundary = np.array(
        [
            np.asarray(
                plane.embed(
                    Point2.at(
                        float(vertices.at(key).coord[0]),
                        float(vertices.at(key).coord[1]),
                    )
                )
                .resolve()
                .coord,
                dtype=np.float64,
            )
            for key in vertices.roster
        ],
        dtype=np.float64,
    )
    return transform, boundary
