"""Tangential-orientation resolvers: the in-plane +x axis of the patch frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.spatial import ConvexHull

from retarget.core.axes import AxisConvention, SemanticAxis
from retarget.core.types import Vec3


class TangentialResolver(ABC):
    """Determines the in-plane +x (tangential) axis of the patch frame.

    The plane fit fixes the normal; this fixes the rotation *about* it. :meth:`tangential_x`
    returns a desired +x direction in the segment frame; the caller projects it onto the
    plane and orthonormalizes. Built via :func:`along_axis` / :func:`toward` /
    :func:`min_area_rectangle`.
    """

    def required_markers(self) -> tuple[str, ...]:
        """Markers this strategy reads (validated against the body model)."""
        return ()

    @abstractmethod
    def tangential_x(
        self,
        *,
        normal: Vec3,
        reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> Vec3:
        """Return the desired +x direction (segment frame)."""


@dataclass(frozen=True, slots=True)
class AlongAxis(TangentialResolver):
    """Orient +x along a semantic axis projected onto the plane (the default)."""

    axis: SemanticAxis

    def tangential_x(
        self,
        *,
        normal: Vec3,
        reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> Vec3:
        return axis_convention.vector(self.axis)


@dataclass(frozen=True, slots=True)
class TowardMarker(TangentialResolver):
    """Orient +x toward a marker's in-plane projection -- an explicit, intuitive sign."""

    marker: str

    def required_markers(self) -> tuple[str, ...]:
        return (self.marker,)

    def tangential_x(
        self,
        *,
        normal: Vec3,
        reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> Vec3:
        pos = np.asarray(marker_positions_segment[self.marker], dtype=np.float64)
        return cast(Vec3, pos - np.asarray(reference, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class MinAreaRectangle(TangentialResolver):
    """Align the in-plane axes to the minimum-area bounding rectangle of these markers.

    +x runs along the rectangle's longer edge; ``forward`` disambiguates the +x sign
    (and seeds the temp basis). Pair with ``extent=bounding_box(*same_markers)`` to
    recover that tight rectangle: the min-area rectangle is exactly the axis-aligned
    box measured in its own edge-aligned frame.
    """

    markers: tuple[str, ...]
    forward: SemanticAxis = SemanticAxis.FORWARD

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def tangential_x(
        self,
        *,
        normal: Vec3,
        reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> Vec3:
        normal_v = np.asarray(normal, dtype=np.float64)
        reference_v = np.asarray(reference, dtype=np.float64)
        forward_v = np.asarray(axis_convention.vector(self.forward), dtype=np.float64)
        e1, e2 = _in_plane_basis(cast(Vec3, normal_v), cast(Vec3, forward_v))
        pts = np.array(
            [np.asarray(marker_positions_segment[m], dtype=np.float64) for m in self.markers]
        )
        if len(pts) < 3:
            raise ValueError("min_area_rectangle needs at least three markers")
        delta = pts - reference_v
        xy = np.column_stack([delta @ e1, delta @ e2])
        angle = _min_area_rectangle_angle(xy)
        x = float(np.cos(angle)) * e1 + float(np.sin(angle)) * e2
        if float(np.dot(x, forward_v)) < 0.0:
            x = -x
        return cast(Vec3, x)


def along_axis(axis: SemanticAxis) -> AlongAxis:
    """Orient +x along a semantic axis projected onto the plane (the default)."""
    return AlongAxis(axis)


def toward(marker: str) -> TowardMarker:
    """Orient +x toward a marker's in-plane projection -- an explicit, intuitive sign."""
    return TowardMarker(marker)


def min_area_rectangle(*markers: str, forward: SemanticAxis = SemanticAxis.FORWARD) -> MinAreaRectangle:
    """Align the in-plane axes to the minimum-area bounding rectangle of these markers.

    +x runs along the rectangle's longer edge (sign disambiguated by ``forward``).
    """
    if len(markers) < 3:
        raise ValueError("min_area_rectangle requires at least three markers")
    return MinAreaRectangle(tuple(markers), forward)


def _in_plane_basis(normal: Vec3, forward_hint: Vec3) -> tuple[Vec3, Vec3]:
    """An orthonormal in-plane basis ``(e1, e2)`` with ``e1`` ~ the projected forward."""
    n = np.asarray(normal, dtype=np.float64)
    e1 = np.asarray(forward_hint, dtype=np.float64)
    e1 = e1 - np.dot(e1, n) * n
    if np.linalg.norm(e1) < 1e-9:
        for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            e1 = axis - np.dot(axis, n) * n
            if np.linalg.norm(e1) >= 1e-9:
                break
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 = e2 / np.linalg.norm(e2)
    return cast(Vec3, e1), cast(Vec3, e2)


def _min_area_rectangle_angle(points_2d: np.ndarray) -> float:
    """Angle (rad) of the longer edge of the min-area bounding rectangle of 2D points."""
    try:
        hull = points_2d[ConvexHull(points_2d).vertices]
    except Exception as exc:  # QhullError: collinear / degenerate footprint
        raise ValueError("min_area_rectangle: markers are collinear or degenerate") from exc
    best_area: float | None = None
    best_angle = 0.0
    n = len(hull)
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        angle = float(np.arctan2(edge[1], edge[0]))
        c, s = np.cos(-angle), np.sin(-angle)
        rotated = hull @ np.array([[c, -s], [s, c]]).T
        extent = rotated.max(axis=0) - rotated.min(axis=0)
        area = float(extent[0] * extent[1])
        if best_area is None or area < best_area:
            best_area = area
            best_angle = angle if extent[0] >= extent[1] else angle + np.pi / 2.0
    return best_angle
