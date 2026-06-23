"""In-plane origin resolvers: where the patch frame's origin sits within the plane."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

import numpy as np

from retarget.core.resolvers.fitted_plane import FittedPlane
from retarget.core.types import Vec2


class OriginResolver(ABC):
    """Locates the patch-frame origin within the fitted plane.

    :meth:`locate` returns the in-plane ``(x, y)`` offset from the plane reference;
    the framework places the origin there (then applies the normal resolver's offset).
    """

    def required_markers(self) -> tuple[str, ...]:
        """Markers this strategy reads (validated against the body model)."""
        return ()

    def offset(self, dx: float, dy: float) -> OriginResolver:
        """Return this origin nudged by ``(dx, dy)`` in the plane (a composable tweak).

        e.g. ``bounding_box_center("heel", "toe").offset(0.0, 0.01)`` auto-fits, then
        shifts the origin 1cm along +y.
        """
        return _OffsetOrigin(self, dx, dy)

    @abstractmethod
    def locate(self, plane: FittedPlane) -> Vec2:
        """Return the in-plane ``(x, y)`` origin offset from ``plane.reference``."""


@dataclass(frozen=True, slots=True)
class _OffsetOrigin(OriginResolver):
    base: OriginResolver
    dx: float
    dy: float

    def required_markers(self) -> tuple[str, ...]:
        return self.base.required_markers()

    def locate(self, plane: FittedPlane) -> Vec2:
        x, y = self.base.locate(plane)
        return cast(Vec2, np.array([x + self.dx, y + self.dy], dtype=np.float64))


@dataclass(frozen=True, slots=True)
class BoundingBoxCenter(OriginResolver):
    """Origin at the center of the markers' projected bounding rectangle."""

    markers: tuple[str, ...]

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def locate(self, plane: FittedPlane) -> Vec2:
        lo, hi = plane.aabb(self.markers)
        return cast(Vec2, (lo + hi) / 2.0)


@dataclass(frozen=True, slots=True)
class Centroid(OriginResolver):
    """Origin at the mean of the markers' projected positions."""

    markers: tuple[str, ...]

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def locate(self, plane: FittedPlane) -> Vec2:
        return plane.centroid_xy(self.markers)


@dataclass(frozen=True, slots=True)
class AtMarker(OriginResolver):
    """Origin at a single marker's projection onto the plane."""

    marker: str

    def required_markers(self) -> tuple[str, ...]:
        return (self.marker,)

    def locate(self, plane: FittedPlane) -> Vec2:
        return plane.xy(self.marker)


def bounding_box_center(*markers: str) -> OriginResolver:
    """Origin at the center of the markers' projected bounding rectangle."""
    if not markers:
        raise ValueError("bounding_box_center requires at least one marker")
    return BoundingBoxCenter(tuple(markers))


def centroid(*markers: str) -> OriginResolver:
    """Origin at the mean of the markers' projected positions."""
    if not markers:
        raise ValueError("centroid requires at least one marker")
    return Centroid(tuple(markers))


def at_marker(marker: str) -> OriginResolver:
    """Origin at a single marker's projection onto the plane."""
    return AtMarker(marker)
