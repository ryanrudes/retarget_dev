"""Pluggable placement of a patch's in-plane origin and rectangle extent.

The calibration markers fit the contact *plane* (normal + in-plane axes); that fixes
orientation, which is purely the ``forward_axis`` convention. These strategies fix the
two remaining choices independently, each from a possibly-different set of markers:

* :class:`PatchOrigin` -- where the origin sits within the plane;
* :class:`PatchExtent` -- how big the rectangle is.

Both are small frozen dataclasses behind an ABC (like the ``SupportModel`` /
``MarkerTranslation`` seams), so projects can add their own. Built-ins are constructed
with the factory functions :func:`bounding_box_center` / :func:`centroid` /
:func:`at_marker` and :func:`fixed` / :func:`bounding_box`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

from retarget.core.contact_region import RectangularRegion
from retarget.core.types import Mat3, Vec2, Vec3


@dataclass(frozen=True, slots=True)
class FittedPlane:
    """The fitted contact plane handed to origin/extent strategies.

    ``rotation`` columns are the patch axes in the segment frame ``[x, y, normal]``;
    ``reference`` is a point on the plane (the calibration-marker centroid). Any
    marker projects into plane-local ``(x, y)`` via :meth:`xy`.
    """

    rotation: Mat3
    reference: Vec3
    marker_positions: Mapping[str, Vec3]

    def xy(self, marker: str) -> Vec2:
        """Project ``marker`` (segment frame) into plane-local ``(x, y)``."""
        pos = np.asarray(self.marker_positions[marker], dtype=np.float64)
        ref = np.asarray(self.reference, dtype=np.float64)
        local = (pos - ref) @ np.asarray(self.rotation, dtype=np.float64)
        return cast(Vec2, local[:2])

    def _xy(self, markers: Sequence[str]) -> np.ndarray:
        return np.array([self.xy(m) for m in markers], dtype=np.float64)

    def aabb(self, markers: Sequence[str]) -> tuple[Vec2, Vec2]:
        """Axis-aligned ``(min, max)`` of the markers' projected ``(x, y)``."""
        xy = self._xy(markers)
        return cast(Vec2, xy.min(axis=0)), cast(Vec2, xy.max(axis=0))

    def centroid_xy(self, markers: Sequence[str]) -> Vec2:
        """Mean of the markers' projected ``(x, y)``."""
        return cast(Vec2, self._xy(markers).mean(axis=0))


# -- origin strategies ---------------------------------------------------------


class PatchOrigin(ABC):
    """Locates the patch-frame origin within the fitted plane.

    :meth:`locate` returns the in-plane ``(x, y)`` offset from the plane reference;
    the framework places the origin there (then applies ``normal_offset``).
    """

    def required_markers(self) -> tuple[str, ...]:
        """Markers this strategy reads (validated against the body model)."""
        return ()

    def offset(self, dx: float, dy: float) -> PatchOrigin:
        """Return this origin nudged by ``(dx, dy)`` in the plane (a composable tweak).

        e.g. ``bounding_box_center("heel", "toe").offset(0.0, 0.01)`` auto-fits, then
        shifts the origin 1cm along +y.
        """
        return _OffsetOrigin(self, dx, dy)

    @abstractmethod
    def locate(self, plane: FittedPlane) -> Vec2:
        """Return the in-plane ``(x, y)`` origin offset from ``plane.reference``."""


@dataclass(frozen=True, slots=True)
class _OffsetOrigin(PatchOrigin):
    base: PatchOrigin
    dx: float
    dy: float

    def required_markers(self) -> tuple[str, ...]:
        return self.base.required_markers()

    def locate(self, plane: FittedPlane) -> Vec2:
        x, y = self.base.locate(plane)
        return cast(Vec2, np.array([x + self.dx, y + self.dy], dtype=np.float64))


@dataclass(frozen=True, slots=True)
class _BoundingBoxCenter(PatchOrigin):
    markers: tuple[str, ...]

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def locate(self, plane: FittedPlane) -> Vec2:
        lo, hi = plane.aabb(self.markers)
        return cast(Vec2, (lo + hi) / 2.0)


@dataclass(frozen=True, slots=True)
class _Centroid(PatchOrigin):
    markers: tuple[str, ...]

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def locate(self, plane: FittedPlane) -> Vec2:
        return plane.centroid_xy(self.markers)


@dataclass(frozen=True, slots=True)
class _AtMarker(PatchOrigin):
    marker: str

    def required_markers(self) -> tuple[str, ...]:
        return (self.marker,)

    def locate(self, plane: FittedPlane) -> Vec2:
        return plane.xy(self.marker)


def bounding_box_center(*markers: str) -> PatchOrigin:
    """Origin at the center of the markers' projected bounding rectangle."""
    if not markers:
        raise ValueError("bounding_box_center requires at least one marker")
    return _BoundingBoxCenter(tuple(markers))


def centroid(*markers: str) -> PatchOrigin:
    """Origin at the mean of the markers' projected positions."""
    if not markers:
        raise ValueError("centroid requires at least one marker")
    return _Centroid(tuple(markers))


def at_marker(marker: str) -> PatchOrigin:
    """Origin at a single marker's projection onto the plane."""
    return _AtMarker(marker)


# -- extent strategies ---------------------------------------------------------


class PatchExtent(ABC):
    """Determines the rectangle's ``width`` x ``height`` for a fitted plane."""

    def required_markers(self) -> tuple[str, ...]:
        return ()

    def default_origin(self) -> PatchOrigin | None:
        """Origin to use when none is given (so the box self-centers); ``None`` keeps
        the plane-marker centroid."""
        return None

    def static_region(self) -> RectangularRegion | None:
        """The region if it does not need the plane (so it can be set at authoring)."""
        return None

    def grow(self, dwidth: float, dheight: float) -> PatchExtent:
        """Return this extent with ``dwidth``/``dheight`` added (a composable tweak).

        e.g. ``bounding_box(*foot).grow(0.02, 0.0)`` auto-fits, then widens 2cm. Unlike
        ``padding`` (symmetric on every side), this adds to the total width/height.
        """
        return _GrownExtent(self, dwidth, dheight)

    @abstractmethod
    def fit(self, plane: FittedPlane) -> RectangularRegion:
        """Return the rectangular region for ``plane``."""


@dataclass(frozen=True, slots=True)
class _GrownExtent(PatchExtent):
    base: PatchExtent
    dwidth: float
    dheight: float

    def required_markers(self) -> tuple[str, ...]:
        return self.base.required_markers()

    def default_origin(self) -> PatchOrigin | None:
        return self.base.default_origin()

    def fit(self, plane: FittedPlane) -> RectangularRegion:
        region = self.base.fit(plane)
        return RectangularRegion(
            width=region.width + self.dwidth, height=region.height + self.dheight
        )


@dataclass(frozen=True, slots=True)
class _FixedExtent(PatchExtent):
    width: float
    height: float

    def static_region(self) -> RectangularRegion | None:
        return RectangularRegion(width=self.width, height=self.height)

    def fit(self, plane: FittedPlane) -> RectangularRegion:
        return RectangularRegion(width=self.width, height=self.height)


@dataclass(frozen=True, slots=True)
class _BoundingBoxExtent(PatchExtent):
    markers: tuple[str, ...]
    padding: float = 0.0

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def default_origin(self) -> PatchOrigin | None:
        return _BoundingBoxCenter(self.markers)

    def fit(self, plane: FittedPlane) -> RectangularRegion:
        lo, hi = plane.aabb(self.markers)
        width = float(hi[0] - lo[0]) + 2.0 * self.padding
        height = float(hi[1] - lo[1]) + 2.0 * self.padding
        return RectangularRegion(width=width, height=height)


def fixed(width: float, height: float) -> PatchExtent:
    """Explicit rectangle size (what ``width=``/``height=`` desugar to)."""
    return _FixedExtent(width, height)


def bounding_box(*markers: str, padding: float = 0.0) -> PatchExtent:
    """Auto-fit the rectangle to the markers' projected bounding box (+ ``padding``).

    With no explicit origin, the patch origin defaults to the same bounding-box
    center, so the rectangle tightly bounds these markers.
    """
    if not markers:
        raise ValueError("bounding_box requires at least one marker")
    return _BoundingBoxExtent(tuple(markers), padding)
