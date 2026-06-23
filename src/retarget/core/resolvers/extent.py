"""Extent resolvers: the rectangle's width x height for a fitted plane."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from retarget.core.contact_region import RectangularRegion
from retarget.core.resolvers.fitted_plane import FittedPlane
from retarget.core.resolvers.origin import BoundingBoxCenter, OriginResolver


class ExtentResolver(ABC):
    """Determines the rectangle's ``width`` x ``height`` for a fitted plane."""

    def required_markers(self) -> tuple[str, ...]:
        return ()

    def default_origin(self) -> OriginResolver | None:
        """Origin to use when none is given (so the box self-centers); ``None`` keeps
        the plane-marker centroid."""
        return None

    def static_region(self) -> RectangularRegion | None:
        """The region if it does not need the plane (so it can be set at authoring)."""
        return None

    def grow(self, dwidth: float, dheight: float) -> ExtentResolver:
        """Return this extent with ``dwidth``/``dheight`` added (a composable tweak).

        e.g. ``bounding_box(*foot).grow(0.02, 0.0)`` auto-fits, then widens 2cm. Unlike
        ``padding`` (symmetric on every side), this adds to the total width/height.
        """
        return _GrownExtent(self, dwidth, dheight)

    @abstractmethod
    def fit(self, plane: FittedPlane) -> RectangularRegion:
        """Return the rectangular region for ``plane``."""


@dataclass(frozen=True, slots=True)
class _GrownExtent(ExtentResolver):
    base: ExtentResolver
    dwidth: float
    dheight: float

    def required_markers(self) -> tuple[str, ...]:
        return self.base.required_markers()

    def default_origin(self) -> OriginResolver | None:
        return self.base.default_origin()

    def fit(self, plane: FittedPlane) -> RectangularRegion:
        region = self.base.fit(plane)
        return RectangularRegion(
            width=region.width + self.dwidth, height=region.height + self.dheight
        )


@dataclass(frozen=True, slots=True)
class Fixed(ExtentResolver):
    """Explicit rectangle size (what ``width``/``height`` desugar to)."""

    width: float
    height: float

    def static_region(self) -> RectangularRegion | None:
        return RectangularRegion(width=self.width, height=self.height)

    def fit(self, plane: FittedPlane) -> RectangularRegion:
        return RectangularRegion(width=self.width, height=self.height)


@dataclass(frozen=True, slots=True)
class BoundingBox(ExtentResolver):
    """Auto-fit the rectangle to the markers' projected bounding box (+ ``padding``)."""

    markers: tuple[str, ...]
    padding: float = 0.0

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def default_origin(self) -> OriginResolver | None:
        return BoundingBoxCenter(self.markers)

    def fit(self, plane: FittedPlane) -> RectangularRegion:
        lo, hi = plane.aabb(self.markers)
        width = float(hi[0] - lo[0]) + 2.0 * self.padding
        height = float(hi[1] - lo[1]) + 2.0 * self.padding
        return RectangularRegion(width=width, height=height)


def fixed(width: float, height: float) -> ExtentResolver:
    """Explicit rectangle size."""
    return Fixed(width, height)


def bounding_box(*markers: str, padding: float = 0.0) -> ExtentResolver:
    """Auto-fit the rectangle to the markers' projected bounding box (+ ``padding``).

    With no explicit origin, the patch origin defaults to the same bounding-box
    center, so the rectangle tightly bounds these markers.
    """
    if not markers:
        raise ValueError("bounding_box requires at least one marker")
    return BoundingBox(tuple(markers), padding)
