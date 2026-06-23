"""The fitted contact plane handed to the in-plane resolvers (origin / extent)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np

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
