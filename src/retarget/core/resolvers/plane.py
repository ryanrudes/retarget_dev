"""Surface-plane resolvers: which markers (and pre-fit offsets) define the plane."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from retarget.core.translation import AxisResolvable, MarkerTranslation
from retarget.core.types import Points3, Vec3


class PlaneResolver(ABC):
    """Supplies the segment-frame points the contact plane is fitted to.

    The one built-in fits the plane through named calibration markers, optionally
    pre-translating them before the fit.
    """

    @abstractmethod
    def required_markers(self) -> tuple[str, ...]:
        """Markers this source reads (validated against the body model)."""

    @abstractmethod
    def surface_points(
        self,
        marker_positions_segment: Mapping[str, Vec3],
        resolver: AxisResolvable,
    ) -> Points3:
        """Return the >=3 segment-frame points to fit the plane to."""


@dataclass(frozen=True, slots=True)
class PlaneFromMarkers(PlaneResolver):
    """Fit the contact plane through ``markers``, optionally pre-translated.

    ``translations`` applies a per-marker :class:`MarkerTranslation` (e.g.
    ``BodyFrameTranslation``, ``SemanticAxisTranslation``, or the
    ``0.3 * SemanticAxis.UP`` sugar) to each named marker's segment-frame position
    *before* the plane fit -- distinct from the contact-surface offset, which is
    applied along the fitted normal afterward (see the normal resolver).
    """

    markers: tuple[str, ...]
    translations: Mapping[str, MarkerTranslation] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if len(self.markers) < 3:
            raise ValueError("PlaneFromMarkers requires at least three markers")

    def required_markers(self) -> tuple[str, ...]:
        return self.markers

    def surface_points(
        self,
        marker_positions_segment: Mapping[str, Vec3],
        resolver: AxisResolvable,
    ) -> Points3:
        translations = self.translations or {}
        points: list[np.ndarray] = []
        for marker in self.markers:
            position = np.asarray(marker_positions_segment[marker], dtype=np.float64)
            translation = translations.get(marker)
            points.append(
                position if translation is None else position + translation.resolve(resolver)
            )
        return cast(Points3, np.stack(points))


def plane_from(
    *markers: str,
    translations: Mapping[str, MarkerTranslation] | None = None,
) -> PlaneFromMarkers:
    """Fit the contact plane through these calibration markers (>=3)."""
    return PlaneFromMarkers(tuple(markers), translations)
