from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, cast

from retarget.core.axes import SemanticAxis
from retarget.core.types import Vec3


class AxisResolvable(Protocol):
    """Anything that can resolve semantic axes into local-frame vectors."""

    def axis(self, axis: SemanticAxis) -> Vec3:
        ...


class MarkerTranslation(ABC):
    """Abstract marker-to-surface translation."""

    @abstractmethod
    def resolve(self, segment: AxisResolvable) -> Vec3:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class BodyFrameTranslation(MarkerTranslation):
    """Explicit marker-to-surface translation in the segment frame."""

    vector_segment: Vec3

    def __mul__(self, scale: float) -> BodyFrameTranslation:
        return BodyFrameTranslation(
            vector_segment=cast(Vec3, float(scale) * self.vector_segment),
        )

    def __rmul__(self, scale: float) -> BodyFrameTranslation:
        return self * scale

    def __neg__(self) -> BodyFrameTranslation:
        return -1.0 * self

    def resolve(self, segment: AxisResolvable) -> Vec3:
        return self.vector_segment


@dataclass(frozen=True, slots=True)
class SemanticAxisTranslation(MarkerTranslation):
    """Marker-to-surface translation along a semantic segment axis."""

    axis: SemanticAxis
    distance: float

    def __mul__(self, scale: float) -> SemanticAxisTranslation:
        return SemanticAxisTranslation(
            axis=self.axis,
            distance=float(scale) * self.distance,
        )

    def __rmul__(self, scale: float) -> SemanticAxisTranslation:
        return self * scale

    def __neg__(self) -> SemanticAxisTranslation:
        return -1.0 * self

    def resolve(self, segment: AxisResolvable) -> Vec3:
        return cast(Vec3, self.distance * segment.axis(self.axis))