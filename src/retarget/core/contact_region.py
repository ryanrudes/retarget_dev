from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal
from dataclasses import dataclass

import numpy as np

from retarget.core.types import Vec2
from retarget.utils.geometry import point_in_polygon


class ContactRegion(ABC):
    """Finite region in a patch's local xy plane."""

    @abstractmethod
    def contains(self, xy: Vec2) -> bool:
        """Check if a point is in the contact region."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RectangularRegion(ContactRegion):
    """A rectangular region in a patch's local xy plane."""

    width: float
    """The width of the rectangular region."""

    height: float
    """The height of the rectangular region."""

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")

    def contains(self, xy: Vec2) -> bool:
        x, y = xy
        return abs(x) <= self.width / 2 and abs(y) <= self.height / 2


@dataclass(frozen=True, slots=True)
class PolygonalRegion(ContactRegion):
    vertices: np.ndarray[
        tuple[int, Literal[2]],
        np.dtype[np.floating[Any]],
    ]

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError(
                f"Expected vertices with shape (N, 2), got {self.vertices.shape}"
            )
        if self.vertices.shape[0] < 3:
            raise ValueError("PolygonalRegion requires at least three vertices")

    def contains(self, xy: Vec2) -> bool:
        return point_in_polygon(xy, self.vertices)