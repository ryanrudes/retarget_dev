from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, cast

import numpy as np

from retarget.core.types import Sign, Vec3

if TYPE_CHECKING:
    from retarget.core.translation import SemanticAxisTranslation


@dataclass(frozen=True)
class SignedAxis:
    """A signed axis is an axis with a direction."""

    axis: CoordinateAxis
    """The axis to represent."""

    sign: Sign = 1
    """The sign of the axis."""

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be either -1 or +1")

    def __pos__(self) -> SignedAxis:
        """Point the axis in the positive direction."""
        return SignedAxis(axis=self.axis, sign=1)

    def __neg__(self) -> SignedAxis:
        """Point the axis in the negative direction."""
        return SignedAxis(axis=self.axis, sign=-1)

    def flip(self) -> SignedAxis:
        """Flip the sign of the axis."""
        return SignedAxis(axis=self.axis, sign=-self.sign)

    def vector(self) -> Vec3:
        """Get the unit vector representation of the axis."""
        v = np.zeros(3, dtype=np.float64)
        v[int(self.axis)] = float(self.sign)
        return cast(Vec3, v)


class CoordinateAxis(IntEnum):
    """An axis in a coordinate system.
    
    Attributes:
        `X` (int): The X axis.
        `Y` (int): The Y axis.
        `Z` (int): The Z axis.
    """

    X = 0
    """The X axis."""
    Y = 1
    """The Y axis."""
    Z = 2
    """The Z axis."""

    def __pos__(self) -> SignedAxis:  # type: ignore[override]
        """Point the axis in the positive direction.
        
        Returns:
            SignedAxis: The axis pointing in the positive direction.
        """
        return SignedAxis(axis=self, sign=+1)

    def __neg__(self) -> SignedAxis:  # type: ignore[override]
        """Point the axis in the negative direction.
        
        Returns:
            SignedAxis: The axis pointing in the negative direction.
        """
        return SignedAxis(axis=self, sign=-1)


class SemanticAxis(StrEnum):
    """An axis in a semantic coordinate system.
    
    Attributes:
        `RIGHT` (str): The right axis.
        `FORWARD` (str): The forward axis.
        `UP` (str): The up axis.
    """

    RIGHT = "right"
    """The right axis."""
    FORWARD = "forward"
    """The forward axis."""
    UP = "up"
    """The up axis."""

    def __mul__(self, distance: float) -> SemanticAxisTranslation:  # type: ignore[override]
        from retarget.core.translation import SemanticAxisTranslation

        return SemanticAxisTranslation(
            axis=self,
            distance=float(distance),
        )

    def __rmul__(self, distance: float) -> SemanticAxisTranslation:  # type: ignore[override]
        return self * distance

    def __neg__(self) -> SignedSemanticAxis:
        return SignedSemanticAxis(axis=self, sign=-1)

    def __pos__(self) -> SignedSemanticAxis:
        return SignedSemanticAxis(axis=self, sign=1)


@dataclass(frozen=True, slots=True)
class SignedSemanticAxis:
    """A semantic axis with a direction sign, e.g. ``+SemanticAxis.UP`` / ``-SemanticAxis.UP``.

    Mirrors :class:`SignedAxis` (over concrete ``CoordinateAxis``) but stays in the
    semantic space, so it can be resolved against any :class:`AxisConvention`.
    """

    axis: SemanticAxis
    sign: Sign = 1

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise ValueError("sign must be either -1 or +1")

    def __pos__(self) -> SignedSemanticAxis:
        return SignedSemanticAxis(axis=self.axis, sign=1)

    def __neg__(self) -> SignedSemanticAxis:
        return SignedSemanticAxis(axis=self.axis, sign=-self.sign)

    def vector(self, convention: AxisConvention) -> Vec3:
        """Unit world vector for this signed semantic axis under ``convention``."""
        return cast(Vec3, convention.vector(self.axis) * float(self.sign))


type SignedSemanticAxisLike = SemanticAxis | SignedSemanticAxis


def as_signed_semantic_axis(value: SignedSemanticAxisLike) -> SignedSemanticAxis:
    """Coerce a bare ``SemanticAxis`` (positive) or ``SignedSemanticAxis`` to the latter."""
    return value if isinstance(value, SignedSemanticAxis) else SignedSemanticAxis(axis=value)


@dataclass(frozen=True, slots=True)
class AxisConvention:
    """Mapping from semantic axes to concrete signed coordinate axes."""

    axes: Mapping[SemanticAxis, SignedAxis]
    """The mapping from semantic axes to concrete signed coordinate axes."""

    def __post_init__(self) -> None:
        missing = set(SemanticAxis) - set(self.axes)
        if missing:
            raise ValueError(f"Missing semantic axes: {missing}")
        
        vectors = np.stack([self.vector(axis) for axis in SemanticAxis], axis=0)
        if np.linalg.matrix_rank(vectors) != 3:
            raise ValueError("Axis convention must define three independent axes")

    def signed_axis(self, axis: SemanticAxis) -> SignedAxis:
        """Get the signed axis for a semantic axis."""
        return self.axes[axis]

    def vector(self, axis: SemanticAxis) -> Vec3:
        """Get the unit vector representation of a semantic axis."""
        return self.signed_axis(axis).vector()


Z_UP_AXES = AxisConvention({
    SemanticAxis.RIGHT: -CoordinateAxis.Y,
    SemanticAxis.FORWARD: +CoordinateAxis.X,
    SemanticAxis.UP: +CoordinateAxis.Z,
})

Y_UP_AXES = AxisConvention({
    SemanticAxis.RIGHT: +CoordinateAxis.X,
    SemanticAxis.FORWARD: +CoordinateAxis.Z,
    SemanticAxis.UP: +CoordinateAxis.Y,
})

MUJOCO_AXES = Z_UP_AXES
ISAAC_AXES = Z_UP_AXES