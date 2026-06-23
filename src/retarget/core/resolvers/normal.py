"""Normal-direction resolvers: which side of the plane is outward, and the offset."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import numpy as np

from retarget.core.axes import (
    AxisConvention,
    SemanticAxis,
    SignedSemanticAxisLike,
    as_signed_semantic_axis,
)
from retarget.core.types import Vec3


class WindingDirection(StrEnum):
    """Direction an ordered marker loop traces as seen from the outward side."""

    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"


class Chirality(StrEnum):
    """Hand rule mapping a winding to a thumb (normal) direction."""

    RIGHT_HANDED = "right_handed"
    LEFT_HANDED = "left_handed"


@dataclass(frozen=True, slots=True)
class NormalResolution:
    """A normal resolver's output: the outward (+normal) direction and contact offset."""

    outward_segment: Vec3
    """Unit outward direction in the segment frame (becomes the patch +normal)."""
    offset: float
    """Contact-surface offset along ``outward_segment`` (applied after origin placement)."""


def _planar_up(defines: SignedSemanticAxisLike) -> None:
    """Validate that a normal resolver's ``defines`` is the (signed) UP axis."""
    signed = as_signed_semantic_axis(defines)
    if signed.axis is not SemanticAxis.UP:
        raise ValueError(
            "a planar patch normal resolver can only define the UP axis (its "
            f"out-of-plane normal); got {signed.axis!r}"
        )


class NormalResolver(ABC):
    """Resolves the patch's outward (+normal) direction and contact-surface offset.

    The SVD plane fit yields a normal only up to sign; a resolver fixes that sign
    (so the offset's sign is unambiguous) and carries the offset. Built via
    :func:`axis_normal` / :func:`side` / :func:`winding`.
    """

    def required_markers(self) -> tuple[str, ...]:
        """Markers this resolver reads (validated against the body model)."""
        return ()

    @abstractmethod
    def resolve(
        self,
        *,
        plane_normal_segment: Vec3,
        plane_reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> NormalResolution:
        """Return the outward direction (segment frame) + offset for this plane."""


@dataclass(frozen=True, slots=True)
class AxisNormal(NormalResolver):
    """Outward normal aligned to a (signed) world semantic axis -- the common default."""

    defines: SignedSemanticAxisLike = SemanticAxis.UP
    offset: float = 0.0

    def __post_init__(self) -> None:
        _planar_up(self.defines)

    def resolve(
        self,
        *,
        plane_normal_segment: Vec3,
        plane_reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> NormalResolution:
        defines = as_signed_semantic_axis(self.defines)
        return NormalResolution(outward_segment=defines.vector(axis_convention), offset=self.offset)


@dataclass(frozen=True, slots=True)
class SideNormal(NormalResolver):
    """Resolve the normal sign from the halfspace containing an anchor marker.

    With ``defines=+UP`` the patch UP (outward normal) points toward the anchor;
    ``-SemanticAxis.UP`` points away (the anchor is on the inward/body side).
    """

    anchor: tuple[str, ...]
    defines: SignedSemanticAxisLike = SemanticAxis.UP
    offset: float = 0.0

    def __post_init__(self) -> None:
        if not self.anchor:
            raise ValueError("SideNormal requires at least one anchor marker")
        _planar_up(self.defines)

    def required_markers(self) -> tuple[str, ...]:
        return self.anchor

    def resolve(
        self,
        *,
        plane_normal_segment: Vec3,
        plane_reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> NormalResolution:
        normal = np.asarray(plane_normal_segment, dtype=np.float64)
        normal = normal / np.linalg.norm(normal)
        reference = np.asarray(plane_reference, dtype=np.float64)
        anchor_centroid = np.mean(
            [np.asarray(marker_positions_segment[m], dtype=np.float64) for m in self.anchor],
            axis=0,
        )
        side = float(np.dot(anchor_centroid - reference, normal))
        if abs(side) < 1e-9:
            raise ValueError(
                f"side anchor {self.anchor} lies in the patch plane; it cannot "
                "disambiguate the normal side"
            )
        toward_anchor = normal if side > 0.0 else -normal
        outward = toward_anchor * float(as_signed_semantic_axis(self.defines).sign)
        return NormalResolution(outward_segment=cast(Vec3, outward), offset=self.offset)


@dataclass(frozen=True, slots=True)
class WindingNormal(NormalResolver):
    """Resolve the normal sign from an ordered marker loop and a hand rule.

    ``order`` (>=3 markers, not necessarily the plane markers) is projected onto
    the plane; ``direction`` states how that order winds *as seen from the outward
    side*; the ``chirality`` hand rule then fixes the outward normal. The loop must
    be simple and convex -- a degenerate or self-intersecting order raises.
    """

    order: tuple[str, ...]
    direction: WindingDirection
    defines: SignedSemanticAxisLike = SemanticAxis.UP
    chirality: Chirality = Chirality.RIGHT_HANDED
    offset: float = 0.0

    def __post_init__(self) -> None:
        if len(self.order) < 3:
            raise ValueError("WindingNormal requires at least three ordered markers")
        _planar_up(self.defines)

    def required_markers(self) -> tuple[str, ...]:
        return self.order

    def resolve(
        self,
        *,
        plane_normal_segment: Vec3,
        plane_reference: Vec3,
        marker_positions_segment: Mapping[str, Vec3],
        axis_convention: AxisConvention,
    ) -> NormalResolution:
        reference = np.asarray(plane_reference, dtype=np.float64)
        loop = [
            np.asarray(marker_positions_segment[m], dtype=np.float64) - reference
            for m in self.order
        ]
        n = len(loop)
        edges = [np.cross(loop[k], loop[(k + 1) % n]) for k in range(n)]
        area = 0.5 * np.sum(edges, axis=0)
        norm = float(np.linalg.norm(area))
        if norm < 1e-12:
            raise ValueError(
                f"winding markers {self.order} are collinear/degenerate when projected "
                "onto the patch plane; they cannot resolve the normal"
            )
        thumb_raw = area / norm
        if not all(float(np.dot(edge, thumb_raw)) > 0.0 for edge in edges):
            raise ValueError(
                f"winding markers {self.order} do not trace a simple convex loop "
                "(self-intersecting or misordered); they cannot resolve the normal"
            )
        hand = 1.0 if self.chirality is Chirality.RIGHT_HANDED else -1.0
        wind = 1.0 if self.direction is WindingDirection.COUNTERCLOCKWISE else -1.0
        outward = thumb_raw * hand * wind * float(as_signed_semantic_axis(self.defines).sign)
        return NormalResolution(outward_segment=cast(Vec3, outward), offset=self.offset)


def axis_normal(
    axis: SignedSemanticAxisLike = SemanticAxis.UP,
    *,
    offset: float = 0.0,
) -> AxisNormal:
    """Outward normal aligned to a (signed) world semantic axis (default outward = UP)."""
    return AxisNormal(axis, offset)


def side(
    anchor: str | Sequence[str],
    *,
    defines: SignedSemanticAxisLike = SemanticAxis.UP,
    offset: float = 0.0,
) -> SideNormal:
    """Resolve the normal sign from the halfspace containing the anchor marker(s)."""
    markers = (anchor,) if isinstance(anchor, str) else tuple(anchor)
    return SideNormal(markers, defines, offset)


def winding(
    order: Sequence[str],
    *,
    direction: WindingDirection,
    defines: SignedSemanticAxisLike = SemanticAxis.UP,
    chirality: Chirality = Chirality.RIGHT_HANDED,
    offset: float = 0.0,
) -> WindingNormal:
    """Resolve the normal sign from an ordered marker loop + a hand rule."""
    return WindingNormal(tuple(order), direction, defines, chirality, offset)
