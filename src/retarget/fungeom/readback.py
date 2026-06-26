"""Read values back out of fungeom resolvers, surfacing partiality honestly.

The resolving helpers (`point_array`, `pose_matrix`, `marker_at`, `joint_at`)
call ``resolve()``, which raises ``UnresolvableError`` carrying fungeom's reason
when a value is partial (occluded marker, antipodal blend, off-support query) —
the reason is surfaced, never swallowed. ``resolvability`` is the non-raising
counterpart: it returns the ``Resolvable``/``Unresolvable`` sum so callers can
pattern-match instead.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from fungeom import (
    Point3,
    Point3BundleSignal,
    Resolvable,
    Resolver,
    Transform,
    TransformBundleSignal,
    Unresolvable,
)

from retarget.core.types import Vec3

__all__ = [
    "Mat4",
    "point_array",
    "pose_matrix",
    "marker_at",
    "joint_at",
    "resolvability",
]

# Deliberately widens fungeom's float64 ``Mat4`` to ``floating[Any]`` to match
# retarget's own array-dtype convention (``Vec3``/``TimeVec3`` in core.types).
type Mat4 = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.floating[Any]]]


def point_array(point: Point3) -> Vec3:
    """Resolve a :class:`Point3` to its ``(3,)`` world coordinates (an owned copy).

    Raises ``UnresolvableError`` (with fungeom's reason) if the point is partial.
    """
    # np.array (not asarray) copies: fungeom freezes its value buffers, and the
    # caller owns this result, so we must not hand back the read-only view.
    return cast(Vec3, np.array(point.resolve().coord, dtype=np.float64))


def pose_matrix(transform: Transform) -> Mat4:
    """Resolve a :class:`Transform` to its ``(4, 4)`` homogeneous matrix (owned copy).

    Raises ``UnresolvableError`` (with fungeom's reason) if the pose is partial.
    """
    return cast(Mat4, np.array(transform.resolve().matrix, dtype=np.float64))


def marker_at(signal: Point3BundleSignal, t: float, key: str) -> Vec3:
    """The world position ``(3,)`` of marker ``key`` at time ``t``.

    Occluded ⇒ ``UnresolvableError("key '…' is absent from the bundle")``.
    """
    return point_array(signal.at(t).at(key))


def joint_at(signal: TransformBundleSignal, t: float, key: str) -> Mat4:
    """The pose ``(4, 4)`` of segment/joint ``key`` at time ``t``.

    Absent ⇒ ``UnresolvableError``. An interpolated instant whose SE(3) blend is
    antipodal surfaces fungeom's op-failure reason rather than crashing.
    """
    return pose_matrix(signal.at(t).at(key))


def resolvability[T](node: Resolver[T]) -> Resolvable[T] | Unresolvable:
    """Decide ``node`` without raising — the ``Resolvable``/``Unresolvable`` sum.

    Use this to branch on partiality (``isinstance(d, Unresolvable)`` → ``d.reason``)
    instead of catching ``UnresolvableError`` from the resolving helpers.
    """
    return node.decide()
