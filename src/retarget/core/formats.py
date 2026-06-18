"""Rotation/pose format conversion and small time-series array helpers.

These are pure array utilities shared by the schema/runtime query surface. They
live in ``retarget.core`` so the typed scene/demo layers can use them without
importing higher layers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.core.enums import PoseFormat, RotationFormat
from retarget.core.transform import RigidTransform
from retarget.core.types import TimeBool

type JumpDetector = Callable[[np.ndarray, np.ndarray], TimeBool]
"""``(positions (T,D or T,), timestamps (T,)) -> (T,) bool`` garbage-sample detector.

Built-ins are :func:`speed_limit` (a physical speed cap) and :func:`robust_jumps`
(self-calibrating, no units). Any callable with this shape can be plugged in.
"""


def implausible_jump_mask(
    positions: np.ndarray,
    timestamps: np.ndarray,
    max_speed: float | None,
) -> TimeBool:
    """Mask frames adjacent to an implausibly fast step (a teleport / garbage sample).

    A frame is flagged when the finite-difference speed of the step into *or* out
    of it exceeds ``max_speed`` (in position-units per second). Such samples are
    physically impossible for the tracked body and make poor interpolation
    anchors, so callers treat them like occlusions.

    Args:
        positions: ``(T,)`` or ``(T, D)`` positions (e.g. segment translations).
        timestamps: Strictly increasing timestamps with shape ``(T,)``.
        max_speed: Speed threshold; ``None`` or non-positive disables detection.

    Returns:
        Boolean mask with shape ``(T,)``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    n = len(t)
    bad = np.zeros(n, dtype=np.bool_)
    if n < 2 or max_speed is None or max_speed <= 0.0:
        return bad
    dt = np.diff(t)
    deltas = np.diff(pos, axis=0)
    displacement = np.linalg.norm(deltas, axis=1) if pos.ndim > 1 else np.abs(deltas)
    speed = np.where(dt > 0.0, displacement / np.where(dt > 0.0, dt, 1.0), 0.0)
    over_speed = speed > max_speed
    bad[:-1] |= over_speed
    bad[1:] |= over_speed
    return bad


def speed_limit(max_speed: float) -> JumpDetector:
    """A :data:`JumpDetector` that flags steps faster than ``max_speed`` (units/s).

    Use when you know the physical limit of the tracked body.
    """

    def detect(positions: np.ndarray, timestamps: np.ndarray) -> TimeBool:
        return implausible_jump_mask(positions, timestamps, max_speed)

    return detect


def robust_jumps(n_sigma: float = 6.0) -> JumpDetector:
    """A self-calibrating :data:`JumpDetector` -- no units, works at any speed.

    Flags samples that deviate from a straight line through their time-neighbours
    by more than ``n_sigma`` robust scales (median + MAD). This is the local
    *curvature* (≈ acceleration), so smooth motion -- however fast -- stays
    unflagged while a teleport/discontinuity stands out. Larger ``n_sigma`` is
    more permissive.
    """

    def detect(positions: np.ndarray, timestamps: np.ndarray) -> TimeBool:
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim == 1:
            pos = pos[:, None]
        t = np.asarray(timestamps, dtype=np.float64)
        n = len(t)
        bad = np.zeros(n, dtype=np.bool_)
        if n < 3:
            return bad
        span = t[2:] - t[:-2]
        weight = np.where(span > 0.0, (t[1:-1] - t[:-2]) / np.where(span > 0.0, span, 1.0), 0.5)
        predicted = pos[:-2] + (pos[2:] - pos[:-2]) * weight[:, None]
        residual = np.linalg.norm(pos[1:-1] - predicted, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(residual))
        if scale <= 1e-12:
            return bad
        bad[1:-1] = residual > median + n_sigma * scale
        return bad

    return detect


def finite_difference_velocity(
    positions: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Return velocity with shape matching ``positions`` using ``np.gradient``."""
    if len(timestamps) == 0:
        return np.empty_like(positions)
    if len(timestamps) == 1:
        return cast(np.ndarray, np.zeros(positions.shape, dtype=positions.dtype))
    return cast(np.ndarray, np.gradient(positions, timestamps, axis=0))


def speed_from_velocity(velocity: np.ndarray) -> np.ndarray:
    """Return the L2 norm of a velocity signal along the last axis."""
    return cast(np.ndarray, np.linalg.norm(velocity, axis=-1))


def rotation_matrices_to_format(
    rotations: np.ndarray,
    *,
    format: RotationFormat,
) -> np.ndarray:
    """Convert rotation matrices with shape ``(T, 3, 3)`` to the requested format."""
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (T, 3, 3)")

    if rotations.shape[0] == 0:
        if format is RotationFormat.MATRIX:
            return np.empty((0, 3, 3), dtype=np.float64)
        if format in {RotationFormat.QUATERNION_XYZW, RotationFormat.QUATERNION_WXYZ}:
            return np.empty((0, 4), dtype=np.float64)
        if format is RotationFormat.ROTVEC:
            return np.empty((0, 3), dtype=np.float64)
        raise ValueError(f"Unsupported rotation format: {format}")

    if format is RotationFormat.MATRIX:
        return rotations

    scipy_rotations = Rotation.from_matrix(rotations)
    if format is RotationFormat.QUATERNION_XYZW:
        return np.asarray(scipy_rotations.as_quat())
    if format is RotationFormat.QUATERNION_WXYZ:
        xyzw = scipy_rotations.as_quat()
        return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
    if format is RotationFormat.ROTVEC:
        return np.asarray(scipy_rotations.as_rotvec())
    raise ValueError(f"Unsupported rotation format: {format}")


def pose_arrays_to_format(
    translations: np.ndarray,
    rotations: np.ndarray,
    *,
    format: PoseFormat,
) -> tuple[RigidTransform, ...] | np.ndarray:
    """Convert pose arrays to the requested format.

    ``translations`` must have shape ``(T, 3)`` and ``rotations`` shape ``(T, 3, 3)``.

    For :attr:`PoseFormat.TRANSLATION_ROTATION_MATRIX`, returns shape ``(T, 12)``
    with layout ``[tx, ty, tz, r00, r01, r02, r10, r11, r12, r20, r21, r22]``
    (translation followed by rotation matrix flattened in row-major order).
    """
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError("translations must have shape (T, 3)")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (T, 3, 3)")
    if translations.shape[0] != rotations.shape[0]:
        raise ValueError("translations and rotations must have the same length")

    num_timesteps = translations.shape[0]
    if num_timesteps == 0:
        if format is PoseFormat.RIGID_TRANSFORM:
            return ()
        if format is PoseFormat.MATRIX_4X4:
            return np.empty((0, 4, 4), dtype=np.float64)
        if format is PoseFormat.TRANSLATION_QUATERNION_XYZW:
            return np.empty((0, 7), dtype=np.float64)
        if format is PoseFormat.TRANSLATION_ROTATION_MATRIX:
            return np.empty((0, 12), dtype=np.float64)
        raise ValueError(f"Unsupported pose format: {format}")

    if format is PoseFormat.RIGID_TRANSFORM:
        return tuple(
            RigidTransform.from_rotation_translation(rotations[i], translations[i])
            for i in range(num_timesteps)
        )

    scipy_rotations = Rotation.from_matrix(rotations)
    if format is PoseFormat.MATRIX_4X4:
        matrices = np.zeros((num_timesteps, 4, 4), dtype=np.float64)
        matrices[:, :3, :3] = scipy_rotations.as_matrix()
        matrices[:, :3, 3] = translations
        matrices[:, 3, 3] = 1.0
        return matrices
    if format is PoseFormat.TRANSLATION_QUATERNION_XYZW:
        quats = scipy_rotations.as_quat()
        return np.concatenate([translations, quats], axis=1)
    if format is PoseFormat.TRANSLATION_ROTATION_MATRIX:
        flat_rot = scipy_rotations.as_matrix().reshape(num_timesteps, 9)
        return np.concatenate([translations, flat_rot], axis=1)
    raise ValueError(f"Unsupported pose format: {format}")
