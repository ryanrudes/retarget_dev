"""Private array caches and format conversion for mocap demonstration tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.core.enums import PoseFormat, RotationFormat
from retarget.core.keys import SegmentKey
from retarget.core.transform import RigidTransform


@dataclass(slots=True)
class MocapArrayCache:
    """Lazy, mutable AoS caches keyed by segment identity.

    Owned privately by :class:`~retarget.demo.mocap.MocapTrack` and not part of
    the public API. ``MocapTrack`` remains frozen; this object holds mutable
    lazy-cache state.
    """

    translations: dict[SegmentKey, np.ndarray] = field(default_factory=dict)
    rotations: dict[SegmentKey, np.ndarray] = field(default_factory=dict)
    observed_markers: dict[SegmentKey, np.ndarray] = field(default_factory=dict)


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
