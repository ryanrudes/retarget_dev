from __future__ import annotations

import numpy as np

from scipy.spatial.transform import Rotation

from smpl.core.types import FloatArray, IntArray


def axis_angle_to_matrix(axis_angle: FloatArray) -> FloatArray:
    """Convert a batch of axis-angle rotation vectors to rotation matrices.

    Args:
        axis_angle: Axis-angle rotation vectors of shape ``(T, J, 3)``. The direction of
            each vector is the axis of rotation and its magnitude is the angle in radians.
            A zero vector maps to the identity rotation.

    Returns:
        Rotation matrices of shape ``(T, J, 3, 3)``.
    """
    aa = np.asarray(axis_angle, dtype=np.float64)
    if aa.ndim != 3 or aa.shape[-1] != 3:
        raise ValueError(f"axis_angle must have shape (T, J, 3); got {aa.shape}.")

    frames, joints = aa.shape[0], aa.shape[1]
    flat = aa.reshape(frames * joints, 3)
    matrices = Rotation.from_rotvec(flat).as_matrix().reshape(frames, joints, 3, 3)
    return np.asarray(matrices, dtype=np.float64)


def compose(rotation: FloatArray, translation: FloatArray) -> FloatArray:
    """Build homogeneous transforms ``[[rot, t], [0, 0, 0, 1]]`` over a batch of frames.

    Args:
        rotation: Rotation matrices of shape ``(T, 3, 3)``.
        translation: Translation vector of shape ``(3,)`` (broadcast across all frames) or
            ``(T, 3)``.

    Returns:
        Homogeneous transforms of shape ``(T, 4, 4)``.
    """
    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64)
    if rot.ndim != 3 or rot.shape[1:] != (3, 3):
        raise ValueError(f"rotation must have shape (T, 3, 3); got {rot.shape}.")

    frames = rot.shape[0]
    out = np.zeros((frames, 4, 4), dtype=np.float64)
    out[:, :3, :3] = rot
    out[:, :3, 3] = trans
    out[:, 3, 3] = 1.0
    return out


def forward_kinematics(
    joint_rotations: FloatArray,
    rest_joints: FloatArray,
    parents: IntArray,
) -> FloatArray:
    """Compose world-space bone transforms over a kinematic tree, batched over frames.

    For each frame the world transform of the root is ``compose(R[0], J_rest[0])`` and, for
    every non-root joint ``i``, ``G[i] = G[parents[i]] @ compose(R[i], J_rest[i] - J_rest[parents[i]])``.
    The kinematic tree is walked once over the ``J`` joints; the ``T`` frames are the batched axis.

    Args:
        joint_rotations: Per-frame joint rotation matrices of shape ``(T, J, 3, 3)``.
        rest_joints: Rest-pose joint positions of shape ``(J, 3)``.
        parents: Parent index per joint of shape ``(J,)``; ``parents[0]`` must be ``-1`` (root)
            and every other entry must reference a strictly earlier joint.

    Returns:
        World bone transforms ``G`` of shape ``(T, J, 4, 4)``. World joint positions are
        ``G[:, :, :3, 3]``.
    """
    rotations = np.asarray(joint_rotations, dtype=np.float64)
    rest = np.asarray(rest_joints, dtype=np.float64)
    par = np.asarray(parents, dtype=np.intp)

    if rotations.ndim != 4 or rotations.shape[2:] != (3, 3):
        raise ValueError(f"joint_rotations must have shape (T, J, 3, 3); got {rotations.shape}.")
    frames, joints = rotations.shape[0], rotations.shape[1]
    if rest.shape != (joints, 3):
        raise ValueError(f"rest_joints must have shape ({joints}, 3); got {rest.shape}.")
    if par.shape != (joints,):
        raise ValueError(f"parents must have shape ({joints},); got {par.shape}.")
    if joints == 0:
        raise ValueError("rest_joints must contain at least the root joint.")
    if par[0] != -1:
        raise ValueError(f"parents[0] must be -1 for the root joint; got {par[0]}.")

    world = np.empty((frames, joints, 4, 4), dtype=np.float64)
    world[:, 0] = compose(rotations[:, 0], rest[0])
    for i in range(1, joints):
        p = int(par[i])
        if not 0 <= p < i:
            raise ValueError(f"parents[{i}] must reference an earlier joint in [0, {i}); got {p}.")
        local = compose(rotations[:, i], rest[i] - rest[p])
        world[:, i] = world[:, p] @ local
    return world
