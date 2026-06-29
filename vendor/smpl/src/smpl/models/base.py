from __future__ import annotations

import numpy as np

from smpl.core.lbs import axis_angle_to_matrix, forward_kinematics
from smpl.core.types import FloatArray
from smpl.io.load import BodyModelData

"""Shared joint-level forward kinematics for every SMPL-family variant.

The per-variant model classes differ only in how their typed params pack the per-joint
axis-angle pose; once a pose is assembled to ``(T, J, 3)`` the math is identical (shape the
template, regress rest joints, pose the tree, translate). These free functions hold that shared
glue so each model stays a thin facade over the same core LBS kernel.
"""


def shaped_template(body: BodyModelData, betas: FloatArray) -> FloatArray:
    """Return the shape-deformed template ``v_template + shapedirs @ betas`` of shape ``(V, 3)``."""
    betas = np.asarray(betas, dtype=np.float64)
    if betas.shape != (body.num_betas,):
        raise ValueError(f"betas must have shape ({body.num_betas},); got {betas.shape}.")
    displacement: FloatArray = np.einsum("vcb,b->vc", body.shapedirs, betas)
    return body.v_template + displacement


def rest_joints(body: BodyModelData, betas: FloatArray) -> FloatArray:
    """Return the rest-pose joints ``j_regressor @ shaped_template(betas)`` of shape ``(J, 3)``."""
    return body.j_regressor @ shaped_template(body, betas)


def forward_transforms(
    body: BodyModelData,
    betas: FloatArray,
    pose: FloatArray,
    transl: FloatArray,
) -> FloatArray:
    """Run joint-level forward kinematics for an assembled axis-angle pose.

    Args:
        body: The structured arrays defining the body model.
        betas: Shape coefficients of shape ``(B,)``.
        pose: The assembled per-joint axis-angle pose of shape ``(T, J, 3)``; index ``0`` is the
            global orientation. ``J`` must match the model's joint count.
        transl: Root translation of shape ``(T, 3)``, added into every joint's translation block.

    Returns:
        World bone transforms of shape ``(T, J, 4, 4)``; world joint positions are ``[..., :3, 3]``.
    """
    pose = np.asarray(pose, dtype=np.float64)
    num_joints = pose.shape[1] if pose.ndim == 3 else -1
    if num_joints != body.num_joints:
        raise ValueError(f"pose has {num_joints} joints but the model has {body.num_joints}.")

    rest = rest_joints(body, betas)
    rotations = axis_angle_to_matrix(pose)
    transforms = forward_kinematics(rotations, rest, body.parents)
    transl = np.asarray(transl, dtype=np.float64)
    transforms[:, :, :3, 3] += transl[:, None, :]
    return transforms


def forward_vertices(
    body: BodyModelData,
    betas: FloatArray,
    pose: FloatArray,
    transl: FloatArray,
) -> FloatArray:
    """Run full linear-blend skinning, returning posed world-frame mesh vertices ``(T, V, 3)``.

    The shaped template is deformed by the pose blendshapes, skinned by the per-vertex blend of the
    rest-corrected world bone transforms, then translated. Requires ``body.has_mesh`` (pose
    blendshapes + skinning weights); raises otherwise.

    Args:
        body: The structured arrays defining the body model (must carry ``posedirs`` + ``weights``).
        betas: Shape coefficients of shape ``(B,)``.
        pose: The assembled per-joint axis-angle pose of shape ``(T, J, 3)``; index ``0`` is global.
        transl: Root translation of shape ``(T, 3)``.

    Returns:
        World-frame mesh vertices of shape ``(T, V, 3)``.
    """
    if body.posedirs is None or body.weights is None:
        raise ValueError("forward_vertices needs pose blendshapes + skinning weights; this model is joint-only.")
    pose = np.asarray(pose, dtype=np.float64)
    num_joints = pose.shape[1] if pose.ndim == 3 else -1
    if num_joints != body.num_joints:
        raise ValueError(f"pose has {num_joints} joints but the model has {body.num_joints}.")
    num_frames = pose.shape[0]

    rest = rest_joints(body, betas)  # (J, 3)
    v_shaped = shaped_template(body, betas)  # (V, 3)
    rotations = axis_angle_to_matrix(pose)  # (T, J, 3, 3)

    # pose blendshapes: feature is the non-root rotation matrices minus identity, flattened.
    pose_feature = (rotations[:, 1:] - np.eye(3)).reshape(num_frames, (num_joints - 1) * 9)
    v_posed = v_shaped[None] + np.einsum("vcp,tp->tvc", body.posedirs, pose_feature)  # (T, V, 3)

    # rest-corrected world bone transforms (no translation -- that is applied to the vertices below).
    transforms = forward_kinematics(rotations, rest, body.parents)  # (T, J, 4, 4)
    unposed = np.broadcast_to(np.eye(4), (num_joints, 4, 4)).copy()
    unposed[:, :3, 3] = -rest
    relative = np.einsum("tjab,jbc->tjac", transforms, unposed)  # (T, J, 4, 4)

    blended = np.einsum("vj,tjab->tvab", body.weights, relative)  # (T, V, 4, 4)
    homogeneous = np.concatenate([v_posed, np.ones((num_frames, v_posed.shape[1], 1))], axis=-1)
    world: FloatArray = np.einsum("tvab,tvb->tva", blended, homogeneous)[..., :3]  # (T, V, 3)
    return world + np.asarray(transl, dtype=np.float64)[:, None, :]
