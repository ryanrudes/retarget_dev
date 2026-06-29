"""Linear-blend-skinning mesh forward kinematics (``forward_vertices``) on the synthetic model."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import smpl


def _smpl() -> tuple[smpl.BodyModelData, smpl.SmplModel]:
    body = smpl.synthetic_model()
    return body, smpl.SmplModel(body)


def _params(body: smpl.BodyModelData, num_frames: int = 3) -> smpl.SmplParams:
    return smpl.SmplParams(
        betas=np.zeros(body.num_betas),
        pose=np.zeros((num_frames, body.num_joints, 3)),
        transl=np.zeros((num_frames, 3)),
    )


def test_synthetic_model_carries_a_valid_mesh() -> None:
    body = smpl.synthetic_model()
    assert body.has_mesh
    assert body.posedirs is not None and body.posedirs.shape == (body.num_vertices, 3, (body.num_joints - 1) * 9)
    assert body.weights is not None and body.weights.shape == (body.num_vertices, body.num_joints)
    np.testing.assert_allclose(body.weights.sum(axis=1), 1.0)


def test_rest_pose_vertices_equal_template() -> None:
    body, model = _smpl()
    verts = model.forward_vertices(_params(body))
    assert verts.shape == (3, body.num_vertices, 3)
    np.testing.assert_allclose(verts[0], body.v_template, atol=1e-12)


def test_shape_betas_deform_the_rest_mesh() -> None:
    body, model = _smpl()
    betas = np.linspace(0.5, -0.5, body.num_betas)
    params = smpl.SmplParams(betas=betas, pose=np.zeros((1, body.num_joints, 3)), transl=np.zeros((1, 3)))
    expected = body.v_template + np.einsum("vcb,b->vc", body.shapedirs, betas)  # rest pose -> just v_shaped
    np.testing.assert_allclose(model.forward_vertices(params)[0], expected, atol=1e-12)


def test_translation_shifts_vertices_rigidly() -> None:
    body, model = _smpl()
    transl = np.array([[1.0, -2.0, 3.0]])
    params = smpl.SmplParams(betas=np.zeros(body.num_betas), pose=np.zeros((1, body.num_joints, 3)), transl=transl)
    np.testing.assert_allclose(model.forward_vertices(params)[0], body.v_template + transl, atol=1e-12)


def test_root_rotation_rotates_all_vertices_about_origin() -> None:
    # Rotating only the root leaves the pose feature (non-root rotations) at zero, so there is no
    # pose-blendshape deformation: the whole mesh rotates rigidly about the (origin) root.
    body, model = _smpl()
    pose = np.zeros((1, body.num_joints, 3))
    pose[0, 0] = [0.0, 0.0, np.pi / 2]
    params = smpl.SmplParams(betas=np.zeros(body.num_betas), pose=pose, transl=np.zeros((1, 3)))
    rotation = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
    np.testing.assert_allclose(model.forward_vertices(params)[0], body.v_template @ rotation.T, atol=1e-12)


def test_nonroot_rotation_skins_descendants() -> None:
    # A non-root rotation skins its weighted descendants a lot, while the root vertex (joint 0) only
    # sees the small global pose blendshape (the synthetic chain is collinear on +x, so rotate
    # about +z to actually swing the subtree out of the axis).
    body, model = _smpl()
    pose = np.zeros((1, body.num_joints, 3))
    pose[0, 1] = [0.0, 0.0, np.pi / 2]  # rotate joint 1 about +z
    params = smpl.SmplParams(betas=np.zeros(body.num_betas), pose=pose, transl=np.zeros((1, 3)))
    verts = model.forward_vertices(params)[0]
    assert not np.allclose(verts, body.v_template)  # the mesh moved
    moved_root = float(np.linalg.norm(verts[0] - body.v_template[0]))
    moved_descendant = float(np.linalg.norm(verts[2] - body.v_template[2]))  # vertex 2 rides joint 2
    assert moved_descendant > 0.5  # skinning swings the descendant out
    assert moved_root < 0.05  # the root only sees the tiny pose blendshape, not the rotation


def test_forward_vertices_requires_a_mesh_model() -> None:
    base = smpl.synthetic_model()
    joint_only = smpl.BodyModelData(
        v_template=base.v_template,
        shapedirs=base.shapedirs,
        j_regressor=base.j_regressor,
        parents=base.parents,
        joint_names=base.joint_names,
    )
    assert not joint_only.has_mesh
    with pytest.raises(ValueError, match="joint-only"):
        smpl.SmplModel(joint_only).forward_vertices(_params(base))


def test_partial_mesh_is_rejected() -> None:
    base = smpl.synthetic_model()
    with pytest.raises(ValueError, match="both"):
        smpl.BodyModelData(
            v_template=base.v_template,
            shapedirs=base.shapedirs,
            j_regressor=base.j_regressor,
            parents=base.parents,
            joint_names=base.joint_names,
            posedirs=base.posedirs,  # weights omitted -> partial mesh
        )


def test_smplx_forward_vertices_rest_pose() -> None:
    body = smpl.synthetic_smplx_model()
    model = smpl.SmplxModel(body)
    params = smpl.SmplxParams.zeros(2, num_betas=body.num_betas)
    verts = model.forward_vertices(params)
    assert verts.shape == (2, body.num_vertices, 3)
    np.testing.assert_allclose(verts[0], body.v_template, atol=1e-12)
