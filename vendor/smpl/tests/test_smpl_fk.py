from __future__ import annotations

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from smpl import SmplModel, SmplParams, synthetic_model


def _zero_params(num_frames: int, num_joints: int, num_betas: int) -> SmplParams:
    return SmplParams(
        betas=np.zeros(num_betas),
        pose=np.zeros((num_frames, num_joints, 3)),
        transl=np.zeros((num_frames, 3)),
    )


def test_zero_pose_zero_betas_recovers_rest_joints() -> None:
    model = SmplModel(synthetic_model())
    j, b = model.body.num_joints, model.body.num_betas
    params = _zero_params(1, j, b)

    rest = model.rest_joints(np.zeros(b))
    # Zero betas => rest joints are just the regressor applied to the template.
    np.testing.assert_allclose(rest, model.body.j_regressor @ model.body.v_template)

    joints = model.forward_joints(params)
    np.testing.assert_allclose(joints[0], rest)

    transforms = model.forward_transforms(params)
    eye = np.broadcast_to(np.eye(3), (j, 3, 3))
    np.testing.assert_allclose(transforms[0, :, :3, :3], eye, atol=1e-12)
    np.testing.assert_allclose(transforms[0, :, :3, 3], rest)


def test_translation_only_shifts_every_joint() -> None:
    model = SmplModel(synthetic_model())
    j, b = model.body.num_joints, model.body.num_betas
    t = np.array([0.3, -1.2, 4.0])

    params = SmplParams(
        betas=np.zeros(b),
        pose=np.zeros((1, j, 3)),
        transl=t[None, :],
    )

    rest = model.rest_joints(np.zeros(b))
    joints = model.forward_joints(params)
    np.testing.assert_allclose(joints[0], rest + t)


def test_root_rotation_rotates_whole_tree_rigidly() -> None:
    model = SmplModel(synthetic_model())
    j, b = model.body.num_joints, model.body.num_betas

    angle = np.pi / 2.0  # 90 degrees about +z
    pose = np.zeros((1, j, 3))
    pose[0, 0] = np.array([0.0, 0.0, angle])
    t = np.array([0.5, 0.25, -0.75])
    params = SmplParams(betas=np.zeros(b), pose=pose, transl=t[None, :])

    rest = model.rest_joints(np.zeros(b))  # root at origin in the synthetic model
    rz = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
    expected = rest @ rz.T + t  # (J, 3): Rz @ rest[k] for each k, then translate

    joints = model.forward_joints(params)
    np.testing.assert_allclose(joints[0], expected, atol=1e-12)


def test_non_root_rotation_moves_subtree_only() -> None:
    model = SmplModel(synthetic_model())  # chain 0 <- 1 <- 2
    j, b = model.body.num_joints, model.body.num_betas
    assert j == 3

    rest_params = _zero_params(1, j, b)
    rest_joints = model.forward_joints(rest_params)[0]

    pose = np.zeros((1, j, 3))
    pose[0, 1] = np.array([0.0, 0.0, np.pi / 3.0])  # rotate the middle joint
    params = SmplParams(betas=np.zeros(b), pose=pose, transl=np.zeros((1, 3)))
    joints = model.forward_joints(params)[0]

    # Ancestor (root, joint 0) is unaffected.
    np.testing.assert_allclose(joints[0], rest_joints[0], atol=1e-12)
    # Joint 1 is the pivot itself: stays put.
    np.testing.assert_allclose(joints[1], rest_joints[1], atol=1e-12)
    # Descendant (joint 2) is displaced.
    assert not np.allclose(joints[2], rest_joints[2])

    # The descendant rotates rigidly about its parent joint 1's rest position.
    rz = Rotation.from_rotvec([0.0, 0.0, np.pi / 3.0]).as_matrix()
    expected_child = rz @ (rest_joints[2] - rest_joints[1]) + rest_joints[1]
    np.testing.assert_allclose(joints[2], expected_child, atol=1e-12)


def test_forward_output_shapes() -> None:
    model = SmplModel(synthetic_model())
    j, b = model.body.num_joints, model.body.num_betas
    num_frames = 5
    params = _zero_params(num_frames, j, b)

    assert model.forward_joints(params).shape == (num_frames, j, 3)
    assert model.forward_transforms(params).shape == (num_frames, j, 4, 4)
    assert model.joint_names == model.body.joint_names
    assert len(model.joint_names) == j


def test_params_shape_validation_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="betas"):
        SmplParams(betas=np.zeros((2, 2)), pose=np.zeros((1, 3, 3)), transl=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="pose"):
        SmplParams(betas=np.zeros(2), pose=np.zeros((1, 3, 2)), transl=np.zeros((1, 3)))

    with pytest.raises(ValueError, match="transl"):
        SmplParams(betas=np.zeros(2), pose=np.zeros((1, 3, 3)), transl=np.zeros((1, 2)))

    with pytest.raises(ValueError, match="frame count"):
        SmplParams(betas=np.zeros(2), pose=np.zeros((4, 3, 3)), transl=np.zeros((2, 3)))


def test_model_rejects_param_joint_mismatch() -> None:
    model = SmplModel(synthetic_model(num_joints=3))
    params = _zero_params(1, 4, model.body.num_betas)
    with pytest.raises(ValueError, match="joints"):
        model.forward_joints(params)
