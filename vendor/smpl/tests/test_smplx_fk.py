from __future__ import annotations

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from smpl import SmplxModel, SmplxParams, synthetic_smplx_model
from smpl.params.smplx import (
    SMPLX_NUM_BODY_JOINTS,
    SMPLX_NUM_HAND_JOINTS,
    SMPLX_NUM_JOINTS,
)


def test_full_pose_assembles_canonical_layout() -> None:
    params = SmplxParams.zeros(4)
    pose = params.full_pose()
    assert pose.shape == (4, SMPLX_NUM_JOINTS, 3)
    assert SMPLX_NUM_JOINTS == 55
    assert params.num_joints == SMPLX_NUM_JOINTS


def test_full_pose_places_each_articulation_in_order() -> None:
    params = SmplxParams.zeros(1)
    # Stamp a distinct nonzero rotation into each block, then read it back from the assembled pose.
    object.__setattr__(params, "global_orient", np.array([[0.1, 0.0, 0.0]]))
    object.__setattr__(params, "body_pose", np.full((1, SMPLX_NUM_BODY_JOINTS, 3), 0.2))
    object.__setattr__(params, "jaw_pose", np.array([[0.3, 0.0, 0.0]]))
    object.__setattr__(params, "leye_pose", np.array([[0.4, 0.0, 0.0]]))
    object.__setattr__(params, "reye_pose", np.array([[0.5, 0.0, 0.0]]))
    object.__setattr__(params, "left_hand_pose", np.full((1, SMPLX_NUM_HAND_JOINTS, 3), 0.6))
    object.__setattr__(params, "right_hand_pose", np.full((1, SMPLX_NUM_HAND_JOINTS, 3), 0.7))

    pose = params.full_pose()
    np.testing.assert_array_equal(pose[0, 0], [0.1, 0.0, 0.0])
    np.testing.assert_array_equal(pose[0, 1 : 1 + SMPLX_NUM_BODY_JOINTS], np.full((SMPLX_NUM_BODY_JOINTS, 3), 0.2))
    jaw = 1 + SMPLX_NUM_BODY_JOINTS
    np.testing.assert_array_equal(pose[0, jaw], [0.3, 0.0, 0.0])
    np.testing.assert_array_equal(pose[0, jaw + 1], [0.4, 0.0, 0.0])
    np.testing.assert_array_equal(pose[0, jaw + 2], [0.5, 0.0, 0.0])
    lhand = jaw + 3
    np.testing.assert_array_equal(pose[0, lhand : lhand + SMPLX_NUM_HAND_JOINTS], np.full((SMPLX_NUM_HAND_JOINTS, 3), 0.6))
    np.testing.assert_array_equal(pose[0, lhand + SMPLX_NUM_HAND_JOINTS :], np.full((SMPLX_NUM_HAND_JOINTS, 3), 0.7))


def test_zero_pose_zero_betas_recovers_rest_joints() -> None:
    model = SmplxModel(synthetic_smplx_model())
    params = SmplxParams.zeros(1, num_betas=model.body.num_betas)

    rest = model.rest_joints(np.zeros(model.body.num_betas))
    np.testing.assert_allclose(rest, model.body.j_regressor @ model.body.v_template)

    joints = model.forward_joints(params)
    assert joints.shape == (1, SMPLX_NUM_JOINTS, 3)
    np.testing.assert_allclose(joints[0], rest)


def test_translation_only_shifts_every_joint() -> None:
    model = SmplxModel(synthetic_smplx_model())
    params = SmplxParams.zeros(1, num_betas=model.body.num_betas)
    t = np.array([0.3, -1.2, 4.0])
    object.__setattr__(params, "transl", t[None, :])

    rest = model.rest_joints(np.zeros(model.body.num_betas))
    joints = model.forward_joints(params)
    np.testing.assert_allclose(joints[0], rest + t)


def test_root_rotation_rotates_whole_tree_rigidly() -> None:
    model = SmplxModel(synthetic_smplx_model())
    params = SmplxParams.zeros(1, num_betas=model.body.num_betas)

    angle = np.pi / 2.0  # 90 degrees about +z, applied through global_orient
    object.__setattr__(params, "global_orient", np.array([[0.0, 0.0, angle]]))
    t = np.array([0.5, 0.25, -0.75])
    object.__setattr__(params, "transl", t[None, :])

    rest = model.rest_joints(np.zeros(model.body.num_betas))  # root at origin
    rz = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
    expected = rest @ rz.T + t

    joints = model.forward_joints(params)
    np.testing.assert_allclose(joints[0], expected, atol=1e-12)


def test_forward_output_shapes() -> None:
    model = SmplxModel(synthetic_smplx_model())
    num_frames = 3
    params = SmplxParams.zeros(num_frames, num_betas=model.body.num_betas)

    assert model.forward_joints(params).shape == (num_frames, SMPLX_NUM_JOINTS, 3)
    assert model.forward_transforms(params).shape == (num_frames, SMPLX_NUM_JOINTS, 4, 4)
    assert model.joint_names == model.body.joint_names
    assert len(model.joint_names) == SMPLX_NUM_JOINTS


def test_params_shape_validation_rejects_bad_input() -> None:
    good = SmplxParams.zeros(2)

    with pytest.raises(ValueError, match="body_pose"):
        SmplxParams(
            betas=good.betas,
            global_orient=good.global_orient,
            body_pose=np.zeros((2, SMPLX_NUM_BODY_JOINTS + 1, 3)),
            jaw_pose=good.jaw_pose,
            leye_pose=good.leye_pose,
            reye_pose=good.reye_pose,
            left_hand_pose=good.left_hand_pose,
            right_hand_pose=good.right_hand_pose,
            expression=good.expression,
            transl=good.transl,
        )

    with pytest.raises(ValueError, match="frame count"):
        SmplxParams(
            betas=good.betas,
            global_orient=np.zeros((4, 3)),
            body_pose=np.zeros((2, SMPLX_NUM_BODY_JOINTS, 3)),
            jaw_pose=good.jaw_pose,
            leye_pose=good.leye_pose,
            reye_pose=good.reye_pose,
            left_hand_pose=good.left_hand_pose,
            right_hand_pose=good.right_hand_pose,
            expression=good.expression,
            transl=good.transl,
        )

    with pytest.raises(ValueError, match="global_orient"):
        SmplxParams(
            betas=good.betas,
            global_orient=np.zeros((2, 2)),
            body_pose=good.body_pose,
            jaw_pose=good.jaw_pose,
            leye_pose=good.leye_pose,
            reye_pose=good.reye_pose,
            left_hand_pose=good.left_hand_pose,
            right_hand_pose=good.right_hand_pose,
            expression=good.expression,
            transl=good.transl,
        )


def test_model_rejects_param_joint_mismatch() -> None:
    from smpl import synthetic_model

    # A SMPL-X model wrapping a too-small body: the assembled 55-joint pose trips the kernel guard.
    wrong_size = SmplxModel(synthetic_model(num_joints=10, num_betas=2))
    params = SmplxParams.zeros(1, num_betas=2)
    with pytest.raises(ValueError, match="joints"):
        wrong_size.forward_joints(params)
