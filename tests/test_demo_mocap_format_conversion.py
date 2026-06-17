from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from conftest import DemoSegmentId, DemoSubjectId, make_mocap_track
from retarget.core import PoseFormat, RotationFormat
from retarget.demo._mocap_arrays import pose_arrays_to_format, rotation_matrices_to_format


def _identity_batch(num_timesteps: int) -> np.ndarray:
    return np.tile(np.eye(3), (num_timesteps, 1, 1))


def _z90_batch(num_timesteps: int) -> np.ndarray:
    matrix = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    return np.tile(matrix, (num_timesteps, 1, 1))


def test_rotation_matrices_to_format_matrix_shape() -> None:
    matrices = _identity_batch(3)
    result = rotation_matrices_to_format(matrices, format=RotationFormat.MATRIX)
    assert result.shape == (3, 3, 3)
    np.testing.assert_allclose(result, matrices)


def test_rotation_matrices_to_format_quaternion_xyzw_shape_and_order() -> None:
    matrices = _z90_batch(2)
    result = rotation_matrices_to_format(matrices, format=RotationFormat.QUATERNION_XYZW)
    expected = Rotation.from_matrix(matrices).as_quat()
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result[0, 3], 0.70710678, rtol=1e-5)


def test_rotation_matrices_to_format_quaternion_wxyz_shape_and_order() -> None:
    matrices = _z90_batch(2)
    result = rotation_matrices_to_format(matrices, format=RotationFormat.QUATERNION_WXYZ)
    xyzw = Rotation.from_matrix(matrices).as_quat()
    expected = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result[0, 0], 0.70710678, rtol=1e-5)


def test_rotation_matrices_to_format_rotvec_shape() -> None:
    matrices = _z90_batch(2)
    result = rotation_matrices_to_format(matrices, format=RotationFormat.ROTVEC)
    expected = Rotation.from_matrix(matrices).as_rotvec()
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result, expected)


def test_pose_arrays_to_format_matrix_shape() -> None:
    translations = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rotations = _identity_batch(2)
    result = pose_arrays_to_format(
        translations,
        rotations,
        format=PoseFormat.MATRIX_4X4,
    )
    assert result.shape == (2, 4, 4)
    np.testing.assert_allclose(result[0, :3, 3], translations[0])
    np.testing.assert_allclose(result[0, 3, :], [0.0, 0.0, 0.0, 1.0])


def test_pose_arrays_to_format_translation_quaternion_shape() -> None:
    translations = np.array([[1.0, 2.0, 3.0]])
    rotations = _z90_batch(1)
    result = pose_arrays_to_format(
        translations,
        rotations,
        format=PoseFormat.TRANSLATION_QUATERNION_XYZW,
    )
    quat = Rotation.from_matrix(rotations).as_quat()
    expected = np.concatenate([translations, quat], axis=1)
    assert result.shape == (1, 7)
    np.testing.assert_allclose(result, expected)


def test_pose_arrays_to_format_translation_rotation_matrix_shape() -> None:
    translations = np.array([[1.0, 2.0, 3.0]])
    rotations = _identity_batch(1)
    result = pose_arrays_to_format(
        translations,
        rotations,
        format=PoseFormat.TRANSLATION_ROTATION_MATRIX,
    )
    flat_rot = rotations.reshape(1, 9)
    expected = np.concatenate([translations, flat_rot], axis=1)
    assert result.shape == (1, 12)
    np.testing.assert_allclose(result, expected)


def test_rotation_matrices_to_format_empty_shapes() -> None:
    empty = np.empty((0, 3, 3), dtype=np.float64)
    assert rotation_matrices_to_format(empty, format=RotationFormat.MATRIX).shape == (0, 3, 3)
    assert rotation_matrices_to_format(
        empty,
        format=RotationFormat.QUATERNION_XYZW,
    ).shape == (0, 4)
    assert rotation_matrices_to_format(
        empty,
        format=RotationFormat.QUATERNION_WXYZ,
    ).shape == (0, 4)
    assert rotation_matrices_to_format(empty, format=RotationFormat.ROTVEC).shape == (0, 3)


def test_pose_arrays_to_format_empty_shapes() -> None:
    empty_trans = np.empty((0, 3), dtype=np.float64)
    empty_rot = np.empty((0, 3, 3), dtype=np.float64)
    assert pose_arrays_to_format(
        empty_trans,
        empty_rot,
        format=PoseFormat.RIGID_TRANSFORM,
    ) == ()
    assert pose_arrays_to_format(
        empty_trans,
        empty_rot,
        format=PoseFormat.MATRIX_4X4,
    ).shape == (0, 4, 4)
    assert pose_arrays_to_format(
        empty_trans,
        empty_rot,
        format=PoseFormat.TRANSLATION_QUATERNION_XYZW,
    ).shape == (0, 7)
    assert pose_arrays_to_format(
        empty_trans,
        empty_rot,
        format=PoseFormat.TRANSLATION_ROTATION_MATRIX,
    ).shape == (0, 12)


def test_rotation_matrices_to_format_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        rotation_matrices_to_format(np.zeros((3, 3)), format=RotationFormat.MATRIX)


def test_segment_view_rotations_and_poses_use_format_helpers() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    assert segment.rotations().shape == (3, 3, 3)
    assert segment.rotations(format=RotationFormat.QUATERNION_XYZW).shape == (3, 4)
    assert segment.rotations(format=RotationFormat.QUATERNION_WXYZ).shape == (3, 4)
    assert segment.rotations(format=RotationFormat.ROTVEC).shape == (3, 3)
    assert segment.poses(format=PoseFormat.MATRIX_4X4).shape == (3, 4, 4)
    assert segment.poses(format=PoseFormat.TRANSLATION_QUATERNION_XYZW).shape == (3, 7)
    assert segment.poses(format=PoseFormat.TRANSLATION_ROTATION_MATRIX).shape == (3, 12)
    assert len(segment.poses()) == 3


def test_empty_slice_pose_format_shapes() -> None:
    segment = make_mocap_track().slice_time(100.0, 101.0)._subject(DemoSubjectId.SUBJECT)._segment(
        DemoSegmentId.SEGMENT
    )
    assert segment.poses() == ()
    assert segment.poses(format=PoseFormat.MATRIX_4X4).shape == (0, 4, 4)
    assert segment.poses(format=PoseFormat.TRANSLATION_QUATERNION_XYZW).shape == (0, 7)
    assert segment.poses(format=PoseFormat.TRANSLATION_ROTATION_MATRIX).shape == (0, 12)
    assert segment.rotations(format=RotationFormat.ROTVEC).shape == (0, 3)
