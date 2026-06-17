from __future__ import annotations

from retarget.core import MarkerRole, PoseFormat, QuaternionOrder, RotationFormat


def test_rotation_format_values_exist() -> None:
    assert RotationFormat.MATRIX.value == "matrix"
    assert RotationFormat.QUATERNION_XYZW.value == "quaternion_xyzw"
    assert RotationFormat.QUATERNION_WXYZ.value == "quaternion_wxyz"
    assert RotationFormat.ROTVEC.value == "rotvec"


def test_pose_format_values_exist() -> None:
    assert PoseFormat.RIGID_TRANSFORM.value == "rigid_transform"
    assert PoseFormat.MATRIX_4X4.value == "matrix_4x4"
    assert PoseFormat.TRANSLATION_QUATERNION_XYZW.value == "translation_quaternion_xyzw"
    assert PoseFormat.TRANSLATION_ROTATION_MATRIX.value == "translation_rotation_matrix"


def test_quaternion_order_values_exist() -> None:
    assert QuaternionOrder.WXYZ.value == "wxyz"
    assert QuaternionOrder.XYZW.value == "xyzw"


def test_marker_role_values_exist() -> None:
    assert MarkerRole.TRACKING.value == "tracking"
    assert MarkerRole.CALIBRATION.value == "calibration"
    assert MarkerRole.TRACKING_AND_CALIBRATION.value == "tracking_and_calibration"
