from __future__ import annotations

from retarget.core import NameId, PoseFormat, RotationFormat, TrackId


class _DemoTrackId(TrackId):
    A = "a"


def test_track_id_behaves_like_name_id() -> None:
    assert issubclass(TrackId, NameId)
    assert _DemoTrackId.A.label == "a"
    assert _DemoTrackId.A.index == 0


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
