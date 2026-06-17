"""Stable value enums used by public configuration and result objects.

This module intentionally contains only *value* enums (formats, roles). Scene
identity (subjects, segments, markers, patches, tracks) is expressed with plain
authored ``str`` names compiled from the typed ``TypedDict`` schema, not with
identifier enums.
"""

from __future__ import annotations

from enum import StrEnum


class MarkerRole(StrEnum):
    """The role of a marker in the tracking system."""

    TRACKING = "tracking"
    """A marker used for tracking."""

    CALIBRATION = "calibration"
    """A marker used for calibration."""

    TRACKING_AND_CALIBRATION = "tracking_and_calibration"
    """A marker used for both tracking and calibration."""


class RotationFormat(StrEnum):
    """Rotation representation for time-series queries."""

    MATRIX = "matrix"
    QUATERNION_XYZW = "quaternion_xyzw"
    QUATERNION_WXYZ = "quaternion_wxyz"
    ROTVEC = "rotvec"


class PoseFormat(StrEnum):
    """Pose representation for time-series queries."""

    RIGID_TRANSFORM = "rigid_transform"
    MATRIX_4X4 = "matrix_4x4"
    TRANSLATION_QUATERNION_XYZW = "translation_quaternion_xyzw"
    TRANSLATION_ROTATION_MATRIX = "translation_rotation_matrix"


class QuaternionOrder(StrEnum):
    """Quaternion storage order.

    Attributes:
        WXYZ (str): Scalar-first ``(w, x, y, z)`` layout.
        XYZW (str): Scalar-last ``(x, y, z, w)`` layout (common in graphics APIs).
    """

    WXYZ = "wxyz"
    XYZW = "xyzw"
