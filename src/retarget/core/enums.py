"""Stable enums used by public configuration and result objects."""

from __future__ import annotations

from enum import StrEnum


class NameId(StrEnum):
    """Base class for user-defined symbolic identifiers."""

    _index: int

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        for i, member in enumerate(cls):
            member._index = i
    
    @property
    def label(self) -> str:
        """External/user-facing label of the member in the vocabulary."""
        return self.value
    
    @property
    def index(self) -> int:
        """Index of the member in the vocabulary."""
        return self._index
    
    @classmethod
    def size(cls) -> int:
        """Number of members in the vocabulary."""
        return len(cls)
    
    @classmethod
    def members(cls) -> tuple[NameId, ...]:
        """All members in declaration order."""
        return tuple(cls)
    
    @classmethod
    def labels(cls) -> list[str]:
        """List of all labels in the vocabulary.
        
        This is useful for debugging and visualization, but should not
        be used for programmatic access in order to maintain type safety.
        """
        return [marker.label for marker in cls]


class SubjectId(NameId):
    """Vocabulary for the subjects which make up a scene."""


class SegmentId(NameId):
    """Vocabulary for the segments which make up a subject."""


class MarkerId(NameId):
    """Vocabulary for the markers which make up a segment."""


class PatchId(NameId):
    """Vocabulary for the contact patches which make up a segment."""


class TrackId(NameId):
    """Base class for user-defined demonstration track identifiers."""


class MarkerRole(StrEnum):
    """The role of a marker in the tracking system."""

    TRACKING = "tracking"
    """A marker used for tracking."""

    CALIBRATION = "calibration"
    """A marker used for calibration."""

    TRACKING_AND_CALIBRATION = "tracking_and_calibration"
    """A marker used for both tracking and calibration."""


class RotationFormat(NameId):
    """Rotation representation for time-series queries."""

    MATRIX = "matrix"
    QUATERNION_XYZW = "quaternion_xyzw"
    QUATERNION_WXYZ = "quaternion_wxyz"
    ROTVEC = "rotvec"


class PoseFormat(NameId):
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