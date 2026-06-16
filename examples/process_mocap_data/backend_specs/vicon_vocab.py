"""Backend/manual Vicon vocabularies for the real calibration scene."""

from __future__ import annotations

from retarget.core import MarkerId, PatchId, SegmentId, SubjectId


class ViconSubjectId(SubjectId):
    """Vocabulary for the subjects which make up the backend/manual scene."""

    LEFT_SHOE = "Left_Shoe_Improved"
    """Symbol representing the left shoe subject."""


class LeftShoeSegmentId(SegmentId):
    """Vocabulary for the segments which make up the left shoe subject."""

    LEFT_SHOE = "Left_Shoe_Improved"
    """Symbol representing the left shoe segment."""


class LeftShoePatchId(PatchId):
    """Vocabulary for the contact patches attached to the left shoe segment."""

    SOLE = "sole"
    """The sole patch."""

    TOE = "toe"
    """The toe patch."""

    HEEL = "heel"
    """The heel patch."""


class LeftShoeMarkerId(MarkerId):
    """Vocabulary for the markers which make up the left shoe segment."""

    # Markers affixed to the toe of the shoe.
    TOE = "toe"
    TOE_INNER = "toe_inner"
    TOE_OUTER = "toe_outer"
    TOE_GRID_1 = "toe_grid_1"
    TOE_GRID_2 = "toe_grid_2"
    TOE_GRID_3 = "toe_grid_3"
    TOE_GRID_4 = "toe_grid_4"

    # Markers affixed to the heel of the shoe.
    HEEL = "heel"
    HEEL_INNER_1 = "heel_inner_1"
    HEEL_INNER_2 = "heel_inner_2"
    HEEL_OUTER_1 = "heel_outer_1"
    HEEL_OUTER_2 = "heel_outer_2"

    # Markers affixed to the side of the sole of the shoe.
    SOLE_INNER = "sole_inner"
    SOLE_OUTER = "sole_outer"

    # Shoe sole plane surface calibration markers.
    PLANE_REAR = "plane_rear"
    PLANE_INNER = "plane_inner"
    PLANE_OUTER = "plane_outer"
