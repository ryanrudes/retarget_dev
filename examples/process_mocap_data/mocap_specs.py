from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from mocap_vocab import (
    LeftShoeMarkerId,
    LeftShoePatchId,
    LeftShoeSegmentId,
    ViconSubjectId,
)

from retarget.core import (
    MarkerRole,
    MarkerSetSpec,
    PatchCalibrationSpec,
    RectangularRegion,
    SceneSpec,
    SegmentSpec,
    SemanticAxis,
    SubjectSpec,
    Z_UP_AXES,
    Vec3,
)


# Define the subject spec for the left shoe Vicon subject
@dataclass(frozen=True, slots=True)
class LeftShoeSubjectSpec(SubjectSpec):
    """Concrete subject spec for the left shoe Vicon subject."""

    left_shoe: SegmentSpec[LeftShoeMarkerId, LeftShoePatchId]

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        yield self.left_shoe


# Define the scene spec for the Vicon scene
@dataclass(frozen=True, slots=True)
class ViconSceneSpec(SceneSpec):
    """Concrete scene spec for this mocap example."""

    left_shoe: LeftShoeSubjectSpec

    def iter_subjects(self) -> Iterable[SubjectSpec]:
        yield self.left_shoe


# Define the marker set spec for the left shoe segment
LEFT_SHOE_MARKERS = MarkerSetSpec(
    marker_type=LeftShoeMarkerId,
    default_role=MarkerRole.TRACKING,
    roles={
        LeftShoeMarkerId.PLANE_REAR: MarkerRole.CALIBRATION,
        LeftShoeMarkerId.PLANE_INNER: MarkerRole.CALIBRATION,
        LeftShoeMarkerId.PLANE_OUTER: MarkerRole.CALIBRATION,
    },
)


LEFT_SHOE_MARKER_POSITIONS_SEGMENT: dict[LeftShoeMarkerId, Vec3] = {
    # Toe markers
    LeftShoeMarkerId.TOE: np.array([0.120, 0.000, 0.040]),
    LeftShoeMarkerId.TOE_INNER: np.array([0.105, 0.035, 0.035]),
    LeftShoeMarkerId.TOE_OUTER: np.array([0.105, -0.035, 0.035]),
    LeftShoeMarkerId.TOE_GRID_1: np.array([0.080, 0.025, 0.035]),
    LeftShoeMarkerId.TOE_GRID_2: np.array([0.080, -0.025, 0.035]),
    LeftShoeMarkerId.TOE_GRID_3: np.array([0.050, 0.025, 0.035]),
    LeftShoeMarkerId.TOE_GRID_4: np.array([0.050, -0.025, 0.035]),

    # Heel markers
    LeftShoeMarkerId.HEEL: np.array([-0.100, 0.000, 0.045]),
    LeftShoeMarkerId.HEEL_INNER_1: np.array([-0.085, 0.035, 0.045]),
    LeftShoeMarkerId.HEEL_INNER_2: np.array([-0.115, 0.035, 0.045]),
    LeftShoeMarkerId.HEEL_OUTER_1: np.array([-0.085, -0.035, 0.045]),
    LeftShoeMarkerId.HEEL_OUTER_2: np.array([-0.115, -0.035, 0.045]),

    # Sole side markers
    LeftShoeMarkerId.SOLE_INNER: np.array([0.000, 0.040, 0.020]),
    LeftShoeMarkerId.SOLE_OUTER: np.array([0.000, -0.040, 0.020]),

    # Sole plane calibration markers.
    # These should be replaced with your calibrated segment-frame positions.
    LeftShoeMarkerId.PLANE_REAR: np.array([-0.090, 0.000, 0.000]),
    LeftShoeMarkerId.PLANE_INNER: np.array([0.040, 0.045, 0.000]),
    LeftShoeMarkerId.PLANE_OUTER: np.array([0.040, -0.045, 0.000]),
}


# Offset from the fitted calibration-marker plane to the physical sole contact
# plane, applied along the fitted patch normal after plane fitting.
#
# CAD/measured marker-to-surface offsets can instead be supplied via
# marker_translations with BodyFrameTranslation vectors from each marker
# center to the corresponding contact point on the shoe sole.
SOLE_PLANE_NORMAL_OFFSET = -0.010

# Define the sole patch calibration spec for the left shoe segment
LEFT_SHOE_SOLE_CALIBRATION = PatchCalibrationSpec(
    patch=LeftShoePatchId.SOLE,
    markers=(
        LeftShoeMarkerId.PLANE_REAR,
        LeftShoeMarkerId.PLANE_INNER,
        LeftShoeMarkerId.PLANE_OUTER,
    ),
    normal_offset=SOLE_PLANE_NORMAL_OFFSET,
    region=RectangularRegion(
        width=0.10,
        height=0.25,
    ),
    outward_axis=SemanticAxis.UP,
    x_axis=SemanticAxis.FORWARD,
)


# Define the left shoe segment spec for the left shoe segment
LEFT_SHOE_SEGMENT: SegmentSpec[LeftShoeMarkerId, LeftShoePatchId] = (
    SegmentSpec(
        segment=LeftShoeSegmentId.LEFT_SHOE,
        marker_type=LeftShoeMarkerId,
        patch_type=LeftShoePatchId,
        axis_convention=Z_UP_AXES,
        marker_set=LEFT_SHOE_MARKERS,
        marker_positions_segment=LEFT_SHOE_MARKER_POSITIONS_SEGMENT,
        patch_calibrations={
            LeftShoePatchId.SOLE: LEFT_SHOE_SOLE_CALIBRATION,
        },
    )
    .with_built_patches()
)


# Define the left shoe subject spec for the left shoe subject
LEFT_SHOE_SUBJECT = LeftShoeSubjectSpec(
    subject=ViconSubjectId.LEFT_SHOE,
    left_shoe=LEFT_SHOE_SEGMENT,
)


# Define the Vicon scene spec for the Vicon scene
VICON_SCENE = ViconSceneSpec(left_shoe=LEFT_SHOE_SUBJECT)