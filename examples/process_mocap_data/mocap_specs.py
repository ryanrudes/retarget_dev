"""Backend-oriented Vicon scene-spec construction used by the loader example."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
)
from retarget.io import read_marker_positions_from_vsk


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


LEFT_SHOE_MARKER_POSITIONS_SEGMENT = read_marker_positions_from_vsk(
    Path("models/Left_Shoe_Improved.vsk"),
    marker_type=LeftShoeMarkerId,
)


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


# Low-level backend scene-spec assembly for the left shoe segment.
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
