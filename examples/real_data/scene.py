from __future__ import annotations

from pathlib import Path

from retarget.io import read_marker_positions_from_vsk

from retarget.core import (
    Subject,
    Segment,
    Patch,
    Marker,
    SemanticAxis,
    calibrate_patch_transform,
)

from .schema import (
    ExampleSubjects,
    LeftFootSegments,
    LeftShoePatches,
    LeftShoeMarkers,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VSK_PATH = REPO_ROOT / "models" / "Left_Shoe_Improved.vsk"

# VSK-derived segment-frame marker positions keyed by Vicon marker name.
BODY_MODEL = read_marker_positions_from_vsk(VSK_PATH)

# Offset from the fitted calibration-marker plane to the physical sole contact
# plane, applied along the fitted patch normal after plane fitting.
SOLE_PLANE_NORMAL_OFFSET = -0.010


SOLE_TRANSFORM = calibrate_patch_transform(
    marker_positions_segment=BODY_MODEL,
    markers=("plane_rear", "plane_inner", "plane_outer"),
    normal_offset=SOLE_PLANE_NORMAL_OFFSET,
    outward_axis=SemanticAxis.UP,
    x_axis=SemanticAxis.FORWARD,
)

# Marker definitions (e.g. Marker(mocap_name))
heel = Marker("heel")
toe = Marker("toe")
heel_inner_1 = Marker("heel_inner_1")
heel_inner_2 = Marker("heel_inner_2")
heel_outer_1 = Marker("heel_outer_1")
heel_outer_2 = Marker("heel_outer_2")
toe_inner = Marker("toe_inner")
toe_outer = Marker("toe_outer")
toe_grid_1 = Marker("toe_grid_1")
toe_grid_2 = Marker("toe_grid_2")
toe_grid_3 = Marker("toe_grid_3")
toe_grid_4 = Marker("toe_grid_4")
sole_inner = Marker("sole_inner")
sole_outer = Marker("sole_outer")
plane_rear = Marker("plane_rear")
plane_inner = Marker("plane_inner")
plane_outer = Marker("plane_outer")

left_shoe_markers = LeftShoeMarkers(
    heel=heel, toe=toe,
    heel_inner_1=heel_inner_1, heel_inner_2=heel_inner_2,
    heel_outer_1=heel_outer_1, heel_outer_2=heel_outer_2,
    toe_inner=toe_inner, toe_outer=toe_outer,
    toe_grid_1=toe_grid_1, toe_grid_2=toe_grid_2,
    toe_grid_3=toe_grid_3, toe_grid_4=toe_grid_4,
    sole_inner=sole_inner, sole_outer=sole_outer,
    plane_rear=plane_rear, plane_inner=plane_inner, plane_outer=plane_outer,
)

# Patch definitions
sole_patch = Patch.rectangular(
    label="sole",
    transform_segment_patch=SOLE_TRANSFORM,
    width=0.10,
    height=0.25,
    frame="sole_frame",
)

left_shoe_patches = LeftShoePatches(
    sole=sole_patch,
)

# Segment definitions
left_shoe = Segment(
    markers=left_shoe_markers,
    patches=left_shoe_patches,
    mocap_name="Left_Shoe_Improved",
)

left_foot_segments = LeftFootSegments(shoe=left_shoe)

# Subject definitions
left_foot = Subject(
    segments=left_foot_segments,
    body_model=BODY_MODEL,
    mocap_name="Left_Shoe_Improved",
)

SUBJECTS = ExampleSubjects(left_foot=left_foot)