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


SUBJECTS = ExampleSubjects(
    left_foot=Subject(
        vicon_name="Left_Shoe_Improved",
        body_model=BODY_MODEL,
        segments=LeftFootSegments(
            shoe=Segment(
                vicon_name="Left_Shoe_Improved",
                markers=LeftShoeMarkers(
                    heel=Marker(vicon_name="heel"),
                    toe=Marker(vicon_name="toe"),
                    heel_inner_1=Marker(vicon_name="heel_inner_1"),
                    heel_inner_2=Marker(vicon_name="heel_inner_2"),
                    heel_outer_1=Marker(vicon_name="heel_outer_1"),
                    heel_outer_2=Marker(vicon_name="heel_outer_2"),
                    toe_inner=Marker(vicon_name="toe_inner"),
                    toe_outer=Marker(vicon_name="toe_outer"),
                    toe_grid_1=Marker(vicon_name="toe_grid_1"),
                    toe_grid_2=Marker(vicon_name="toe_grid_2"),
                    toe_grid_3=Marker(vicon_name="toe_grid_3"),
                    toe_grid_4=Marker(vicon_name="toe_grid_4"),
                    sole_inner=Marker(vicon_name="sole_inner"),
                    sole_outer=Marker(vicon_name="sole_outer"),
                    plane_rear=Marker(vicon_name="plane_rear"),
                    plane_inner=Marker(vicon_name="plane_inner"),
                    plane_outer=Marker(vicon_name="plane_outer"),
                ),
                patches=LeftShoePatches(
                    sole=Patch.rectangular(
                        label="sole",
                        transform_segment_patch=SOLE_TRANSFORM,
                        width=0.10,
                        height=0.25,
                        frame="sole_frame",
                    )
                )
            )
        )
    )
)