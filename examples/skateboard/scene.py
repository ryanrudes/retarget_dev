from __future__ import annotations

from pathlib import Path

from retarget.io import read_marker_positions_from_vsk

from retarget.demo import (
    MocapTrack,
    Demonstration,
)

from retarget.core import (
    Subject,
    Segment,
    Patch,
    Marker,
    SemanticAxis,
    bounding_box,
)

from schema import (
    ExampleTracks,
    ExampleSubjects,
    LeftFootSegments,
    LeftShoePatches,
    LeftShoeMarkers,
    SkateboardSegments,
    SkateboardPatches,
    SkateboardMarkers,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNBAGGED_DIR = REPO_ROOT / "bags" / "skateboard_glide" / "unbagged"

LEFT_SHOE_VSK_PATH = REPO_ROOT / "models" / "Left_Shoe_Improved.vsk"
SKATEBOARD_VSK_PATH = REPO_ROOT / "models" / "Skateboard.vsk"

# VSK-derived segment-frame marker positions keyed by Vicon marker name.
LEFT_SHOE_BODY_MODEL = read_marker_positions_from_vsk(LEFT_SHOE_VSK_PATH)
SKATEBOARD_BODY_MODEL = read_marker_positions_from_vsk(SKATEBOARD_VSK_PATH)

# Offset from the fitted calibration-marker plane to the physical sole contact
# plane, applied along the fitted patch normal after plane fitting.
SOLE_PLANE_NORMAL_OFFSET = -0.010

# The skateboard standing surface sits 35mm above the plane fit through the eight
# pole markers, along that plane's upward normal.
SKATEBOARD_SURFACE_NORMAL_OFFSET = 0.035

# The eight "pole" markers whose plane fit defines the skateboard deck plane.
SKATEBOARD_POLE_MARKERS = (
    "front_left_pole_left",
    "front_left_pole_right",
    "front_right_pole_left",
    "front_right_pole_right",
    "rear_left_pole_left",
    "rear_left_pole_right",
    "rear_right_pole_left",
    "rear_right_pole_right",
)


def get_left_shoe_markers() -> LeftShoeMarkers:
    return LeftShoeMarkers(
        heel=Marker("heel"),
        toe=Marker("toe"),
        heel_inner_1=Marker("heel_inner_1"),
        heel_inner_2=Marker("heel_inner_2"),
        heel_outer_1=Marker("heel_outer_1"),
        heel_outer_2=Marker("heel_outer_2"),
        toe_inner=Marker("toe_inner"),
        toe_outer=Marker("toe_outer"),
        toe_grid_1=Marker("toe_grid_1"),
        toe_grid_2=Marker("toe_grid_2"),
        toe_grid_3=Marker("toe_grid_3"),
        toe_grid_4=Marker("toe_grid_4"),
        sole_inner=Marker("sole_inner"),
        sole_outer=Marker("sole_outer"),
        plane_rear=Marker("plane_rear"),
        plane_inner=Marker("plane_inner"),
        plane_outer=Marker("plane_outer"),
    )


def get_skateboard_markers() -> SkateboardMarkers:
    return SkateboardMarkers(
        front_center_board=Marker("front_center_board"),
        front_left_board=Marker("front_left_board"),
        rear_left_board=Marker("rear_left_board"),
        rear_right_board=Marker("rear_right_board"),
        front_left_pole_left=Marker("front_left_pole_left"),
        front_left_pole_right=Marker("front_left_pole_right"),
        front_right_pole_left=Marker("front_right_pole_left"),
        front_right_pole_right=Marker("front_right_pole_right"),
        rear_left_pole_left=Marker("rear_left_pole_left"),
        rear_left_pole_right=Marker("rear_left_pole_right"),
        rear_right_pole_left=Marker("rear_right_pole_left"),
        rear_right_pole_right=Marker("rear_right_pole_right"),
        front_left_wheel=Marker("front_left_wheel"),
        front_right_wheel=Marker("front_right_wheel"),
        rear_left_wheel=Marker("rear_left_wheel"),
        rear_right_wheel=Marker("rear_right_wheel"),
    )


# The sole plane is fit from the three jig "plane" markers (the shoe is stationed on a
# flat block during calibration), but their centroid is meaningless as the patch origin
# and size. So the origin + extent auto-fit the actual foot markers, projected onto that
# plane: the rectangle bounds the footprint, centered on it.
SOLE_FOOTPRINT_MARKERS = (
    "heel",
    "toe",
    "sole_inner",
    "sole_outer",
    "heel_inner_1",
    "heel_inner_2",
    "heel_outer_1",
    "heel_outer_2",
    "toe_inner",
    "toe_outer",
)


def get_left_shoe_patches() -> LeftShoePatches:
    return LeftShoePatches(
        sole=Patch.rectangle(
            label="sole",
            markers=("plane_rear", "plane_inner", "plane_outer"),  # fit the plane only
            extent=bounding_box(*SOLE_FOOTPRINT_MARKERS, padding=0.005),  # origin + size
            outward_axis=SemanticAxis.UP,
            forward_axis=SemanticAxis.FORWARD,
            normal_offset=SOLE_PLANE_NORMAL_OFFSET,
            frame="sole_frame",
        )
    )


# The skateboard surface is a plane fit through all eight pole markers, raised 35mm along
# the fitted upward normal to the standing surface. The rectangle auto-fits the poles'
# footprint (the deck's standing area) instead of hand-guessed width/height.
def get_skateboard_patches() -> SkateboardPatches:
    return SkateboardPatches(
        surface=Patch.rectangle(
            label="surface",
            markers=SKATEBOARD_POLE_MARKERS,
            extent=bounding_box(*SKATEBOARD_POLE_MARKERS, padding=0.02),
            outward_axis=SemanticAxis.UP,
            forward_axis=SemanticAxis.FORWARD,
            normal_offset=SKATEBOARD_SURFACE_NORMAL_OFFSET,
            frame="surface_frame",
        )
    )


def get_left_foot_segments() -> LeftFootSegments:
    return LeftFootSegments(
        shoe=Segment(
            markers=get_left_shoe_markers(),
            patches=get_left_shoe_patches(),
            mocap_name="Left_Shoe_Improved",
        )
    )


def get_skateboard_segments() -> SkateboardSegments:
    return SkateboardSegments(
        deck=Segment(
            markers=get_skateboard_markers(),
            patches=get_skateboard_patches(),
            mocap_name="Skateboard",
        )
    )


def get_subjects() -> ExampleSubjects:
    left_foot = Subject(
        segments=get_left_foot_segments(),
        body_model=LEFT_SHOE_BODY_MODEL,
        mocap_name="Left_Shoe_Improved",
    )
    skateboard = Subject(
        segments=get_skateboard_segments(),
        body_model=SKATEBOARD_BODY_MODEL,
        mocap_name="Skateboard",
    )
    return ExampleSubjects(
        left_foot=left_foot,
        skateboard=skateboard,
    )


def get_tracks(unbagged_dir: Path) -> ExampleTracks:
    subjects = get_subjects()
    mocap_track = MocapTrack.from_unbagged(unbagged_dir, subjects, rebase_time=True)
    return ExampleTracks(mocap=mocap_track)


def get_demo(unbagged_dir: Path = DEFAULT_UNBAGGED_DIR) -> Demonstration[ExampleTracks]:
    tracks = get_tracks(unbagged_dir)
    return Demonstration(tracks)
