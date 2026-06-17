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
)

from schema import (
    ExampleTracks,
    ExampleSubjects,
    LeftFootSegments,
    LeftShoePatches,
    LeftShoeMarkers,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VSK_PATH = REPO_ROOT / "models" / "Left_Shoe_Improved.vsk"
DEFAULT_UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"

# VSK-derived segment-frame marker positions keyed by Vicon marker name.
BODY_MODEL = read_marker_positions_from_vsk(VSK_PATH)

# Offset from the fitted calibration-marker plane to the physical sole contact
# plane, applied along the fitted patch normal after plane fitting.
SOLE_PLANE_NORMAL_OFFSET = -0.010


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


# Patch definitions. The sole frame is fit at bind time from the three plane
# calibration markers using their body_model segment-frame positions.
def get_left_shoe_patches() -> LeftShoePatches:
    return LeftShoePatches(
        sole=Patch.rectangle(
            label="sole",
            markers=(
                "plane_rear",
                "plane_inner",
                "plane_outer",
            ),
            width=0.10,
            height=0.25,
            outward_axis=SemanticAxis.UP,
            forward_axis=SemanticAxis.FORWARD,
            normal_offset=SOLE_PLANE_NORMAL_OFFSET,
            frame="sole_frame",
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


def get_subjects() -> ExampleSubjects:
    left_foot = Subject(
        segments=get_left_foot_segments(),
        body_model=BODY_MODEL,
        mocap_name="Left_Shoe_Improved",
    )
    return ExampleSubjects(left_foot=left_foot)


def get_tracks(unbagged_dir: Path) -> ExampleTracks:
    subjects = get_subjects()
    mocap_track = MocapTrack.from_unbagged(unbagged_dir, subjects, rebase_time=True)
    return ExampleTracks(mocap=mocap_track)


def get_demo(unbagged_dir: Path = DEFAULT_UNBAGGED_DIR) -> Demonstration[ExampleTracks]:
    tracks = get_tracks(unbagged_dir)
    return Demonstration(tracks)