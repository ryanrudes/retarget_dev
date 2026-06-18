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
    RightFootSegments,
    RightShoePatches,
    RightShoeMarkers,
    BalanceBoardSegments,
    BalanceBoardPatches,
    BalanceBoardMarkers,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNBAGGED_DIR = REPO_ROOT / "bags" / "stepping_on_board" / "unbagged"

LEFT_SHOE_VSK_PATH = REPO_ROOT / "models" / "Left_Shoe_Improved.vsk"
RIGHT_SHOE_VSK_PATH = REPO_ROOT / "models" / "Right_Shoe_Improved.vsk"
BALANCE_BOARD_VSK_PATH = REPO_ROOT / "models" / "Balance_Board.vsk"

# VSK-derived segment-frame marker positions keyed by Vicon marker name.
LEFT_SHOE_BODY_MODEL = read_marker_positions_from_vsk(LEFT_SHOE_VSK_PATH)
RIGHT_SHOE_BODY_MODEL = read_marker_positions_from_vsk(RIGHT_SHOE_VSK_PATH)
BALANCE_BOARD_BODY_MODEL = read_marker_positions_from_vsk(BALANCE_BOARD_VSK_PATH)

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


def get_right_shoe_markers() -> RightShoeMarkers:
    return RightShoeMarkers(
        marker1=Marker("Unlabeled18579"),
        marker2=Marker("Unlabeled20165"),
        marker3=Marker("Unlabeled20991"),
        marker4=Marker("Unlabeled21322"),
        marker5=Marker("Unlabeled21323"),
        marker6=Marker("Unlabeled21324"),
    )


def get_balance_board_markers() -> BalanceBoardMarkers:
    return BalanceBoardMarkers(
        surface1=Marker("Surface1"),
        surface2=Marker("Surface2"),
        surface3=Marker("Surface3"),
        surface4=Marker("Surface4"),
        edge1=Marker("Edge1"),
        edge2=Marker("Edge2"),
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


def get_right_shoe_patches() -> RightShoePatches:
    return RightShoePatches()


def get_balance_board_patches() -> BalanceBoardPatches:
    return BalanceBoardPatches(
        surface=Patch.rectangle(
            label="surface",
            markers=("surface1", "surface2", "surface3", "surface4"),
            width=0.6,
            height=0.3,
            outward_axis=SemanticAxis.UP,
            forward_axis=SemanticAxis.FORWARD,
            normal_offset=0.0,
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


def get_right_foot_segments() -> RightFootSegments:
    return RightFootSegments(
        shoe=Segment(
            markers=get_right_shoe_markers(),
            patches=get_right_shoe_patches(),
            mocap_name="Right_Shoe_Improved",
        )
    )


def get_balance_board_segments() -> BalanceBoardSegments:
    return BalanceBoardSegments(
        board=Segment(
            markers=get_balance_board_markers(),
            patches=get_balance_board_patches(),
            mocap_name="Balance_Board",
        )
    )


def get_subjects() -> ExampleSubjects:
    left_foot = Subject(
        segments=get_left_foot_segments(),
        body_model=LEFT_SHOE_BODY_MODEL,
        mocap_name="Left_Shoe_Improved",
    )
    right_foot = Subject(
        segments=get_right_foot_segments(),
        body_model=RIGHT_SHOE_BODY_MODEL,
        mocap_name="Right_Shoe_Improved",
    )
    balance_board = Subject(
        segments=get_balance_board_segments(),
        body_model=BALANCE_BOARD_BODY_MODEL,
        mocap_name="Balance_Board",
    )
    return ExampleSubjects(
        left_foot=left_foot,
        right_foot=right_foot,
        balance_board=balance_board,
    )


def get_tracks(unbagged_dir: Path) -> ExampleTracks:
    subjects = get_subjects()
    mocap_track = MocapTrack.from_unbagged(unbagged_dir, subjects, rebase_time=True)
    return ExampleTracks(mocap=mocap_track)


def get_demo(unbagged_dir: Path = DEFAULT_UNBAGGED_DIR) -> Demonstration[ExampleTracks]:
    tracks = get_tracks(unbagged_dir)
    return Demonstration(tracks)