"""Shared scene definition for the ``left_shoe_*`` roller-board examples.

Every ``left_shoe_*`` example authors the *same* typed scene (a left shoe on a
balance board) and only differs in which captured bag it loads. The scene is
defined once here; each example's ``scene.py`` is a thin shim that calls
:func:`get_demo` with its own bag directory (derived from the example folder
name).
"""

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
    min_area_rectangle,
    plane_from,
    side,
)

from .schema import (
    ExampleTracks,
    ExampleSubjects,
    LeftFootSegments,
    LeftShoePatches,
    LeftShoeMarkers,
    BalanceBoardSegments,
    BalanceBoardPatches,
    BalanceBoardMarkers,
)

# Foot markers whose projected footprint sizes + centers the sole patch (the three
# "plane" markers are a calibration jig and only fit the plane).
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

BALANCE_BOARD_FOOTPRINT_MARKERS = (
    "surface1",
    "surface2",
    "surface3",
    "surface4",
    "edge1",
    "edge2",
)


REPO_ROOT = Path(__file__).resolve().parents[2]

LEFT_SHOE_VSK_PATH = REPO_ROOT / "models" / "Left_Shoe_Improved.vsk"
BALANCE_BOARD_VSK_PATH = REPO_ROOT / "models" / "Balance_Board.vsk"

# VSK-derived segment-frame marker positions keyed by Vicon marker name.
LEFT_SHOE_BODY_MODEL = read_marker_positions_from_vsk(LEFT_SHOE_VSK_PATH)
BALANCE_BOARD_BODY_MODEL = read_marker_positions_from_vsk(BALANCE_BOARD_VSK_PATH)

# Offset from the fitted calibration-marker plane to the physical sole contact
# plane, applied along the patch normal. A POSITIVE offset nudges the contact plane
# outward, whereas a NEGATIVE offset nudges it inward.
SOLE_PLANE_NORMAL_OFFSET = 0.00395  # 3.95 mm outward

# Offset from the fitted calibration-marker plane to the physical balance board surface
# plane, applied along the patch normal. A POSITIVE offset nudges the contact plane
# outward, whereas a NEGATIVE offset nudges it inward.
BALANCE_BOARD_PLANE_NORMAL_OFFSET = -0.0046  # 4.6 mm inward


def unbagged_dir_for(example_name: str) -> Path:
    """The ``bags/<example_name>/unbagged`` directory for a given example folder."""
    return REPO_ROOT / "bags" / example_name / "unbagged"


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
#
# Instead of guessing which way the fitted normal points (and therefore the sign
# of normal_offset), we anchor it to a marker we KNOW the side of: the shoe body
# (toe_grid_1) is on the INWARD side, so the outward normal points the other way --
# down, toward the board the sole contacts. With "outward = toward the contact",
# +normal_offset always nudges the plane toward the board.
def get_left_shoe_patches() -> LeftShoePatches:
    return LeftShoePatches(
        sole=Patch.planar(
            label="sole",
            plane=plane_from("plane_rear", "plane_inner", "plane_outer"),  # fit the plane only
            extent=bounding_box(*SOLE_FOOTPRINT_MARKERS, padding=0.005),  # origin + size
            # shoe body (toe_grid_1) is inward -> outward normal faces the board;
            # +offset then nudges the contact plane toward the board.
            normal=side("toe_grid_1", defines=-SemanticAxis.UP, offset=SOLE_PLANE_NORMAL_OFFSET),
            frame="sole_frame",
        )
    )


# The deck edge sits below the standing surface, so it is on the INWARD side: the
# outward normal points up, toward the shoe. The surface markers already lie on the
# standing surface, so no offset is needed.
#
# The board is not axis-aligned to the capture's FORWARD, so orient the in-plane
# axes to the footprint's own minimum-area rectangle; bounding_box over the same
# markers then yields that tight, naturally-aligned rectangle (not a rotated,
# oversized axis-aligned box).
def get_balance_board_patches() -> BalanceBoardPatches:
    return BalanceBoardPatches(
        surface=Patch.planar(
            label="surface",
            plane=plane_from("surface1", "surface2", "surface3", "surface4"),
            tangential=min_area_rectangle(*BALANCE_BOARD_FOOTPRINT_MARKERS),
            extent=bounding_box(*BALANCE_BOARD_FOOTPRINT_MARKERS, padding=0.005),  # origin + size
            # deck edge sits below the surface -> outward normal faces the shoe.
            normal=side("edge1", defines=-SemanticAxis.UP, offset=BALANCE_BOARD_PLANE_NORMAL_OFFSET),
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
    balance_board = Subject(
        segments=get_balance_board_segments(),
        body_model=BALANCE_BOARD_BODY_MODEL,
        mocap_name="Balance_Board",
    )
    return ExampleSubjects(
        left_foot=left_foot,
        balance_board=balance_board,
    )


def get_tracks(unbagged_dir: Path) -> ExampleTracks:
    subjects = get_subjects()
    mocap_track = MocapTrack.from_unbagged(unbagged_dir, subjects, rebase_time=True)
    return ExampleTracks(mocap=mocap_track)


def get_demo(unbagged_dir: Path) -> Demonstration[ExampleTracks]:
    tracks = get_tracks(unbagged_dir)
    return Demonstration(tracks)
