"""Shared scene definition for the ``left_shoe_*`` roller-board examples.

Every ``left_shoe_*`` example authors the *same* typed scene (a left shoe on a
balance board) and only differs in which captured bag it loads. The scene is
defined once here; each example's ``scene.py`` is a thin shim that calls
:func:`get_demo` with its own bag directory (derived from the example folder
name).
"""

from __future__ import annotations

from pathlib import Path

from fungeom import Face, Point3Bundle, Region2

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


def get_left_foot_segments() -> LeftFootSegments:
    # Author the shoe markers once as typed symbols, then express the sole patch as plain fungeom
    # DATA over those symbols (``m.rest`` -- the marker's segment-frame rest position as a free
    # variable). No string keys, no callable; a misspelled marker is a NameError, not a silent key.
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

    # Fit the contact plane through the three calibration markers. The shoe body (toe_grid_1) is on
    # the INWARD side, so ``facing(toe_grid_1.rest).flipped()`` points the outward normal the other
    # way -- toward the board the sole contacts -- then +offset nudges the plane to the physical
    # contact. The footprint is the convex hull of the foot markers flattened into that plane, +5 mm.
    plane = (
        Point3Bundle.of([m.rest for m in (plane_rear, plane_inner, plane_outer)])
        .fit_plane()
        .facing(toe_grid_1.rest)
        .flipped()
        .offset(SOLE_PLANE_NORMAL_OFFSET)
    )
    footprint_markers = (
        heel, toe, sole_inner, sole_outer,
        heel_inner_1, heel_inner_2, heel_outer_1, heel_outer_2, toe_inner, toe_outer,
    )
    footprint = Region2.hull(
        Point3Bundle.of([m.rest for m in footprint_markers]).in_frame(plane)
    ).offset(0.005)

    return LeftFootSegments(
        shoe=Segment(
            markers=LeftShoeMarkers(
                heel=heel, toe=toe,
                heel_inner_1=heel_inner_1, heel_inner_2=heel_inner_2,
                heel_outer_1=heel_outer_1, heel_outer_2=heel_outer_2,
                toe_inner=toe_inner, toe_outer=toe_outer,
                toe_grid_1=toe_grid_1, toe_grid_2=toe_grid_2, toe_grid_3=toe_grid_3, toe_grid_4=toe_grid_4,
                sole_inner=sole_inner, sole_outer=sole_outer,
                plane_rear=plane_rear, plane_inner=plane_inner, plane_outer=plane_outer,
            ),
            patches=LeftShoePatches(sole=Patch(label="sole", geometry=Face.on(plane, footprint), frame="sole_frame")),
            mocap_name="Left_Shoe_Improved",
        )
    )


def get_balance_board_segments() -> BalanceBoardSegments:
    surface1 = Marker("Surface1")
    surface2 = Marker("Surface2")
    surface3 = Marker("Surface3")
    surface4 = Marker("Surface4")
    edge1 = Marker("Edge1")
    edge2 = Marker("Edge2")

    # Fit the deck plane through the four surface markers. The deck edge (edge1) sits below the
    # surface (INWARD), so ``facing(edge1.rest).flipped()`` orients the outward normal up toward the
    # shoe (offset is inward, negative). The hull of the footprint markers in-plane is the footprint.
    plane = (
        Point3Bundle.of([m.rest for m in (surface1, surface2, surface3, surface4)])
        .fit_plane()
        .facing(edge1.rest)
        .flipped()
        .offset(BALANCE_BOARD_PLANE_NORMAL_OFFSET)
    )
    footprint = Region2.hull(
        Point3Bundle.of([m.rest for m in (surface1, surface2, surface3, surface4, edge1, edge2)]).in_frame(plane)
    ).offset(0.005)

    return BalanceBoardSegments(
        board=Segment(
            markers=BalanceBoardMarkers(
                surface1=surface1, surface2=surface2, surface3=surface3, surface4=surface4, edge1=edge1, edge2=edge2
            ),
            patches=BalanceBoardPatches(surface=Patch(label="surface", geometry=Face.on(plane, footprint), frame="surface_frame")),
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
