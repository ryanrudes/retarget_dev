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
    # The shoe markers are declared once on ``LeftShoeMarkers`` (as typed field defaults); construct
    # the schema and express the sole patch as plain fungeom DATA over those symbols (``m.<marker>.rest``
    # -- the marker's segment-frame rest position as a free variable). No string keys, no callable, no
    # second listing of the markers; a misspelled symbol is a mypy error, not a silent key.
    m = LeftShoeMarkers()

    # Fit the contact plane through the three calibration markers. The shoe body (toe_grid_1) is on
    # the INWARD side, so ``facing(toe_grid_1.rest).flipped()`` points the outward normal the other
    # way -- toward the board the sole contacts -- then +offset nudges the plane to the physical
    # contact. The footprint is the convex hull of the foot markers flattened into that plane, +5 mm.
    plane = (
        Point3Bundle.of([p.rest for p in (m.plane_rear, m.plane_inner, m.plane_outer)])
        .fit_plane()
        .facing(m.toe_grid_1.rest)
        .flipped()
        .offset(SOLE_PLANE_NORMAL_OFFSET)
    )
    footprint_markers = (
        m.heel, m.toe, m.sole_inner, m.sole_outer,
        m.heel_inner_1, m.heel_inner_2, m.heel_outer_1, m.heel_outer_2, m.toe_inner, m.toe_outer,
    )
    footprint = Region2.hull(
        Point3Bundle.of([p.rest for p in footprint_markers]).in_frame(plane)
    ).offset(0.005)

    return LeftFootSegments(
        shoe=Segment(
            markers=m,
            patches=LeftShoePatches(sole=Patch(label="sole", geometry=Face.on(plane, footprint), frame="sole_frame")),
            mocap_name="Left_Shoe_Improved",
        )
    )


def get_balance_board_segments() -> BalanceBoardSegments:
    m = BalanceBoardMarkers()

    # Fit the deck plane through the four surface markers. The deck edge (edge1) sits below the
    # surface (INWARD), so ``facing(edge1.rest).flipped()`` orients the outward normal up toward the
    # shoe (offset is inward, negative). The hull of the footprint markers in-plane is the footprint.
    plane = (
        Point3Bundle.of([p.rest for p in (m.surface1, m.surface2, m.surface3, m.surface4)])
        .fit_plane()
        .facing(m.edge1.rest)
        .flipped()
        .offset(BALANCE_BOARD_PLANE_NORMAL_OFFSET)
    )
    footprint = Region2.hull(
        Point3Bundle.of([p.rest for p in (m.surface1, m.surface2, m.surface3, m.surface4, m.edge1, m.edge2)]).in_frame(plane)
    ).offset(0.005)

    return BalanceBoardSegments(
        board=Segment(
            markers=m,
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
