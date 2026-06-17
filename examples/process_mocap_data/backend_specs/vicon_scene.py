"""Backend/manual typed scene for the real Vicon left-shoe data.

This is a backend/manual loader-support module: it authors the same public typed
``Subjects`` schema as ordinary examples, but derives marker geometry and the
sole patch calibration from a real VSK skeleton file rather than hand-tuned
constants. Ordinary typed authoring would instead pass geometry inline.
"""

from __future__ import annotations

from pathlib import Path

from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    Segment,
    Segments,
    SemanticAxis,
    Subject,
    Subjects,
    calibrate_patch_transform,
)
from retarget.demo import MocapTrack, Tracks
from retarget.io import read_marker_positions_from_vsk

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VSK_PATH = _REPO_ROOT / "models" / "Left_Shoe_Improved.vsk"

# VSK-derived segment-frame marker positions keyed by Vicon marker name.
_MARKER_POSITIONS = read_marker_positions_from_vsk(_VSK_PATH)

# Offset from the fitted calibration-marker plane to the physical sole contact
# plane, applied along the fitted patch normal after plane fitting.
SOLE_PLANE_NORMAL_OFFSET = -0.010


class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker
    plane_rear: Marker
    plane_inner: Marker
    plane_outer: Marker


class ShoePatches(Patches):
    sole: Patch


class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


class GroundEstimationSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


class GroundEstimationTracks(Tracks):
    mocap: MocapTrack[GroundEstimationSubjects]


def _marker(mocap_name: str) -> Marker:
    return Marker(mocap_name=mocap_name, position_segment=_MARKER_POSITIONS[mocap_name])


# Sole patch frame fit from the three shoe-sole plane calibration markers.
_SOLE_TRANSFORM = calibrate_patch_transform(
    marker_positions_segment=_MARKER_POSITIONS,
    markers=("plane_rear", "plane_inner", "plane_outer"),
    normal_offset=SOLE_PLANE_NORMAL_OFFSET,
    outward_axis=SemanticAxis.UP,
    x_axis=SemanticAxis.FORWARD,
)


VICON_SUBJECTS = GroundEstimationSubjects(
    left_shoe=Subject(
        mocap_name="Left_Shoe_Improved",
        segments=ShoeSegments(
            shoe=Segment(
                mocap_name="Left_Shoe_Improved",
                markers=ShoeMarkers(
                    heel=_marker("heel"),
                    toe=_marker("toe"),
                    plane_rear=_marker("plane_rear"),
                    plane_inner=_marker("plane_inner"),
                    plane_outer=_marker("plane_outer"),
                ),
                patches=ShoePatches(
                    sole=Patch.rectangular(
                        label="sole",
                        transform_segment_patch=_SOLE_TRANSFORM,
                        width=0.10,
                        height=0.25,
                        frame="sole_frame",
                    ),
                ),
            )
        ),
    ),
)
