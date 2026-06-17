"""Backend/manual ground-estimation loader for real VSK-derived bag data."""

from __future__ import annotations

from pathlib import Path

from .vicon_scene import GroundEstimationTracks, VICON_SUBJECTS
from retarget.demo import Demonstration, MocapTrack
from retarget.io import UnbaggedDirectory


def load_ground_estimation_demo(
    root: Path | UnbaggedDirectory,
) -> Demonstration[GroundEstimationTracks]:
    """Load a typed ground-estimation demonstration from an unbagged export."""
    # Demonstration time is relative to the first frame so slice_time(0.0, 1.0)
    # means "the first second of the clip", not ROS epoch seconds.
    mocap = MocapTrack.from_unbagged(root, VICON_SUBJECTS).with_rebased_time()
    return Demonstration(GroundEstimationTracks(mocap=mocap))
