"""Backend/manual ground-estimation loader for real VSK-derived bag data."""

from __future__ import annotations

from pathlib import Path

from .vicon_scene import GroundEstimationTracks, VICON_SUBJECTS
from retarget.demo import Demonstration, build_demonstration, load_mocap_track
from retarget.io import UnbaggedDirectory


def load_ground_estimation_demo(
    root: Path | UnbaggedDirectory,
) -> Demonstration[GroundEstimationTracks]:
    """Load a typed ground-estimation demonstration from an unbagged export."""
    # Demonstration time is relative to the first frame so slice_time(0.0, 1.0)
    # means "the first second of the clip", not ROS epoch seconds.
    mocap = load_mocap_track(root, VICON_SUBJECTS).with_rebased_time()
    return build_demonstration(GroundEstimationTracks(mocap=mocap))
