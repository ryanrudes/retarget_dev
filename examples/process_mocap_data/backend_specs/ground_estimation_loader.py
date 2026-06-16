"""Backend/manual ground-estimation loader for real VSK-derived bag data."""

from __future__ import annotations

from pathlib import Path

from demo_vocab import GroundEstimationTrackId

from .vicon_scene import VICON_SCENE
from retarget.demo.demo import Demonstration
from retarget.demo.loaders import load_mocap_track
from retarget.io import UnbaggedDirectory


def load_ground_estimation_demo(
    root: Path | UnbaggedDirectory,
) -> Demonstration[GroundEstimationTrackId]:
    """Load a ground-estimation demonstration from an unbagged export directory."""
    # Demonstration time is relative to the first frame so slice_time(0.0, 1.0)
    # means "the first second of the clip", not ROS epoch seconds.
    mocap = load_mocap_track(root, VICON_SCENE).with_rebased_time()
    return Demonstration(
        tracks={
            GroundEstimationTrackId.MOCAP: mocap,
        }
    )
