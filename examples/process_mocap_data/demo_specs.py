"""Example-specific ground-estimation track IDs and loader."""

from __future__ import annotations

from pathlib import Path

from mocap_specs import VICON_SCENE
from retarget.core.enums import TrackId
from retarget.demo.demo import Demonstration
from retarget.demo.loaders import load_mocap_track
from retarget.demo.mocap import MocapTrack
from retarget.io import UnbaggedDirectory


class GroundEstimationTrackId(TrackId):
    """Track identifiers for the ground-estimation demonstration."""

    MOCAP = "mocap"
    VIDEO = "video"
    SMPL = "smpl"
    CONTACTS = "contacts"


def load_ground_estimation_demo(
    root: Path | UnbaggedDirectory,
) -> Demonstration[GroundEstimationTrackId]:
    """
    Load a ground-estimation demonstration from an unbagged export directory.

    The first pass is mocap-only. Video, SMPL, and contact tracks may be added later.
    """
    mocap = load_mocap_track(root, VICON_SCENE)
    # Demonstration time is relative to the first frame so slice_time(0.0, 1.0)
    # means "the first second of the clip", not ROS epoch seconds.
    if len(mocap.timestamps) > 0:
        timestamps = mocap.timestamps - mocap.timestamps[0]
        mocap = MocapTrack(
            scene_spec=mocap.scene_spec,
            state=mocap.state,
            timestamps=timestamps,
            marker_frames=mocap.marker_frames,
        )
    return Demonstration(
        tracks={
            GroundEstimationTrackId.MOCAP: mocap
        }
    )
