"""Generic demonstration loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retarget.core.specs import SceneSpec
from retarget.demo.mocap import MocapTrack
from retarget.io import UnbaggedDirectory, iter_vicon_marker_frames, load_scene_state


def load_mocap_track(
    root: Path | UnbaggedDirectory,
    scene: SceneSpec,
    *,
    tf_prefix: str = "vicon",
) -> MocapTrack:
    """Load a mocap track from an unbagged export directory and scene spec."""
    export = root if isinstance(root, UnbaggedDirectory) else UnbaggedDirectory(root)
    state = load_scene_state(export, scene, tf_prefix=tf_prefix)
    marker_frames = tuple(iter_vicon_marker_frames(export))
    timestamps = np.asarray(
        [frame.stamp_seconds for frame in marker_frames],
        dtype=np.float64,
    )
    if state.num_timesteps != len(marker_frames):
        raise ValueError(
            "SceneState timestep count does not match marker frame count: "
            f"{state.num_timesteps} != {len(marker_frames)}"
        )
    return MocapTrack(
        scene_spec=scene,
        state=state,
        timestamps=timestamps,
        marker_frames=marker_frames,
    )
