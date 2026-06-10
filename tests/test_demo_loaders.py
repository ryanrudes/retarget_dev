from __future__ import annotations

from pathlib import Path

import pytest

from demo_specs import load_ground_estimation_demo
from retarget.demo import load_mocap_track
from mocap_specs import VICON_SCENE
from mocap_vocab import LeftShoeMarkerId, LeftShoeSegmentId, ViconSubjectId

REPO_ROOT = Path(__file__).resolve().parents[1]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"


@pytest.mark.skipif(not UNBAGGED_DIR.is_dir(), reason="ground-estimation bag not present")
def test_loader_normalizes_timestamps_to_start_at_zero() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    timestamps = demo.mocap.timestamps
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 0.0


@pytest.mark.skipif(not UNBAGGED_DIR.is_dir(), reason="ground-estimation bag not present")
def test_loader_slice_first_second_is_non_empty() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    clip = demo.slice_time(0.0, 1.0)
    assert len(clip.mocap.timestamps) > 0


@pytest.mark.skipif(not UNBAGGED_DIR.is_dir(), reason="ground-estimation bag not present")
def test_load_mocap_track_uses_example_scene_spec() -> None:
    mocap = load_mocap_track(UNBAGGED_DIR, VICON_SCENE)
    assert mocap.scene_spec is VICON_SCENE
    assert len(mocap.timestamps) == mocap.state.num_timesteps
    assert mocap.timestamps[0] != 0.0 or len(mocap.timestamps) == 1


@pytest.mark.skipif(not UNBAGGED_DIR.is_dir(), reason="ground-estimation bag not present")
def test_loader_accepts_example_vocab_marker_ids() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    left_shoe = demo.mocap.subject(ViconSubjectId.LEFT_SHOE).segment(
        LeftShoeSegmentId.LEFT_SHOE
    )
    heel = left_shoe.marker_positions(LeftShoeMarkerId.HEEL)
    assert heel.ndim == 2
    assert heel.shape[1] == 3
