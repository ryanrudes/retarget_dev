from __future__ import annotations

from pathlib import Path

import pytest

from backend_specs.ground_estimation_loader import load_ground_estimation_demo
from backend_specs.vicon_scene import VICON_SUBJECTS
from retarget.demo import Demonstration
from retarget.demo.mocap import MocapTrack

REPO_ROOT = Path(__file__).resolve().parents[1]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"

pytestmark = pytest.mark.skipif(
    not UNBAGGED_DIR.is_dir(), reason="ground-estimation bag not present"
)


def test_loader_returns_typed_demonstration() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    assert isinstance(demo, Demonstration)
    assert isinstance(demo.tracks["mocap"], MocapTrack)


def test_loader_normalizes_timestamps_to_start_at_zero() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    timestamps = demo.tracks["mocap"].timestamps
    assert timestamps[0] == 0.0
    assert timestamps[-1] > 0.0


def test_loader_slice_first_second_is_non_empty() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    clip = demo.tracks["mocap"].slice_time(0.0, 1.0)
    assert len(clip.timestamps) > 0


def test_from_unbagged_uses_authored_subjects() -> None:
    mocap = MocapTrack.from_unbagged(UNBAGGED_DIR, VICON_SUBJECTS)
    assert isinstance(mocap, MocapTrack)
    assert len(mocap.timestamps) == mocap.state.num_timesteps
    assert mocap.timestamps[0] != 0.0 or len(mocap.timestamps) == 1


def test_loader_typed_chain_queries() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    shoe = demo.tracks["mocap"].subjects["left_shoe"].segments["shoe"]
    heel = shoe.markers["heel"].positions()
    assert heel.ndim == 2
    assert heel.shape[1] == 3
    assert shoe.patches["sole"].points().shape[1] == 3
