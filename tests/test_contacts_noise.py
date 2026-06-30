"""Noise calibration + confidence-grounded robustness of contact detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from retarget.contacts import (
    ContactDetectionConfig,
    classify_contacts,
    estimate_noise,
    ground_plane,
)
from retarget.core import RigidTransform, SceneState, SegmentKey, SegmentPoseTrajectory

from conftest import (
    SEGMENT,
    SUBJECT,
    demo_segment,
    make_demo_patch_target,
    make_demo_subjects,
    make_descending_mocap_track,
)
from retarget.demo.mocap import MocapTrack


def _sole(track: Any) -> Any:
    return demo_segment(track).patches.sole


def _static_noisy_track(sigma_z: float, *, n: int = 300, seed: int = 0) -> MocapTrack:
    """A still segment whose pose jitters by ``sigma_z`` in z -- a noise calibration clip."""
    rng = np.random.default_rng(seed)
    timestamps = np.linspace(0.0, 3.0, n)
    z = rng.normal(0.0, sigma_z, n)
    poses = tuple(RigidTransform.from_translation(np.array([0.0, 0.0, float(z[i])])) for i in range(n))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(subjects=make_demo_subjects(), state=state, timestamps=timestamps, marker_frames=None)


def test_estimate_noise_recovers_injected_clearance_sigma() -> None:
    track = _static_noisy_track(sigma_z=0.002)  # 2 mm positional jitter
    calib = estimate_noise(track)
    sigma = calib.for_patch(make_demo_patch_target("sole")).clearance_sigma
    # Recovered within ~40% of the injected 2 mm (robust MAD estimate, σ-floored).
    assert 0.0012 < sigma < 0.0030


def test_large_gap_has_near_zero_confidence() -> None:
    # Foot rests at z=0; the floor is 4 cm below -> a 4 cm gap -> not in contact.
    track = make_descending_mocap_track()
    t = track.timestamps
    states = classify_contacts(_sole(track), against={"floor": ground_plane(-0.04)})
    target = make_demo_patch_target("sole")
    rest = (t >= 1.2) & (t <= 2.8)
    assert float(np.mean(states.support_scores(target)["floor"][rest])) < 1e-3
    assert (states.support_state(target, none="air")[rest] == "air").all()


def test_which_support_is_robust_across_confidence_sweep() -> None:
    # The shoe rests on the "board" (z=0); the "floor" is 4 cm below. The board should
    # win at every confidence level -- the choice is driven by the data, not a knob.
    track = make_descending_mocap_track()
    t = track.timestamps
    rest = (t >= 1.2) & (t <= 2.8)
    target = make_demo_patch_target("sole")
    against = {"floor": ground_plane(-0.04), "board": ground_plane(0.0)}  # floor declared first

    for confidence in (0.5, 0.8, 0.9, 0.95, 0.99):
        states = classify_contacts(
            _sole(track), against=against, config=ContactDetectionConfig(contact_confidence=confidence)
        )
        labels = states.support_state(target, none="air")
        assert (labels[rest] == "board").all(), f"flipped at confidence={confidence}"


def _moved_track(*, z_low: float = 0.0, z_high: float = 0.045, n: int = 300) -> MocapTrack:
    """A segment that rests at ``z_low`` for the first half, then at ``z_high``.

    Models a patch moved from one support to another (shoe: floor -> board)."""
    timestamps = np.linspace(0.0, 3.0, n)
    z = np.where(timestamps < 1.5, z_low, z_high)
    poses = tuple(RigidTransform.from_translation(np.array([0.0, 0.0, float(z[i])])) for i in range(n))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(subjects=make_demo_subjects(), state=state, timestamps=timestamps, marker_frames=None)


def test_resting_bias_is_per_support_and_not_pooled_across_the_clip() -> None:
    # Shoe rests on the floor (z=0), then on a board (z=4.5 cm). The board's resting bias
    # must come only from the on-board frames -- pooling the on-floor frames (4.5 cm below
    # the board) would shift the genuine on-board clearance into a false gap.
    track = _moved_track()
    t = track.timestamps
    target = make_demo_patch_target("sole")
    states = classify_contacts(
        _sole(track), against={"ground": ground_plane(0.0), "board": ground_plane(0.045)}
    )
    labels = states.support_state(target, none="air")
    early = (t >= 0.3) & (t <= 1.2)
    late = (t >= 1.8) & (t <= 2.8)
    assert (labels[early] == "ground").all()
    assert (labels[late] == "board").all()


def test_calibrated_noise_keeps_the_answer() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    rest = (t >= 1.2) & (t <= 2.8)
    target = make_demo_patch_target("sole")
    calib = estimate_noise(track)  # calibrate on the clip itself (floored σ)
    states = classify_contacts(
        _sole(track),
        against={"floor": ground_plane(-0.04), "board": ground_plane(0.0)},
        config=ContactDetectionConfig(noise=calib),
    )
    assert (states.support_state(target, none="air")[rest] == "board").all()
