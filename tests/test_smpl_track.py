"""A SMPL joint track + its energy extractor, and end-to-end temporal sync against a reference."""

from __future__ import annotations

import numpy as np
import pytest

from retarget.demo import (
    Demonstration,
    EnergySignal,
    SmplTrack,
    SyncEdge,
    SyncPlan,
    Tracks,
    estimate_sync,
    smpl_joint_energy,
)


def _ankle_track(shift: float = 0.0) -> SmplTrack:
    # A distinctive vertical "footstep" signature on the left ankle -- three bumps of different
    # heights at 0.5/1.0/1.4 s (unevenly spaced, so the cross-correlation peak is unambiguous);
    # the root is static. 100 Hz over 2 s.
    t = np.linspace(0.0, 2.0, 201)
    z = sum(h * np.exp(-((t - (c + shift)) ** 2) / 0.002) for c, h in ((0.5, 1.0), (1.0, 0.6), (1.4, 0.9)))
    joints = np.zeros((t.shape[0], 2, 3))
    joints[:, 0, 2] = z
    return SmplTrack(joints=joints, joint_names=("left_ankle", "root"), timestamps=t)


def test_construction_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        SmplTrack(joints=np.zeros((3, 2)), joint_names=("a", "b"), timestamps=np.arange(3.0))
    with pytest.raises(ValueError, match="joint_names"):
        SmplTrack(joints=np.zeros((3, 2, 3)), joint_names=("a",), timestamps=np.arange(3.0))
    with pytest.raises(ValueError, match="unique"):
        SmplTrack(joints=np.zeros((3, 2, 3)), joint_names=("a", "a"), timestamps=np.arange(3.0))
    with pytest.raises(ValueError, match="frames"):
        SmplTrack(joints=np.zeros((3, 1, 3)), joint_names=("a",), timestamps=np.arange(4.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        SmplTrack(joints=np.zeros((3, 1, 3)), joint_names=("a",), timestamps=np.array([0.0, 0.0, 1.0]))


def test_joint_queries() -> None:
    track = _ankle_track()
    assert track.joint_index("root") == 1
    assert track.joint_positions("left_ankle").shape == (201, 3)
    assert track.joint_velocities("left_ankle").shape == (201, 3)
    np.testing.assert_array_equal(track.joint_positions("root"), np.zeros((201, 3)))
    with pytest.raises(KeyError, match="has no joint 'missing'"):
        track.joint_index("missing")


def test_slice_time_returns_smpl_track() -> None:
    clip = _ankle_track().slice_time(0.4, 0.6)
    assert isinstance(clip, SmplTrack)
    assert clip.joints.shape[1:] == (2, 3)
    assert clip.joint_names == ("left_ankle", "root")
    assert np.all((clip.timestamps >= 0.4) & (clip.timestamps < 0.6))


def test_energy_extractor_returns_scalar_signal() -> None:
    energy = smpl_joint_energy("left_ankle", axis=2)(_ankle_track())
    assert isinstance(energy, EnergySignal)
    assert energy.values.ndim == 1 and energy.values.shape == (201,)
    assert energy.values.max() > 0.0  # the footstep moves vertically
    # 3-D speed (axis=None) agrees here since only z moves
    speed = smpl_joint_energy("left_ankle")(_ankle_track())
    np.testing.assert_allclose(speed.values, energy.values, atol=1e-9)


def test_energy_extractor_rejects_foreign_track() -> None:
    class _NotSmpl:
        pass

    with pytest.raises(TypeError, match="expects a SmplTrack"):
        smpl_joint_energy("left_ankle")(_NotSmpl())  # type: ignore[arg-type]


def test_smpl_track_syncs_to_reference_recovering_lag() -> None:
    # The video (source) ankle motion happens 0.07 s LATER than the reference; sync must recover it
    # through the existing energy/timeline machinery, with no sync-side changes for the new track.
    reference = _ankle_track(shift=0.0)
    video = _ankle_track(shift=0.07)

    class _Tracks(Tracks):
        reference: SmplTrack
        video: SmplTrack

    demo = Demonstration(_Tracks(reference=reference, video=video))
    energy = smpl_joint_energy("left_ankle", axis=2)
    plan = SyncPlan(
        reference="reference",
        edges=(
            SyncEdge(
                source="video",
                reference="reference",
                source_signal=energy,
                reference_signal=energy,
                max_lag_seconds=0.25,
            ),
        ),
    )
    (alignment,) = estimate_sync(demo, plan)
    # t_reference = t_source + offset; the source event is 0.07 s late, so offset ~ -0.07.
    assert alignment.transform.offset == pytest.approx(-0.07, abs=0.02)
