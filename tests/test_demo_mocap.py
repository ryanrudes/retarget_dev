from __future__ import annotations

import numpy as np
import pytest

from conftest import SEGMENT, SUBJECT, make_demo_patch_target, make_demo_subjects
from retarget.core import (
    RigidTransform,
    SceneState,
    SegmentKey,
    SegmentPoseTrajectory,
)
from retarget.demo.contact import ContactTrack
from retarget.demo.mocap import MocapTrack
from retarget.demo.resampling import ResampleMethod
from retarget.io import ViconMarkersFrame


def _segment_key() -> SegmentKey:
    return SegmentKey(SUBJECT, SEGMENT)


def _rotation_z_90() -> np.ndarray:
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_z_180() -> np.ndarray:
    return np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _mocap_track(*, with_contacts: bool = False) -> tuple[MocapTrack, object]:
    rotations = (np.eye(3, dtype=np.float64), _rotation_z_90(), _rotation_z_180())
    translations = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
    )
    state = SceneState(
        segment_poses={
            _segment_key(): SegmentPoseTrajectory(
                tuple(
                    RigidTransform.from_rotation_translation(rotation=r, translation=t)
                    for r, t in zip(rotations, translations, strict=True)
                )
            )
        }
    )
    timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    target = make_demo_patch_target("sole")
    contacts = None
    if with_contacts:
        contacts = ContactTrack(
            timestamps=timestamps,
            contacts={target: np.array([False, True, True], dtype=np.bool_)},
            confidences={target: np.array([0.1, 0.2, 0.3], dtype=np.float64)},
        )
    track = MocapTrack(
        subjects=make_demo_subjects(),
        state=state,
        timestamps=timestamps,
        marker_frames=(
            ViconMarkersFrame(stamp_seconds=0.0, markers=()),
            ViconMarkersFrame(stamp_seconds=1.0, markers=()),
            ViconMarkersFrame(stamp_seconds=2.0, markers=()),
        ),
        contacts=contacts,
    )
    return track, target


def test_resample_interpolates_translations() -> None:
    track, _ = _mocap_track()
    resampled = track.resample_to(np.array([0.5, 1.5], dtype=np.float64))
    np.testing.assert_allclose(
        resampled.segment_translations(_segment_key()),
        np.array([[0.5, 1.0, 1.5], [1.5, 3.0, 4.5]]),
    )


def test_resample_samples_rotations_discretely() -> None:
    track, _ = _mocap_track()
    resampled = track.resample_to(
        np.array([0.6, 1.6], dtype=np.float64),
        method=ResampleMethod.PREVIOUS,
    )
    rotations = resampled.segment_rotations(_segment_key())
    np.testing.assert_allclose(rotations[0], np.eye(3))
    np.testing.assert_allclose(rotations[1], _rotation_z_90())


def test_resample_relabels_output_timestamps() -> None:
    track, _ = _mocap_track()
    resampled = track.resample_to(
        np.array([0.5, 1.5], dtype=np.float64),
        output_timestamps=np.array([10.0, 20.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(resampled.timestamps, np.array([10.0, 20.0]))
    np.testing.assert_allclose(
        resampled.segment_translations(_segment_key())[:, 0], np.array([0.5, 1.5])
    )


def test_resample_delegates_attached_contacts() -> None:
    track, target = _mocap_track(with_contacts=True)
    resampled = track.resample_to(
        np.array([0.1, 1.6], dtype=np.float64),
        output_timestamps=np.array([10.0, 20.0], dtype=np.float64),
    )
    assert resampled.contacts is not None
    np.testing.assert_array_equal(resampled.contacts.timestamps, np.array([10.0, 20.0]))
    np.testing.assert_array_equal(resampled.contacts.state(target), np.array([False, True]))
    np.testing.assert_allclose(resampled.contacts.confidence(target), np.array([0.1, 0.3]))


def test_resample_method_applies_to_rotations_and_contacts() -> None:
    # A single `method` drives every discrete quantity: rotations and contacts.
    track, target = _mocap_track(with_contacts=True)
    sample = np.array([0.6, 1.6], dtype=np.float64)

    nearest = track.resample_to(sample)
    previous = track.resample_to(sample, method=ResampleMethod.PREVIOUS)

    assert nearest.contacts is not None
    assert previous.contacts is not None
    np.testing.assert_array_equal(
        nearest.contacts.state(target), np.array([True, True], dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        previous.contacts.state(target), np.array([False, True], dtype=np.bool_)
    )
    np.testing.assert_allclose(
        nearest.segment_rotations(_segment_key())[0], _rotation_z_90()
    )
    np.testing.assert_allclose(
        previous.segment_rotations(_segment_key())[0], np.eye(3)
    )


def test_resample_drops_raw_marker_frames() -> None:
    track, _ = _mocap_track()
    resampled = track.resample_to(np.array([0.5, 1.5], dtype=np.float64))
    assert track.marker_frames is not None
    assert resampled.marker_frames is None


def test_resample_rejects_empty_timestamps() -> None:
    track, _ = _mocap_track()
    with pytest.raises(ValueError, match="empty timestamps"):
        track.resample_to(np.array([], dtype=np.float64))


def test_resample_rejects_output_timestamp_length_mismatch() -> None:
    track, _ = _mocap_track()
    with pytest.raises(ValueError, match="same length"):
        track.resample_to(
            np.array([0.5, 1.5], dtype=np.float64),
            output_timestamps=np.array([10.0], dtype=np.float64),
        )
