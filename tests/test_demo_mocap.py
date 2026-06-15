

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    DEMO_SEGMENT_SPEC,
    DemoPatchId,
    DemoSceneSpec,
    DemoSegmentId,
    DemoSubjectId,
    DemoSubjectSpec,
)
from retarget.core import PatchHandle
from retarget.core.keys import SegmentKey
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.targets import PatchTarget
from retarget.core.transform import RigidTransform
from retarget.demo.contact import ContactTrack
from retarget.demo.mocap import MocapTrack
from retarget.demo.resampling import ResampleMethod
from retarget.io import ViconMarkersFrame


def _scene_spec() -> DemoSceneSpec:
    return DemoSceneSpec(
        subject_spec=DemoSubjectSpec(
            subject=DemoSubjectId.SUBJECT,
            segment_spec=DEMO_SEGMENT_SPEC,
        )
    )


def _segment_key() -> SegmentKey:
    return SegmentKey(DemoSubjectId.SUBJECT, DemoSegmentId.SEGMENT)


def _rotation_z_90() -> np.ndarray:
    return np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_z_180() -> np.ndarray:
    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _mocap_track(*, with_contacts: bool = False) -> tuple[MocapTrack, PatchTarget[PatchHandle]]:
    rotations = (
        np.eye(3, dtype=np.float64),
        _rotation_z_90(),
        _rotation_z_180(),
    )
    translations = (
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([2.0, 4.0, 6.0], dtype=np.float64),
    )
    state = SceneState(
        segment_poses={
            _segment_key(): SegmentPoseTrajectory(
                tuple(
                    RigidTransform.from_rotation_translation(
                        rotation=rotation,
                        translation=translation,
                    )
                    for rotation, translation in zip(rotations, translations, strict=True)
                )
            )
        }
    )
    timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(
            segment=DemoSegmentId.SEGMENT,
            patch=DemoPatchId.SOLE,
        ),
    )
    contacts = None
    if with_contacts:
        contacts = ContactTrack(
            timestamps=timestamps,
            contacts={target: np.array([False, True, True], dtype=np.bool_)},
            confidences={target: np.array([0.1, 0.2, 0.3], dtype=np.float64)},
        )
    track = MocapTrack(
        scene_spec=_scene_spec(),
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


def test_mocap_track_resample_to_interpolates_segment_translations() -> None:
    track, _ = _mocap_track()

    resampled = track.resample_to(np.array([0.5, 1.5], dtype=np.float64))

    translations = resampled.segment_translations(_segment_key())
    np.testing.assert_allclose(
        translations,
        np.array(
            [
                [0.5, 1.0, 1.5],
                [1.5, 3.0, 4.5],
            ],
            dtype=np.float64,
        ),
    )


def test_mocap_track_resample_to_samples_rotations_discretely() -> None:
    track, _ = _mocap_track()

    resampled = track.resample_to(
        np.array([0.6, 1.6], dtype=np.float64),
        rotation_method=ResampleMethod.PREVIOUS,
    )

    rotations = resampled.segment_rotations(_segment_key())
    np.testing.assert_allclose(rotations[0], np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(rotations[1], _rotation_z_90())


def test_mocap_track_resample_to_can_relabel_output_timestamps() -> None:
    track, _ = _mocap_track()

    resampled = track.resample_to(
        np.array([0.5, 1.5], dtype=np.float64),
        output_timestamps=np.array([10.0, 20.0], dtype=np.float64),
    )

    np.testing.assert_array_equal(
        resampled.timestamps,
        np.array([10.0, 20.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        resampled.segment_translations(_segment_key())[:, 0],
        np.array([0.5, 1.5], dtype=np.float64),
    )


def test_mocap_track_resample_to_delegates_attached_contacts() -> None:
    track, target = _mocap_track(with_contacts=True)

    resampled = track.resample_to(
        np.array([0.1, 1.6], dtype=np.float64),
        output_timestamps=np.array([10.0, 20.0], dtype=np.float64),
    )

    assert resampled.contacts is not None
    np.testing.assert_array_equal(
        resampled.contacts.timestamps,
        np.array([10.0, 20.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled.contacts.state(target),
        np.array([False, True], dtype=np.bool_),
    )
    np.testing.assert_allclose(
        resampled.contacts.confidence(target),
        np.array([0.1, 0.3], dtype=np.float64),
    )


def test_mocap_track_resample_to_drops_raw_marker_frames() -> None:
    track, _ = _mocap_track()

    resampled = track.resample_to(np.array([0.5, 1.5], dtype=np.float64))

    assert track.marker_frames is not None
    assert resampled.marker_frames is None


def test_mocap_track_resample_to_rejects_empty_timestamps() -> None:
    track, _ = _mocap_track()

    with pytest.raises(ValueError, match="empty timestamps"):
        track.resample_to(np.array([], dtype=np.float64))


def test_mocap_track_resample_to_rejects_output_timestamp_length_mismatch() -> None:
    track, _ = _mocap_track()

    with pytest.raises(ValueError, match="same length"):
        track.resample_to(
            np.array([0.5, 1.5], dtype=np.float64),
            output_timestamps=np.array([10.0], dtype=np.float64),
        )