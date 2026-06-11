

from __future__ import annotations

import numpy as np

from retarget.core.targets import PatchTarget
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.resampling import ResampleMethod


def _target(name: str = "patch") -> PatchTarget[str]:
    return PatchTarget(subject="subject", handle=name)


def _contact_track() -> tuple[ContactTrack, PatchTarget[str]]:
    target = _target()
    track = ContactTrack(
        timestamps=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        contacts={
            target: np.array([False, True, True, False], dtype=np.bool_),
        },
        confidences={
            target: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        },
    )
    return track, target


def test_contact_track_resample_to_nearest() -> None:
    track, target = _contact_track()

    resampled = track.resample_to(
        np.array([0.1, 0.9, 2.6], dtype=np.float64),
        method=ResampleMethod.NEAREST,
    )

    assert isinstance(resampled, ContactTrack)
    np.testing.assert_array_equal(resampled.timestamps, np.array([0.1, 0.9, 2.6]))
    np.testing.assert_array_equal(
        resampled.state(target),
        np.array([False, True, False], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled.confidence(target),
        np.array([0.1, 0.2, 0.4], dtype=np.float64),
    )


def test_contact_track_resample_to_previous() -> None:
    track, target = _contact_track()

    resampled = track.resample_to(
        np.array([0.1, 1.0, 2.6], dtype=np.float64),
        method=ResampleMethod.PREVIOUS,
    )

    np.testing.assert_array_equal(
        resampled.state(target),
        np.array([False, True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled.confidence(target),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )


def test_contact_track_resample_to_accepts_string_method() -> None:
    track, target = _contact_track()

    resampled = track.resample_to(
        np.array([0.1, 1.6], dtype=np.float64),
        method="previous",
    )

    np.testing.assert_array_equal(
        resampled.state(target),
        np.array([False, True], dtype=np.bool_),
    )


def test_contact_track_view_resample_uses_visible_time_basis() -> None:
    track, target = _contact_track()
    view = track.slice_time(1.0, 4.0)

    assert isinstance(view, ContactTrackView)
    resampled = view.resample_to(
        np.array([1.1, 2.6], dtype=np.float64),
        method=ResampleMethod.NEAREST,
    )

    np.testing.assert_array_equal(resampled.timestamps, np.array([1.1, 2.6]))
    np.testing.assert_array_equal(
        resampled.state(target),
        np.array([True, False], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled.confidence(target),
        np.array([0.2, 0.4], dtype=np.float64),
    )


def test_contact_track_view_resample_to_previous_uses_visible_arrays() -> None:
    track, target = _contact_track()
    view = track.slice_time(1.0, 4.0)

    assert isinstance(view, ContactTrackView)
    resampled = view.resample_to(
        np.array([1.1, 2.6], dtype=np.float64),
        method=ResampleMethod.PREVIOUS,
    )

    np.testing.assert_array_equal(
        resampled.state(target),
        np.array([True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled.confidence(target),
        np.array([0.2, 0.3], dtype=np.float64),
    )