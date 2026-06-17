from __future__ import annotations

import numpy as np

from conftest import make_demo_patch_target, make_mocap_track
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.mocap import MocapTrack


def test_slice_time_returns_materialized_mocap_track() -> None:
    track = make_mocap_track()
    clip = track.slice_time(0.1, 0.25)
    assert isinstance(clip, MocapTrack)
    np.testing.assert_allclose(clip.timestamps, np.array([0.1, 0.2]))


def test_nested_slice_time_narrows_timestamps() -> None:
    track = make_mocap_track()
    outer = track.slice_time(0.1, 0.3)
    np.testing.assert_allclose(outer.timestamps, np.array([0.1, 0.2]))
    inner = outer.slice_time(0.15, 0.25)
    np.testing.assert_allclose(inner.timestamps, np.array([0.2]))


def test_track_nearest_index() -> None:
    track = make_mocap_track()
    assert track.nearest_index(0.14) == 1
    assert track.nearest_index(0.0) == 0


def test_sliced_track_nearest_index() -> None:
    track = make_mocap_track().slice_time(0.1, 0.3)
    assert track.nearest_index(0.19) == 1
    np.testing.assert_allclose(track.timestamps[1], 0.2)


def test_contact_track_state_on_full_track() -> None:
    target = make_demo_patch_target("sole")
    track = ContactTrack(
        timestamps=np.array([0.0, 0.1, 0.2]),
        contacts={target: np.array([True, False, True])},
    )
    np.testing.assert_array_equal(track.state(target), np.array([True, False, True]))


def test_contact_track_view_state_on_slice() -> None:
    target = make_demo_patch_target("sole")
    track = ContactTrack(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        contacts={target: np.array([True, False, True, False])},
    )
    view = track.slice_time(0.1, 0.3)
    assert isinstance(view, ContactTrackView)
    np.testing.assert_array_equal(view.state(target), np.array([False, True]))
