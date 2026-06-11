from __future__ import annotations

import numpy as np
import pytest

from retarget.core import PatchHandle
from retarget.core.targets import PatchTarget
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.mocap import MocapTrack, MocapTrackView
from conftest import DemoPatchId, DemoSegmentId, DemoSubjectId, make_mocap_track


def test_track_slice_time_uses_view_type() -> None:
    track = make_mocap_track()
    view = track.slice_time(0.1, 0.25)
    assert type(view) is MocapTrackView
    assert view.source is track


def test_track_view_slice_time_remaps_nested_indices() -> None:
    track = make_mocap_track()
    outer = track.slice_time(0.1, 0.3)
    assert outer.indices == (1, 2)
    inner = outer.slice_time(0.15, 0.25)
    assert inner.indices == (2,)
    np.testing.assert_allclose(inner.timestamps, np.array([0.2]))


def test_track_nearest_index() -> None:
    track = make_mocap_track()
    assert track.nearest_index(0.14) == 1
    assert track.nearest_index(0.0) == 0


def test_track_view_nearest_index() -> None:
    track = make_mocap_track()
    view = track.slice_time(0.1, 0.3)
    assert view.nearest_index(0.19) == 1
    np.testing.assert_allclose(view.timestamps[1], 0.2)


def test_contact_track_state_on_full_track() -> None:
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    track = ContactTrack(
        timestamps=np.array([0.0, 0.1, 0.2]),
        contacts={target: np.array([True, False, True])},
    )
    np.testing.assert_array_equal(
        track.state(target),
        np.array([True, False, True]),
    )


def test_contact_track_view_state_on_slice() -> None:
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    track = ContactTrack(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        contacts={target: np.array([True, False, True, False])},
    )
    view = track.slice_time(0.1, 0.3)
    assert isinstance(view, ContactTrackView)
    np.testing.assert_array_equal(view.state(target), np.array([False, True]))
