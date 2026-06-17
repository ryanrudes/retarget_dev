from __future__ import annotations

import numpy as np
import pytest

from conftest import demo_segment, make_demo_patch_target, make_mocap_track
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.mocap import MocapTrack


def test_contact_track_validates_array_lengths() -> None:
    with pytest.raises(ValueError, match="expected"):
        ContactTrack(
            timestamps=np.array([0.0, 0.1, 0.2]),
            contacts={make_demo_patch_target("sole"): np.array([True, False])},
        )


def test_contact_track_rejects_non_bool_contacts() -> None:
    with pytest.raises(TypeError, match="bool dtype"):
        ContactTrack(
            timestamps=np.array([0.0, 0.1, 0.2]),
            contacts={make_demo_patch_target("sole"): np.array([1.0, 0.0, 1.0])},
        )


def test_contact_track_rejects_unknown_confidence_target() -> None:
    with pytest.raises(ValueError, match="not present in contacts"):
        ContactTrack(
            timestamps=np.array([0.0, 0.1, 0.2]),
            contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
            confidences={make_demo_patch_target("toe"): np.array([0.5, 0.5, 0.5])},
        )


def test_contact_track_rejects_confidence_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ContactTrack(
            timestamps=np.array([0.0, 0.1, 0.2]),
            contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
            confidences={make_demo_patch_target("sole"): np.array([0.5, 1.5, 0.5])},
        )


def test_contact_track_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ContactTrack(
            timestamps=np.array([0.0, 0.1, 0.1]),
            contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
        )


def test_contact_state_lookup_and_slice() -> None:
    sole = make_demo_patch_target("sole")
    toe = make_demo_patch_target("toe")
    track = ContactTrack(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        contacts={
            sole: np.array([True, False, True, False]),
            toe: np.array([False, True, False, True]),
        },
    )
    view = track.slice_time(0.1, 0.3)
    assert isinstance(view, ContactTrackView)
    stacked = view.state([sole, toe])
    assert stacked.shape == (2, 2)
    np.testing.assert_array_equal(stacked[:, 0], np.array([False, True]))
    by_target = view.state([sole, toe], return_dict=True)
    assert by_target[sole].shape == (2,)


def test_mocap_patch_contacts_resolves_patch_target() -> None:
    track = make_mocap_track()
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
    )
    track = MocapTrack(
        subjects=track.subjects,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    segment = demo_segment(track)
    np.testing.assert_array_equal(
        segment.patches["sole"].contacts(), np.array([True, False, True])
    )
    assert segment.patch_contacts("sole").shape == (3, 1)


def test_mocap_rejects_mismatched_contact_timestamp_count() -> None:
    track = make_mocap_track()
    contacts = ContactTrack(
        timestamps=np.array([0.0, 0.1]),
        contacts={make_demo_patch_target("sole"): np.array([True, False])},
    )
    with pytest.raises(ValueError, match="timestamp count"):
        MocapTrack(
            subjects=track.subjects,
            state=track.state,
            timestamps=track.timestamps,
            marker_frames=track.marker_frames,
            contacts=contacts,
        )


def test_empty_slice_patch_contacts_returns_empty_bool_array() -> None:
    track = make_mocap_track()
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
    )
    track = MocapTrack(
        subjects=track.subjects,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    segment = demo_segment(track.slice_time(100.0, 101.0))
    values = segment.patches["sole"].contacts()
    assert values.shape == (0,)
    assert values.dtype == np.bool_
