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


def test_select_indices_materializes_and_preserves_nominal_hz() -> None:
    sole = make_demo_patch_target("sole")
    track = ContactTrack(
        timestamps=np.array([0.0, 0.1, 0.2, 0.3]),
        contacts={sole: np.array([True, False, True, False])},
        confidences={sole: np.array([0.1, 0.2, 0.3, 0.4])},
        nominal_hz_override=200.0,
    )

    selected = track.select_indices((1, 3))

    assert isinstance(selected, ContactTrack)
    np.testing.assert_array_equal(selected.timestamps, np.array([0.1, 0.3]))
    np.testing.assert_array_equal(selected.state(sole), np.array([False, False]))
    np.testing.assert_allclose(selected.confidence(sole), np.array([0.2, 0.4]))
    assert selected.nominal_hz_override == 200.0
    assert selected.nominal_hz == 200.0


def test_mocap_slice_materializes_contacts_and_preserves_nominal_hz() -> None:
    base = make_mocap_track()
    sole = make_demo_patch_target("sole")
    contacts = ContactTrack(
        timestamps=base.timestamps,
        contacts={sole: np.array([True, False, True])},
        confidences={sole: np.array([0.2, 0.4, 0.6])},
        nominal_hz_override=123.0,
    )
    track = MocapTrack(
        subjects=base.subjects,
        state=base.state,
        timestamps=base.timestamps,
        marker_frames=base.marker_frames,
        contacts=contacts,
    )

    clip = track.slice_time(0.0, 0.15)

    assert clip.contacts is not None
    assert isinstance(clip.contacts, ContactTrack)
    np.testing.assert_array_equal(clip.contacts.timestamps, np.array([0.0, 0.1]))
    np.testing.assert_array_equal(clip.contacts.state(sole), np.array([True, False]))
    np.testing.assert_allclose(clip.contacts.confidence(sole), np.array([0.2, 0.4]))
    assert clip.contacts.nominal_hz_override == 123.0


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
