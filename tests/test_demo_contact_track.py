from __future__ import annotations

import numpy as np
import pytest

from retarget.core import PatchHandle, PatchId, SegmentId, SubjectId
from retarget.core.targets import PatchTarget
from retarget.demo.contact import ContactTrack, ContactTrackView
from conftest import DemoPatchId, DemoSegmentId, DemoSubjectId, make_mocap_track

from retarget.demo.mocap import MocapTrack


class _SubjectId(SubjectId):
    SUBJECT = "subject"


class _SegmentId(SegmentId):
    SEGMENT = "segment"


class _PatchId(PatchId):
    SOLE = "sole"
    TOE = "toe"


def _target(patch: _PatchId) -> PatchTarget[_PatchId]:
    return PatchTarget(
        subject=_SubjectId.SUBJECT,
        handle=PatchHandle(segment=_SegmentId.SEGMENT, patch=patch),
    )


def test_contact_track_validates_array_lengths() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    target = _target(_PatchId.SOLE)
    with pytest.raises(ValueError, match="expected"):
        ContactTrack(
            timestamps=timestamps,
            contacts={target: np.array([True, False])},
        )


def test_contact_track_rejects_non_bool_contacts() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    target = _target(_PatchId.SOLE)
    with pytest.raises(TypeError, match="bool dtype"):
        ContactTrack(
            timestamps=timestamps,
            contacts={target: np.array([1.0, 0.0, 1.0])},
        )


def test_contact_track_rejects_unknown_confidence_target() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    sole = _target(_PatchId.SOLE)
    toe = _target(_PatchId.TOE)
    with pytest.raises(ValueError, match="not present in contacts"):
        ContactTrack(
            timestamps=timestamps,
            contacts={sole: np.array([True, False, True])},
            confidences={toe: np.array([0.5, 0.5, 0.5])},
        )


def test_contact_track_rejects_confidence_wrong_shape() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    sole = _target(_PatchId.SOLE)
    with pytest.raises(ValueError, match="expected"):
        ContactTrack(
            timestamps=timestamps,
            contacts={sole: np.array([True, False, True])},
            confidences={sole: np.array([0.5, 0.5])},
        )


def test_contact_track_rejects_confidence_non_float_dtype() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    sole = _target(_PatchId.SOLE)
    with pytest.raises(TypeError, match="floating-point"):
        ContactTrack(
            timestamps=timestamps,
            contacts={sole: np.array([True, False, True])},
            confidences={sole: np.array([1, 0, 1], dtype=np.int64)},
        )


def test_contact_track_rejects_confidence_outside_unit_interval() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    sole = _target(_PatchId.SOLE)
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ContactTrack(
            timestamps=timestamps,
            contacts={sole: np.array([True, False, True])},
            confidences={sole: np.array([0.5, 1.5, 0.5])},
        )


def test_contact_track_accepts_valid_confidence() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    sole = _target(_PatchId.SOLE)
    track = ContactTrack(
        timestamps=timestamps,
        contacts={sole: np.array([True, False, True])},
        confidences={sole: np.array([1.0, 0.0, 0.75])},
    )
    np.testing.assert_allclose(track.confidences[sole], np.array([1.0, 0.0, 0.75]))


def test_contact_track_rejects_duplicate_timestamps() -> None:
    timestamps = np.array([0.0, 0.1, 0.1])
    sole = _target(_PatchId.SOLE)
    with pytest.raises(ValueError, match="strictly increasing"):
        ContactTrack(
            timestamps=timestamps,
            contacts={sole: np.array([True, False, True])},
        )


def test_single_element_sequence_state_shape() -> None:
    timestamps = np.array([0.0, 0.1, 0.2])
    sole = _target(_PatchId.SOLE)
    track = ContactTrack(
        timestamps=timestamps,
        contacts={sole: np.array([True, False, True])},
    )
    view = ContactTrackView(source=track, indices=(0, 1, 2))
    assert view.state(sole).shape == (3,)
    assert view.state([sole]).shape == (3, 1)


def test_contact_state_lookup_and_slice() -> None:
    timestamps = np.array([0.0, 0.1, 0.2, 0.3])
    sole = _target(_PatchId.SOLE)
    toe = _target(_PatchId.TOE)
    track = ContactTrack(
        timestamps=timestamps,
        contacts={
            sole: np.array([True, False, True, False]),
            toe: np.array([False, True, False, True]),
        },
    )
    view = track.slice_time(0.1, 0.3)
    stacked = view.state([sole, toe])
    by_target = view.state([sole, toe], return_dict=True)
    assert stacked.shape == (2, 2)
    assert by_target[sole].shape == (2,)
    np.testing.assert_array_equal(stacked[:, 0], np.array([False, True]))


def test_mocap_patch_contacts_resolves_patch_target() -> None:
    track = make_mocap_track()
    timestamps = track.timestamps
    sole_target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=timestamps,
        contacts={sole_target: np.array([True, False, True])},
    )
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    segment = track.subject(DemoSubjectId.SUBJECT).segment(DemoSegmentId.SEGMENT)
    values = segment.patch_contacts(DemoPatchId.SOLE)
    np.testing.assert_array_equal(values, np.array([True, False, True]))
    stacked = segment.patch_contacts([DemoPatchId.SOLE])
    assert stacked.shape == (3, 1)
    np.testing.assert_array_equal(stacked[:, 0], values)


def test_mocap_rejects_mismatched_contact_timestamp_count() -> None:
    track = make_mocap_track()
    sole_target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=np.array([0.0, 0.1]),
        contacts={sole_target: np.array([True, False])},
    )
    with pytest.raises(ValueError, match="timestamp count"):
        MocapTrack(
            scene_spec=track.scene_spec,
            state=track.state,
            timestamps=track.timestamps,
            marker_frames=track.marker_frames,
            contacts=contacts,
        )


def test_mocap_rejects_mismatched_contact_timestamps() -> None:
    track = make_mocap_track()
    sole_target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=np.array([0.0, 0.2, 0.3]),
        contacts={sole_target: np.array([True, False, True])},
    )
    with pytest.raises(ValueError, match="must match MocapTrack timestamps"):
        MocapTrack(
            scene_spec=track.scene_spec,
            state=track.state,
            timestamps=track.timestamps,
            marker_frames=track.marker_frames,
            contacts=contacts,
        )


def test_empty_slice_patch_contacts_returns_empty_bool_array() -> None:
    track = make_mocap_track()
    sole_target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={sole_target: np.array([True, False, True])},
    )
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    segment = track.slice_time(100.0, 101.0).subject(DemoSubjectId.SUBJECT).segment(
        DemoSegmentId.SEGMENT
    )
    values = segment.patch_contacts(DemoPatchId.SOLE)
    assert values.shape == (0,)
    assert values.dtype == np.bool_


def test_patch_contacts_without_track_raises() -> None:
    segment = (
        make_mocap_track()
        .subject(DemoSubjectId.SUBJECT)
        .segment(DemoSegmentId.SEGMENT)
    )
    with pytest.raises(ValueError, match="No contact track"):
        segment.patch_contacts(DemoPatchId.SOLE)
