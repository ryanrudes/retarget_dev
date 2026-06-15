from __future__ import annotations

import numpy as np
import pytest

from retarget.core import PatchHandle
from retarget.core.targets import PatchTarget
from demo_vocab import GroundEstimationTrackId
from retarget.demo.contact import ContactTrack
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.mocap import MocapTrack, MocapTrackView
from conftest import DemoPatchId, DemoSegmentId, DemoSubjectId, make_mocap_track


def _make_demo(mocap: MocapTrack | None = None) -> Demonstration[GroundEstimationTrackId]:
    track = mocap if mocap is not None else make_mocap_track()
    return Demonstration(tracks={GroundEstimationTrackId.MOCAP: track})


def _mocap_track(value: object) -> MocapTrack | MocapTrackView:
    assert isinstance(value, MocapTrack | MocapTrackView)
    return value


def test_load_pattern_returns_demonstration() -> None:
    demo = _make_demo()
    assert isinstance(demo, Demonstration)
    mocap = _mocap_track(demo.get_track(GroundEstimationTrackId.MOCAP))
    assert isinstance(mocap, MocapTrack)


def test_track_lookup_by_enum_id() -> None:
    demo = _make_demo()
    mocap = make_mocap_track()
    demo = Demonstration(tracks={GroundEstimationTrackId.MOCAP: mocap})
    assert demo.get_track(GroundEstimationTrackId.MOCAP) is mocap


def test_slice_time_returns_demonstration_view() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.2)
    assert isinstance(clip, DemonstrationView)
    mocap = _mocap_track(clip.get_track(GroundEstimationTrackId.MOCAP))
    assert isinstance(mocap, MocapTrackView)
    assert len(mocap.timestamps) == 2


def test_demonstration_view_is_a_demonstration() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.2)
    assert isinstance(clip, Demonstration)


def test_demonstration_view_slice_time_preserves_original_source() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.3)
    nested = clip.slice_time(0.1, 0.2)
    assert nested.source is demo


def test_demonstration_rejects_non_track_values() -> None:
    with pytest.raises(TypeError, match="must be a Track"):
        Demonstration(tracks={GroundEstimationTrackId.MOCAP: object()})  # type: ignore[dict-item]


def test_with_contacts_attaches_contact_track() -> None:
    mocap = make_mocap_track()
    demo = _make_demo(mocap)
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=mocap.timestamps,
        contacts={target: np.array([True, False, True])},
    )
    mocap_with_contacts = MocapTrack(
        scene_spec=mocap.scene_spec,
        state=mocap.state,
        timestamps=mocap.timestamps,
        marker_frames=mocap.marker_frames,
        contacts=contacts,
    )
    demo_with_contacts = Demonstration(
        tracks={
            GroundEstimationTrackId.MOCAP: mocap_with_contacts,
            GroundEstimationTrackId.CONTACTS: contacts,
        }
    )
    mocap_track = _mocap_track(
        demo_with_contacts.get_track(GroundEstimationTrackId.MOCAP)
    )
    assert mocap_track.contacts is contacts
    contact_track = demo_with_contacts.get_track(GroundEstimationTrackId.CONTACTS)
    assert isinstance(contact_track, ContactTrack)
    assert contact_track is contacts


def test_resample_to_preserves_reference_track_without_resampling() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.2)

    resampled = clip.resample_to(GroundEstimationTrackId.MOCAP)

    assert resampled[GroundEstimationTrackId.MOCAP] is clip[GroundEstimationTrackId.MOCAP]


def test_get_track_returns_mocap_track() -> None:
    mocap = make_mocap_track()
    demo = _make_demo(mocap)
    result = demo.get_track(GroundEstimationTrackId.MOCAP)
    assert isinstance(result, MocapTrack | MocapTrackView)
    assert result is mocap


def test_get_track_on_view_returns_sliced_mocap_view() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.2)
    result = clip.get_track(GroundEstimationTrackId.MOCAP)
    assert isinstance(result, MocapTrackView)
    assert len(result.timestamps) == 2
