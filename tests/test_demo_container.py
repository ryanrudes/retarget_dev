from __future__ import annotations

from typing import assert_type

import numpy as np
import pytest

from retarget.core import PatchHandle
from retarget.core.targets import PatchTarget
from demo_vocab import GroundEstimationTrackId
from retarget.demo import Tracks, build_demonstration
from retarget.demo.contact import ContactTrack
from retarget.demo.authoring import TypedDemonstration
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.mocap import MocapTrack, MocapTrackView
from conftest import (
    DemoPatchId,
    DemoSegmentId,
    DemoSubjectId,
    make_mocap_track,
    make_string_contact_track,
)


class GroundEstimationTracks(Tracks):
    mocap: MocapTrack
    contacts: ContactTrack


class MocapTracks(Tracks):
    mocap: MocapTrack


def _make_demo(mocap: MocapTrack | None = None) -> Demonstration[GroundEstimationTrackId]:
    track = mocap if mocap is not None else make_mocap_track()
    return Demonstration(tracks={GroundEstimationTrackId.MOCAP: track})


def _mocap_track(value: object) -> MocapTrack | MocapTrackView:
    assert isinstance(value, MocapTrack | MocapTrackView)
    return value


def _typed_demo() -> TypedDemonstration[GroundEstimationTracks]:
    mocap = make_mocap_track()
    contacts, _ = make_string_contact_track(target_name="contact")
    return build_demonstration(
        GroundEstimationTracks(
            mocap=mocap,
            contacts=contacts,
        )
    )


def _typed_mocap_demo() -> TypedDemonstration[MocapTracks]:
    return build_demonstration(
        MocapTracks(
            mocap=make_mocap_track(),
        )
    )


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


def test_enum_keyed_demo_rejects_raw_string_track_names() -> None:
    demo = _make_demo()
    with pytest.raises(KeyError, match="mocap"):
        demo.get_track("mocap")


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


def test_typed_demonstration_compiles_track_ids_and_resolves_strings() -> None:
    demo = _typed_demo()

    assert demo._generated_ids is not None
    assert demo._generated_ids.tracks.mocap.value == "mocap"
    assert demo._generated_ids.tracks.contacts.value == "contacts"
    assert demo._generated_ids.tracks.mocap.index == 0
    assert demo._generated_ids.tracks.contacts.index == 1

    mocap = demo.get_track("mocap")
    contacts = demo.get_track("contacts")
    assert isinstance(mocap, MocapTrack)
    assert isinstance(contacts, ContactTrack)
    assert_type(demo.typed_tracks["mocap"], MocapTrack)
    assert_type(demo.typed_tracks["contacts"], ContactTrack)
    assert demo.typed_tracks["mocap"] is mocap
    assert demo.typed_tracks["contacts"] is contacts
    assert demo.get_track(demo._generated_ids.tracks.mocap) is mocap
    assert demo.get_track(GroundEstimationTrackId.MOCAP) is mocap


def test_typed_demonstration_rejects_unknown_string_track_names() -> None:
    demo = _typed_demo()

    with pytest.raises(KeyError, match="missing"):
        demo.get_track("missing")


def test_typed_demonstration_bridge_survives_slice_and_resample() -> None:
    demo = _typed_mocap_demo()
    clip = demo.slice_time(0.0, 0.2)

    assert isinstance(clip.get_track("mocap"), MocapTrackView)
    assert clip.typed_tracks["mocap"] is clip.get_track("mocap")
    assert clip.get_track(GroundEstimationTrackId.MOCAP) is clip.get_track("mocap")

    resampled = clip.resample_to("mocap")
    assert isinstance(resampled.get_track("mocap"), MocapTrackView)
    assert resampled.typed_tracks["mocap"] is resampled.get_track("mocap")
    assert resampled.get_track(GroundEstimationTrackId.MOCAP) is resampled.get_track(
        "mocap"
    )
