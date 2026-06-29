from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type

import pytest

from conftest import make_mocap_track, make_string_contact_track
from retarget.demo import (
    ContactTrack,
    Demonstration,
    DemonstrationView,
    Tracks,
)
from retarget.demo.mocap import MocapTrack


@dataclass(frozen=True, slots=True)
class GroundEstimationTracks(Tracks):
    mocap: MocapTrack
    contacts: ContactTrack


@dataclass(frozen=True, slots=True)
class MocapTracks(Tracks):
    mocap: MocapTrack


def _typed_demo() -> Demonstration[GroundEstimationTracks]:
    contacts, _ = make_string_contact_track(target_name="contact")
    return Demonstration(
        GroundEstimationTracks(mocap=make_mocap_track(), contacts=contacts)
    )


def test_demonstration_constructor_returns_typed_demonstration() -> None:
    demo = _typed_demo()
    assert isinstance(demo, Demonstration)
    assert isinstance(demo.tracks.mocap, MocapTrack)
    assert isinstance(demo.tracks.contacts, ContactTrack)
    assert_type(demo.tracks.mocap, MocapTrack)
    assert_type(demo.tracks.contacts, ContactTrack)


def test_tracks_mapping_and_string_getitem_agree() -> None:
    demo = _typed_demo()
    assert demo["mocap"] is demo.tracks["mocap"]
    assert "mocap" in demo
    assert "missing" not in demo
    assert demo.track_ids() == ("mocap", "contacts")
    assert tuple(demo) == ("mocap", "contacts")


def test_deep_chain_through_typed_demo() -> None:
    demo = Demonstration(MocapTracks(mocap=make_mocap_track()))
    shoe = demo.tracks.mocap.subjects.subject.segments.segment
    assert shoe.markers.heel.positions().shape == (3, 3)
    assert shoe.patches.sole.points().shape == (3, 3)


def test_slice_time_returns_demonstration_view_with_sliced_tracks() -> None:
    demo = Demonstration(MocapTracks(mocap=make_mocap_track()))
    clip = demo.slice_time(0.0, 0.2)
    assert isinstance(clip, DemonstrationView)
    assert clip.source is demo
    mocap = clip["mocap"]
    assert isinstance(mocap, MocapTrack)
    assert len(mocap.timestamps) == 2


def test_demonstration_rejects_non_track_values() -> None:
    with pytest.raises(TypeError, match="must be a Track"):
        Demonstration(MocapTracks(mocap=object()))  # type: ignore[arg-type]


def test_resample_to_preserves_reference_track() -> None:
    demo = Demonstration(MocapTracks(mocap=make_mocap_track()))
    clip = demo.slice_time(0.0, 0.2)
    resampled = clip.resample_to("mocap")
    assert resampled["mocap"] is clip["mocap"]
