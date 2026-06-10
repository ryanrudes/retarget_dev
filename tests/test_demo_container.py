from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from retarget.core import PatchHandle
from retarget.core.targets import PatchTarget
from demo_specs import GroundEstimationTrackId
from retarget.demo.contact import ContactTrack
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.mocap import MocapTrack, MocapTrackView
from conftest import DemoPatchId, DemoSegmentId, DemoSubjectId, make_mocap_track


def _make_demo(mocap: MocapTrack | None = None) -> Demonstration[GroundEstimationTrackId]:
    track = mocap if mocap is not None else make_mocap_track()
    return Demonstration(tracks={GroundEstimationTrackId.MOCAP: track})


def _mocap_track(value: object) -> MocapTrack | MocapTrackView:
    if not isinstance(value, MocapTrack | MocapTrackView):
        raise TypeError("expected a mocap track or view")
    return value


def test_load_pattern_returns_demonstration() -> None:
    demo = _make_demo()
    assert isinstance(demo, Demonstration)
    mocap = _mocap_track(demo.track(GroundEstimationTrackId.MOCAP))
    assert isinstance(mocap, MocapTrack)


def test_track_lookup_by_enum_id() -> None:
    demo = _make_demo()
    mocap = make_mocap_track()
    demo = Demonstration(tracks={GroundEstimationTrackId.MOCAP: mocap})
    assert demo.track(GroundEstimationTrackId.MOCAP) is mocap


def test_slice_time_and_with_track() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.2)
    assert isinstance(clip, DemonstrationView)
    mocap = _mocap_track(clip.track(GroundEstimationTrackId.MOCAP))
    assert len(mocap.timestamps) == 2
    extra = object()
    updated = demo.with_track(GroundEstimationTrackId.VIDEO, extra)
    assert updated.track(GroundEstimationTrackId.VIDEO) is extra


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
    demo_with_contacts = (
        demo
        .with_track(GroundEstimationTrackId.MOCAP, mocap_with_contacts)
        .with_track(GroundEstimationTrackId.CONTACTS, contacts)
    )
    mocap_track = _mocap_track(demo_with_contacts.track(GroundEstimationTrackId.MOCAP))
    assert mocap_track.contacts is contacts
    assert cast(
        ContactTrack, demo_with_contacts.track(GroundEstimationTrackId.CONTACTS)
    ) is contacts


def test_resample_to_raises_not_implemented() -> None:
    demo = _make_demo()
    clip = demo.slice_time(0.0, 0.2)
    with pytest.raises(NotImplementedError, match="alignment-aware"):
        clip.resample_to(GroundEstimationTrackId.MOCAP)
