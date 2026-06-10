from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from retarget.core import PatchHandle
from retarget.core.targets import PatchTarget
from demo_specs import GroundEstimationDemo, GroundEstimationTrackId
from retarget.demo.contact import ContactTrack
from retarget.demo.demo import Demonstration
from retarget.demo.mocap import MocapTrack
from conftest import DemoPatchId, DemoSegmentId, DemoSubjectId, make_mocap_track


def test_track_lookup_and_mocap_property() -> None:
    mocap = make_mocap_track()
    demo = GroundEstimationDemo(
        mocap=mocap,
        tracks={GroundEstimationTrackId.MOCAP: mocap},
    )
    assert demo.track(GroundEstimationTrackId.MOCAP) is mocap
    assert demo.mocap is mocap


def test_slice_time_and_with_track() -> None:
    mocap = make_mocap_track()
    demo = GroundEstimationDemo(
        mocap=mocap,
        tracks={GroundEstimationTrackId.MOCAP: mocap},
    )
    clip = demo.slice_time(0.0, 0.2)
    assert len(clip.mocap.timestamps) == 2
    extra = object()
    updated = demo.with_track(GroundEstimationTrackId.VIDEO, extra)
    assert updated.track(GroundEstimationTrackId.VIDEO) is extra


def test_with_contacts_attaches_contact_track() -> None:
    mocap = make_mocap_track()
    demo = GroundEstimationDemo(
        mocap=mocap,
        tracks={GroundEstimationTrackId.MOCAP: mocap},
    )
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=mocap.timestamps,
        contacts={target: np.array([True, False, True])},
    )
    demo_with_contacts = demo.with_contacts(contacts)
    assert demo_with_contacts.mocap.contacts is contacts
    mocap_track = cast(MocapTrack, demo_with_contacts.track(GroundEstimationTrackId.MOCAP))
    assert mocap_track.contacts is contacts
    assert cast(ContactTrack, demo_with_contacts.track(GroundEstimationTrackId.CONTACTS)) is contacts


def test_optional_track_properties_raise_clear_errors() -> None:
    mocap = make_mocap_track()
    demo = GroundEstimationDemo(
        mocap=mocap,
        tracks={GroundEstimationTrackId.MOCAP: mocap},
    )
    with pytest.raises(KeyError, match="No video track"):
        _ = demo.video
    with pytest.raises(KeyError, match="No SMPL track"):
        _ = demo.smpl
    with pytest.raises(KeyError, match="No contact track"):
        _ = demo.contacts


def test_resample_to_raises_not_implemented() -> None:
    mocap = make_mocap_track()
    demo = GroundEstimationDemo(
        mocap=mocap,
        tracks={GroundEstimationTrackId.MOCAP: mocap},
    )
    clip = demo.slice_time(0.0, 0.2)
    with pytest.raises(NotImplementedError, match="alignment-aware"):
        clip.resample_to(GroundEstimationTrackId.MOCAP)

    generic = Demonstration(tracks={GroundEstimationTrackId.MOCAP: mocap})
    generic_clip = generic.slice_time(0.0, 0.2)
    with pytest.raises(NotImplementedError, match="alignment-aware"):
        generic_clip.resample_to(GroundEstimationTrackId.MOCAP)
