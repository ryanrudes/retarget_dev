"""Tests for the contact-detection driver (single-support boolean ContactTrack)."""

from __future__ import annotations

from typing import Any

import numpy as np

from retarget.contacts import (
    ContactDetector,
    ContactDetectionConfig,
    detect_contacts,
    ground_plane,
    merge_contact_tracks,
)
from retarget.contacts.supports import TimeIndexedSupport
from retarget.demo.mocap import MocapTrack

from conftest import (
    make_demo_patch_target,
    make_descending_mocap_track,
    demo_segment,
)


def _sole(track: MocapTrack[Any]) -> Any:
    return demo_segment(track).patches["sole"]


def test_detect_contacts_descent_onto_known_plane() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    contacts = detect_contacts(_sole(track), against=ground_plane(0.0))

    target = make_demo_patch_target("sole")
    mask = contacts.state(target)
    confidence = contacts.confidence(target)

    assert mask.dtype == np.bool_
    assert confidence.shape == t.shape
    assert float(confidence.min()) >= 0.0 and float(confidence.max()) <= 1.0

    # Resting on the floor -> in contact; descending in the air -> not.
    assert mask[(t >= 1.2) & (t <= 2.8)].all()
    assert not mask[np.argmin(np.abs(t - 0.3))]


def test_detect_contacts_infers_floor_when_against_none() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    contacts = detect_contacts(_sole(track))  # default infer_ground() -> fit the floor
    mask = contacts.state(make_demo_patch_target("sole"))
    assert mask[(t >= 1.2) & (t <= 2.8)].all()


def test_detect_contacts_segment_scope_skips_geometryless_patches() -> None:
    track = make_descending_mocap_track()
    segment = demo_segment(track)
    contacts = detect_contacts(segment, against=ground_plane(0.0))
    # The demo segment declares sole (calibrated) and toe (declaration-only).
    assert make_demo_patch_target("sole") in contacts.contacts
    assert make_demo_patch_target("toe") not in contacts.contacts


def test_detect_contacts_against_moving_support() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    # A moving floor that sits at z=0 the whole time is equivalent to the ground plane.
    n = len(t)
    support = TimeIndexedSupport(
        origins=np.zeros((n, 3)), normals=np.tile([0.0, 0.0, 1.0], (n, 1))
    )
    contacts = detect_contacts(_sole(track), against=support)
    mask = contacts.state(make_demo_patch_target("sole"))
    assert mask[(t >= 1.2) & (t <= 2.8)].all()


def test_min_contact_time_removes_short_rest() -> None:
    track = make_descending_mocap_track()
    # The rest lasts ~2.2s; require a longer minimum so it is dropped entirely.
    contacts = detect_contacts(
        _sole(track),
        against=ground_plane(0.0),
        config=ContactDetectionConfig(min_contact_time=5.0),
    )
    assert not contacts.state(make_demo_patch_target("sole")).any()


def test_detector_reuse_matches_function() -> None:
    track = make_descending_mocap_track()
    config = ContactDetectionConfig(contact_clearance=0.02)
    detector = ContactDetector(config)
    target = make_demo_patch_target("sole")

    via_detector = detector.detect(_sole(track), against=ground_plane(0.0))
    via_function = detect_contacts(_sole(track), against=ground_plane(0.0), config=config)
    np.testing.assert_array_equal(via_detector.state(target), via_function.state(target))


def test_merge_contact_tracks_or_combines() -> None:
    track = make_descending_mocap_track()
    sole = _sole(track)
    ground = detect_contacts(sole, against=ground_plane(0.0))
    # A floor far below is never contacted; merging should equal the ground result.
    never = detect_contacts(sole, against=ground_plane(-5.0))
    merged = merge_contact_tracks(ground, never)
    target = make_demo_patch_target("sole")
    np.testing.assert_array_equal(merged.state(target), ground.state(target))
