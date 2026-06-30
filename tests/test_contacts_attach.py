"""Tests for attaching detected contacts/support-states to a MocapTrack and
reading them back through the patch timeline view."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from retarget.contacts import (
    classify_contacts,
    detect_contacts,
    ground_plane,
    multi_label,
)
from retarget.demo.contact import ContactTrack
from retarget.demo.resampling import ResampleMethod

from conftest import demo_segment, make_demo_patch_target, make_descending_mocap_track


def _sole(track: Any) -> Any:
    return demo_segment(track).patches.sole


def test_with_contacts_is_immutable_and_reads_through_patch() -> None:
    track = make_descending_mocap_track()
    contacts = detect_contacts(_sole(track), against=ground_plane(0.0))
    attached = track.with_contacts(contacts)

    assert track.contacts is None  # original is unchanged
    assert attached.contacts is not None

    sole = demo_segment(attached).patches.sole
    target = make_demo_patch_target("sole")
    np.testing.assert_array_equal(sole.contacts(), contacts.state(target))
    np.testing.assert_array_equal(sole.confidence(), contacts.confidence(target))


def test_with_contacts_rejects_mismatched_timestamps() -> None:
    track = make_descending_mocap_track()
    target = make_demo_patch_target("sole")
    bad = ContactTrack(
        timestamps=np.array([0.0, 1.0], dtype=np.float64),
        contacts={target: np.array([True, False], dtype=np.bool_)},
    )
    with pytest.raises(ValueError):
        track.with_contacts(bad)


def test_with_support_states_reads_through_patch() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = classify_contacts(
        _sole(track), against={"ground": ground_plane(0.0), "board": ground_plane(0.01)}
    )
    attached = track.with_support_states(states)

    sole = demo_segment(attached).patches.sole
    assert set(sole.support_contacts()) == {"ground", "board"}

    rest = (t >= 1.2) & (t <= 2.8)
    assert (sole.support_state()[rest] == "ground").all()
    multi = sole.support_state(resolve=multi_label())
    assert multi[int(np.argmin(np.abs(t - 2.0)))] == frozenset({"ground", "board"})


def test_patch_support_state_without_attachment_is_none() -> None:
    track = make_descending_mocap_track()
    sole = demo_segment(track).patches.sole
    assert sole.support_contacts() == {}
    assert all(label is None for label in sole.support_state())
    assert (sole.support_state(none="air") == "air").all()


def test_with_contacts_preserves_support_states() -> None:
    track = make_descending_mocap_track()
    states = classify_contacts(_sole(track), against={"ground": ground_plane(0.0)})
    contacts = detect_contacts(_sole(track), against=ground_plane(0.0))
    attached = track.with_support_states(states).with_contacts(contacts)
    assert attached.contacts is not None
    assert attached.support_states is not None


def test_resample_attached_contacts_preserves_booleans() -> None:
    track = make_descending_mocap_track()
    attached = track.with_contacts(detect_contacts(_sole(track), against=ground_plane(0.0)))
    coarse = np.linspace(0.0, 3.0, 31)
    resampled = attached.resample_to(coarse, method=ResampleMethod.NEAREST)
    sole = demo_segment(resampled).patches.sole
    assert sole.contacts().dtype == np.bool_
    assert sole.contacts().shape == coarse.shape


def test_slice_materializes_support_states() -> None:
    track = make_descending_mocap_track()
    states = classify_contacts(_sole(track), against={"ground": ground_plane(0.0)})
    clip = track.with_support_states(states).slice_time(1.5, 2.5)
    sole = demo_segment(clip).patches.sole
    labels = sole.support_state()
    assert labels.shape[0] == clip.timestamps.shape[0]
    assert (labels == "ground").all()
