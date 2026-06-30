"""Tests for support-state classification (multi-contact + customizable resolvers)."""

from __future__ import annotations

from typing import Any

import numpy as np

from retarget.contacts import (
    ContactDetector,
    SupportStateTrack,
    classify_contacts,
    ground_plane,
    highest_score,
    infer_ground,
    multi_label,
    priority,
)
from retarget.demo.resampling import ResampleMethod

from conftest import demo_segment, make_demo_patch_target, make_descending_mocap_track


def _sole(track: Any) -> Any:
    return demo_segment(track).patches.sole


def _classify_ground_and_board(track: Any) -> SupportStateTrack:
    # Two near-coincident floors: while the foot rests on z=0 it contacts both.
    return classify_contacts(
        _sole(track),
        against={"ground": ground_plane(0.0), "board": ground_plane(0.01)},
    )


def test_classify_returns_multi_contact_substrate() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = _classify_ground_and_board(track)
    assert isinstance(states, SupportStateTrack)

    target = make_demo_patch_target("sole")
    per_support = states.support_contacts(target)
    assert set(per_support) == {"ground", "board"}

    rest = (t >= 1.2) & (t <= 2.8)
    # Both supports are in contact simultaneously while resting.
    assert per_support["ground"][rest].all()
    assert per_support["board"][rest].all()


def test_support_state_default_is_most_confident_not_declared_order() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    # "high" (5 cm up) is declared first but the foot rests on z=0, so the most-confident
    # support is "ground" -- the default resolver picks it over the declared-first one.
    states = classify_contacts(
        _sole(track), against={"high": ground_plane(0.05), "ground": ground_plane(0.0)}
    )
    target = make_demo_patch_target("sole")

    labels = states.support_state(target, none="air")
    rest = (t >= 1.2) & (t <= 2.8)
    assert (labels[rest] == "ground").all()  # most confident wins, not declared order
    assert labels[np.argmin(np.abs(t - 0.3))] == "air"  # airborne during descent


def test_support_state_priority_order_override() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = _classify_ground_and_board(track)
    target = make_demo_patch_target("sole")

    labels = states.support_state(target, resolve=priority(order=["board", "ground"]))
    rest = (t >= 1.2) & (t <= 2.8)
    assert (labels[rest] == "board").all()


def test_support_state_multi_label_keeps_simultaneous_contacts() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = _classify_ground_and_board(track)
    target = make_demo_patch_target("sole")

    labels = states.support_state(target, resolve=multi_label())
    rest_index = int(np.argmin(np.abs(t - 2.0)))
    assert labels[rest_index] == frozenset({"ground", "board"})


def test_support_state_highest_score_picks_a_real_support() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = _classify_ground_and_board(track)
    target = make_demo_patch_target("sole")

    labels = states.support_state(target, resolve=highest_score())
    rest = (t >= 1.2) & (t <= 2.8)
    assert set(np.unique(labels[rest])) <= {"ground", "board"}


def test_support_state_custom_resolver() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = _classify_ground_and_board(track)
    target = make_demo_patch_target("sole")

    def only_board(contacts: Any, scores: Any) -> np.ndarray:
        return np.where(contacts["board"], "board", "off")

    labels = states.support_state(target, resolve=only_board)
    rest = (t >= 1.2) & (t <= 2.8)
    assert (labels[rest] == "board").all()


def test_classify_with_inferred_support() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = classify_contacts(_sole(track), against={"floor": infer_ground()})
    labels = states.support_state(make_demo_patch_target("sole"))
    rest = (t >= 1.2) & (t <= 2.8)
    assert (labels[rest] == "floor").all()


def test_support_state_track_discrete_resample_preserves_booleans() -> None:
    track = make_descending_mocap_track()
    states = _classify_ground_and_board(track)
    coarse = np.linspace(0.0, 3.0, 31)
    resampled = states.resample_to(coarse, method=ResampleMethod.NEAREST)
    target = make_demo_patch_target("sole")
    per_support = resampled.support_contacts(target)
    assert per_support["ground"].dtype == np.bool_
    assert per_support["ground"].shape == coarse.shape


def test_support_state_track_slice_view() -> None:
    track = make_descending_mocap_track()
    states = _classify_ground_and_board(track)
    view = states.slice_time(1.5, 2.5)
    target = make_demo_patch_target("sole")
    labels = view.support_state(target)
    assert labels.shape[0] == view.timestamps.shape[0]
    assert (labels == "ground").all()


def test_support_state_reads_patch_object_without_attach() -> None:
    track = make_descending_mocap_track()
    sole = _sole(track)
    states = classify_contacts(sole, against={"ground": ground_plane(0.0)})
    target = make_demo_patch_target("sole")
    # Pass the patch you already hold -- no with_support_states / re-fetch needed.
    np.testing.assert_array_equal(
        states.support_state(sole, none="air"),
        states.support_state(target, none="air"),
    )


def test_support_state_none_label_is_yours() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    states = ContactDetector().classify(_sole(track), against={"ground": ground_plane(0.0)})
    target = make_demo_patch_target("sole")
    # No-contact reads as None by default, or whatever label you ask for.
    assert states.support_state(target)[np.argmin(np.abs(t - 0.3))] is None
    assert states.support_state(target, none="flight")[np.argmin(np.abs(t - 0.3))] == "flight"
