"""Tests for the declarative ContactPlan + apply_contact_plan one-step workflow."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import pytest

from retarget.contacts import (
    ContactDetectionConfig,
    ContactPlan,
    ContactQuery,
    apply_contact_plan,
    classify_contacts,
    ground_plane,
    infer_ground,
)
from retarget.demo.mocap import MocapTrack

from conftest import (
    DemoSubjects,
    demo_segment,
    make_demo_patch_target,
    make_descending_mocap_track,
)


def _sole(track: MocapTrack[DemoSubjects]):
    return demo_segment(track).patches.sole


def test_contact_plan_rejects_empty_queries() -> None:
    with pytest.raises(ValueError):
        ContactPlan(queries=())


def test_contact_query_rejects_empty_against() -> None:
    track = make_descending_mocap_track()
    with pytest.raises(ValueError):
        ContactQuery(_sole(track), against={})


def test_single_query_plan_matches_bare_classify() -> None:
    track = make_descending_mocap_track()
    plan = ContactPlan(queries=(ContactQuery(_sole(track), against={"ground": ground_plane(0.0)}),))
    attached = apply_contact_plan(track, plan)

    via_plan = _sole(attached).support_state(none="air")
    via_bare = classify_contacts(_sole(track), against={"ground": ground_plane(0.0)}).support_state(
        make_demo_patch_target("sole"), none="air"
    )
    np.testing.assert_array_equal(via_plan, via_bare)


def test_infer_ground_via_plan_reads_ground_at_rest() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    plan = ContactPlan(queries=(ContactQuery(_sole(track), against={"ground": infer_ground()}),))
    attached = apply_contact_plan(track, plan)

    labels = _sole(attached).support_state(none="air")
    rest = (t >= 1.2) & (t <= 2.8)
    assert (labels[rest] == "ground").all()


def test_one_query_many_named_supports() -> None:
    # The payoff: one patch vs several supports -> categorical state over both.
    track = make_descending_mocap_track()
    plan = ContactPlan(
        queries=(
            ContactQuery(
                _sole(track),
                against={"ground": ground_plane(0.0), "board": ground_plane(0.01)},
            ),
        )
    )
    attached = apply_contact_plan(track, plan)
    assert set(_sole(attached).support_contacts()) == {"ground", "board"}


def test_merges_supports_across_queries_for_same_patch() -> None:
    # Two separate queries on the same patch combine onto the one attached track.
    track = make_descending_mocap_track()
    plan = ContactPlan(
        queries=(
            ContactQuery(_sole(track), against={"ground": ground_plane(0.0)}),
            ContactQuery(_sole(track), against={"board": ground_plane(0.01)}),
        )
    )
    attached = apply_contact_plan(track, plan)
    assert set(_sole(attached).support_contacts()) == {"ground", "board"}


def test_per_query_config_overrides_plan_default() -> None:
    # A per-query config (here an impossibly long min contact) beats the plan default,
    # so this query's support never registers -> all "air".
    track = make_descending_mocap_track()
    plan = ContactPlan(
        queries=(
            ContactQuery(
                _sole(track),
                against={"ground": ground_plane(0.0)},
                config=ContactDetectionConfig(min_contact_time=10.0),
            ),
        ),
        config=ContactDetectionConfig(),
    )
    attached = apply_contact_plan(track, plan)
    assert (_sole(attached).support_state(none="air") == "air").all()

    # Without the override the plan default does find ground at rest.
    plan_default = ContactPlan(
        queries=(ContactQuery(_sole(track), against={"ground": ground_plane(0.0)}),)
    )
    labels = _sole(apply_contact_plan(track, plan_default)).support_state(none="air")
    assert (labels == "ground").any()


def test_apply_contact_plan_is_immutable_and_returns_new_track() -> None:
    track = make_descending_mocap_track()
    plan = ContactPlan(queries=(ContactQuery(_sole(track), against={"ground": ground_plane(0.0)}),))
    attached = apply_contact_plan(track, plan)

    assert attached is not track
    assert _sole(track).support_contacts() == {}  # original untouched
    assert set(_sole(attached).support_contacts()) == {"ground"}


def test_apply_contact_plan_preserves_typed_subjects() -> None:
    track = make_descending_mocap_track()
    plan = ContactPlan(queries=(ContactQuery(_sole(track), against={"ground": infer_ground()}),))
    assert_type(apply_contact_plan(track, plan), MocapTrack[DemoSubjects])
