"""Region-aware contact: clearance = closest approach of the patch footprint."""

from __future__ import annotations

import numpy as np

from retarget.contacts import ContactDetectionConfig, classify_contacts, ground_plane
from retarget.contacts.detect import _evaluate_support, _gather_patch_data
from retarget.contacts.supports import TimeIndexedSupport

from conftest import (
    demo_segment,
    make_demo_patch_target,
    make_descending_mocap_track,
)


def test_evaluate_support_takes_min_clearance_over_samples() -> None:
    target = make_demo_patch_target("sole")
    n = 4
    support = TimeIndexedSupport(
        origins=np.zeros((n, 3)), normals=np.tile([0.0, 0.0, 1.0], (n, 1)), name="floor"
    )
    center = np.tile([0.0, 0.0, 0.05], (n, 1))  # 5cm above the z=0 plane

    flat = center[:, None, :]  # one sample == center
    clr_flat, _ = _evaluate_support(support, flat, center, target)
    np.testing.assert_allclose(clr_flat, 0.05)

    # corners at z = 0, 0.1, 0, 0.1 + the 0.05 center -> closest approach is 0.0
    tilted = np.zeros((n, 5, 3))
    tilted[:, :, 2] = [0.0, 0.1, 0.0, 0.1, 0.05]
    clr_tilted, _ = _evaluate_support(support, tilted, center, target)
    np.testing.assert_allclose(clr_tilted, 0.0)


def test_region_contact_config_defaults_on_and_threads() -> None:
    assert ContactDetectionConfig().region_contact is True
    assert ContactDetectionConfig().resolve().region_contact is True
    assert ContactDetectionConfig(region_contact=False).resolve().region_contact is False


def test_gather_patch_data_samples_the_footprint() -> None:
    track = make_descending_mocap_track()
    sole = demo_segment(track).patches["sole"]
    t = len(track.timestamps)

    on = _gather_patch_data(sole, ContactDetectionConfig(region_contact=True).resolve())
    off = _gather_patch_data(sole, ContactDetectionConfig(region_contact=False).resolve())

    assert on.sample_points.shape == (t, 5, 3)  # 4 rectangle corners + center
    assert off.sample_points.shape == (t, 1, 3)  # center only
    np.testing.assert_allclose(on.sample_points[:, :4], np.asarray(sole.boundary_points()))
    np.testing.assert_allclose(on.sample_points[:, 4], np.asarray(sole.points()))
    np.testing.assert_allclose(off.sample_points[:, 0], np.asarray(sole.points()))


def _labels(track, region: bool):
    states = classify_contacts(
        demo_segment(track).patches["sole"],
        against={"ground": ground_plane(0.0)},
        config=ContactDetectionConfig(region_contact=region),
    )
    return states.support_state(make_demo_patch_target("sole"), none="air")


def test_flat_patch_region_aware_equals_point_based() -> None:
    # The descending sole stays parallel to the floor, so every corner shares the
    # center's clearance -- region-aware must reduce exactly to point-based.
    track = make_descending_mocap_track()
    np.testing.assert_array_equal(_labels(track, region=True), _labels(track, region=False))
