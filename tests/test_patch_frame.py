"""Tests for pluggable patch plane/origin/extent (core/resolvers)."""

from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    Patch,
    at_marker,
    axis_normal,
    bounding_box,
    bounding_box_center,
    calibrate_patch_transform,
    centroid,
    fixed,
    plane_from,
)
from retarget.core.schema.patch import resolve_patch_calibration

# Plane markers fit the z=0 plane (normal +z); their centroid (1/3, 1/3, 0) is
# deliberately off-center so we can prove the origin does NOT come from them.
# Region markers form a 0.4 x 0.1 rectangle centered on the segment origin, lifted
# 0.1 above the plane.
BODY = {
    "p1": np.array([0.0, 0.0, 0.0]),
    "p2": np.array([1.0, 0.0, 0.0]),
    "p3": np.array([0.0, 1.0, 0.0]),
    "r1": np.array([-0.2, -0.05, 0.1]),
    "r2": np.array([0.2, -0.05, 0.1]),
    "r3": np.array([0.2, 0.05, 0.1]),
    "r4": np.array([-0.2, 0.05, 0.1]),
}
PLANE = ("p1", "p2", "p3")
REGION = ("r1", "r2", "r3", "r4")


def _fit(**kwargs):
    return calibrate_patch_transform(
        marker_positions_segment=BODY,
        plane=plane_from(*PLANE),
        **kwargs,
    )


def test_bounding_box_extent_auto_fits_and_self_centers() -> None:
    # extent alone -> origin defaults to the region's bbox center; the box bounds it.
    transform, region = _fit(extent=bounding_box(*REGION))
    np.testing.assert_allclose(region.width, 0.4, atol=1e-9)
    np.testing.assert_allclose(region.height, 0.1, atol=1e-9)
    # origin lands at the foot center (segment origin), NOT the plane centroid (1/3,1/3,0).
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.0], atol=1e-9)


def test_padding_grows_the_box() -> None:
    _, region = _fit(extent=bounding_box(*REGION, padding=0.01))
    np.testing.assert_allclose([region.width, region.height], [0.42, 0.12], atol=1e-9)


def test_at_marker_origin_projects_marker_onto_plane() -> None:
    transform, _ = _fit(origin=at_marker("r1"), extent=fixed(1.0, 1.0))
    np.testing.assert_allclose(transform.translation, [-0.2, -0.05, 0.0], atol=1e-9)


def test_centroid_vs_bounding_box_center_differ() -> None:
    body = {
        **{k: BODY[k] for k in PLANE},
        "a": np.array([0.0, 0.0, 0.1]),
        "b": np.array([0.3, 0.0, 0.1]),
        "c": np.array([0.3, 0.1, 0.1]),  # clustered toward +x,+y
    }
    bb = calibrate_patch_transform(
        marker_positions_segment=body, plane=plane_from(*PLANE),
        origin=bounding_box_center("a", "b", "c"), extent=fixed(1.0, 1.0),
    )[0].translation
    cen = calibrate_patch_transform(
        marker_positions_segment=body, plane=plane_from(*PLANE),
        origin=centroid("a", "b", "c"), extent=fixed(1.0, 1.0),
    )[0].translation
    np.testing.assert_allclose(bb, [0.15, 0.05, 0.0], atol=1e-9)          # bbox center
    np.testing.assert_allclose(cen, [0.2, 1.0 / 30.0, 0.0], atol=1e-9)    # mean
    assert not np.allclose(bb, cen)


def test_no_origin_with_fixed_extent_is_plane_centroid() -> None:
    # No origin + fixed extent -> origin = plane-marker centroid.
    transform, region = _fit(extent=fixed(0.4, 0.1))
    np.testing.assert_allclose(transform.translation, [1.0 / 3.0, 1.0 / 3.0, 0.0], atol=1e-9)
    np.testing.assert_allclose([region.width, region.height], [0.4, 0.1])


def test_surface_offset_shifts_along_normal() -> None:
    transform, _ = _fit(extent=bounding_box(*REGION), normal=axis_normal(offset=0.05))
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.05], atol=1e-9)


def test_planar_resolves_region_through_calibration() -> None:
    patch = Patch.planar("sole", plane=plane_from(*PLANE), extent=bounding_box(*REGION))
    assert patch.region is None  # auto-fit extent is deferred to bind time
    transform, region = resolve_patch_calibration(
        patch.calibration, BODY, subject="s", segment="seg", patch_name="sole"
    )
    np.testing.assert_allclose([region.width, region.height], [0.4, 0.1], atol=1e-9)


def test_planar_fixed_extent_sets_region_eagerly() -> None:
    patch = Patch.planar("sole", plane=plane_from(*PLANE), extent=fixed(0.4, 0.1))
    assert patch.region is not None
    np.testing.assert_allclose([patch.region.width, patch.region.height], [0.4, 0.1])


def test_resolve_validates_region_marker_present() -> None:
    patch = Patch.planar("sole", plane=plane_from(*PLANE), extent=bounding_box("missing_marker"))
    with pytest.raises(ValueError, match="missing_marker"):
        resolve_patch_calibration(patch.calibration, BODY, subject="s", segment="g", patch_name="sole")


def test_origin_offset_is_composable() -> None:
    # bbox center of REGION is the segment origin; .offset nudges it in the plane.
    transform, _ = _fit(
        origin=bounding_box_center(*REGION).offset(0.1, 0.2), extent=fixed(1.0, 1.0)
    )
    np.testing.assert_allclose(transform.translation, [0.1, 0.2, 0.0], atol=1e-9)


def test_extent_grow_is_composable_and_keeps_autocenter() -> None:
    transform, region = _fit(extent=bounding_box(*REGION).grow(0.1, 0.05))
    np.testing.assert_allclose([region.width, region.height], [0.5, 0.15], atol=1e-9)
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.0], atol=1e-9)
