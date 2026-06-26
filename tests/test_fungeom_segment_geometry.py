"""Tests for the SegmentGeometry view (the patch-region callable's input surface)."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import pytest
from conftest import SEGMENT, SUBJECT, demo_segment, make_mocap_track

import retarget.fungeom as fg
from fungeom import Point3, Point3Bundle, Resolvable
from retarget.core.targets import SegmentTarget


def _seg() -> fg.SegmentGeometry:
    track = make_mocap_track(num_timesteps=3)
    return fg.segment_geometry(demo_segment(track))


def test_single_marker_is_rest_position() -> None:
    seg = _seg()
    # conftest rest positions: heel [0,0,0], toe [1,0,0], mid [0,1,0]
    np.testing.assert_allclose(fg.point_array(seg.markers["heel"]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(fg.point_array(seg.markers["toe"]), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(fg.point_array(seg.markers["mid"]), [0.0, 1.0, 0.0])


def test_tuple_access_is_bundle_with_centroid() -> None:
    seg = _seg()
    bundle = seg.markers["heel", "toe", "mid"]
    np.testing.assert_allclose(fg.point_array(bundle.centroid()), [1 / 3, 1 / 3, 0.0])


def test_fit_plane_through_marker_rest_positions() -> None:
    seg = _seg()
    plane = seg.markers["heel", "toe", "mid"].fit_plane()
    # the three rest positions are coplanar in z=0, so a point at z=1 is unit distance away
    sd = plane.signed_distance(Point3.at(0.0, 0.0, 1.0)).resolve()
    assert abs(sd) == pytest.approx(1.0)


def test_patch_callable_offset_expression() -> None:
    # the headline open-algebra shape: fit a plane through markers, then offset it
    seg = _seg()
    plane = seg.markers["heel", "toe", "mid"].fit_plane()
    shifted = plane.offset(0.5)
    base = plane.signed_distance(Point3.at(0.0, 0.0, 1.0)).resolve()
    moved = shifted.signed_distance(Point3.at(0.0, 0.0, 1.0)).resolve()
    assert abs(base - moved) == pytest.approx(0.5)


def test_names_in_declared_order_and_contains() -> None:
    seg = _seg()
    assert seg.markers.names() == ("heel", "toe", "mid")
    assert "heel" in seg.markers
    assert "nope" not in seg.markers


def test_missing_marker_raises_clear_keyerror() -> None:
    seg = _seg()
    with pytest.raises(KeyError, match="no segment-frame position"):
        _ = seg.markers["nope"]


def test_name_and_target() -> None:
    seg = _seg()
    assert seg.name == SEGMENT
    assert seg.target == SegmentTarget(subject=SUBJECT, segment=SEGMENT)


def test_all_returns_full_cloud() -> None:
    seg = _seg()
    support = fg.resolvability(seg.markers.all().support())
    assert isinstance(support, Resolvable)
    assert set(support.value.keys) == {"heel", "toe", "mid"}


def test_getitem_overload_typing() -> None:
    seg = _seg()
    # str key -> Point3, tuple key -> Point3Bundle (locks the __getitem__ overloads)
    assert_type(seg.markers["heel"], Point3)
    assert_type(seg.markers["heel", "toe"], Point3Bundle)
