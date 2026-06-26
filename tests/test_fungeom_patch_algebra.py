"""The open patch algebra running on a real bound segment (Stage-2 capability proof).

A patch is authored as a callable over the segment geometry that returns a fungeom Face
(oriented plane + bounded region). This exercises the verified substrate (Region2/Face/the
Plane<->2D bridge) end-to-end against retarget's own data, without touching the public Patch
schema (that change is the sign-off-gated Stage-2 step).
"""

from __future__ import annotations

import pytest
from conftest import demo_segment, make_mocap_track

import retarget.fungeom as fg
from fungeom import Face, Point3, Region2


def _sole(seg: fg.SegmentGeometry) -> Face:
    markers = seg.markers["heel", "toe", "mid"]
    plane = markers.fit_plane()
    region = Region2.hull(markers.in_frame(plane))
    return Face.on(plane, region)


def test_patch_face_open_algebra_on_real_segment() -> None:
    track = make_mocap_track(num_timesteps=3)
    face = fg.patch_face(demo_segment(track), _sole)

    # heel [0,0,0], toe [1,0,0], mid [0,1,0] form a right triangle of area 0.5 in the plane
    assert face.region().area().resolve() == pytest.approx(0.5)

    # footprint membership (normal offset ignored): inside vs. outside the triangle
    assert face.contains(Point3.at(0.2, 0.2, 1.0)).resolve() is True
    assert face.contains(Point3.at(0.6, 0.6, 1.0)).resolve() is False

    # bounded-patch clearance straight above an interior point is the height
    assert face.clearance(Point3.at(0.2, 0.2, 1.0)).resolve() == pytest.approx(1.0)


def test_patch_face_with_offset_and_hole() -> None:
    # the full open-algebra shape: erode, then punch a hole — area must drop accordingly
    def holed(seg: fg.SegmentGeometry) -> Face:
        markers = seg.markers["heel", "toe", "mid"]
        plane = markers.fit_plane()
        region = Region2.hull(markers.in_frame(plane)).offset(-0.05)
        centroid = region.centroid().resolve()
        region = region.difference(Region2.disc(0.02, center=(float(centroid.coord[0]), float(centroid.coord[1]))))
        return Face.on(plane, region)

    track = make_mocap_track(num_timesteps=3)
    face = fg.patch_face(demo_segment(track), holed)
    area = face.region().area().resolve()
    # eroded triangle (< 0.5) minus a small disc (pi*0.02**2 ~ 0.00126)
    assert 0.0 < area < 0.5
