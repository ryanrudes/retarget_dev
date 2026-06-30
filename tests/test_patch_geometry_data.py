"""A patch authored as fungeom DATA over free-variable marker symbols (``m.rest``) drives the
query surface *identically* to the equivalent geometry callable -- the "patch is data" form of the
open algebra, resolved at bind time via fungeom's ``Face.bind(env)`` (>= 0.4.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fungeom import Face, Point3Bundle, Region2
from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    Segments,
    Subject,
    Subjects,
)
from retarget.core.geometry import SegmentGeometry
from retarget.demo.mocap import MocapTrack


@dataclass(frozen=True, slots=True)
class _Markers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker


@dataclass(frozen=True, slots=True)
class _Patches(Patches):
    from_callable: Patch
    from_data: Patch


@dataclass(frozen=True, slots=True)
class _Segs(Segments):
    seg: Segment[_Markers, _Patches]


@dataclass(frozen=True, slots=True)
class _Subs(Subjects):
    body: Subject[_Segs]


def _callable_sole(seg: SegmentGeometry) -> Face:
    markers = seg.markers["heel", "toe", "mid"]
    plane = markers.fit_plane()
    return Face.on(plane, Region2.hull(markers.in_frame(plane)))


def _subjects() -> _Subs:
    heel = Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0]))
    toe = Marker(mocap_name="toe", position_segment=np.array([1.0, 0.0, 0.0]))
    mid = Marker(mocap_name="mid", position_segment=np.array([0.0, 1.0, 0.0]))

    # The data form: a Face over the markers' free-variable rest points -- no strings, no callable.
    # A misspelled symbol (e.g. `hel.rest`) is a NameError here, not a silent bind-time KeyError.
    cloud = Point3Bundle.of([heel.rest, toe.rest, mid.rest])
    plane = cloud.fit_plane()
    data_face = Face.on(plane, Region2.hull(cloud.in_frame(plane)))

    return _Subs(
        body=Subject(
            segments=_Segs(
                seg=Segment(
                    markers=_Markers(heel=heel, toe=toe, mid=mid),
                    patches=_Patches(
                        from_callable=Patch(label="c", geometry=_callable_sole),
                        from_data=Patch(label="d", geometry=data_face),
                    ),
                )
            )
        )
    )


def _track() -> MocapTrack[_Subs]:
    poses = tuple(RigidTransform.from_translation(np.array([float(i), 0.0, 0.0])) for i in range(3))
    state = SceneState(segment_poses={SegmentKey("body", "seg"): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(subjects=_subjects(), state=state, timestamps=np.arange(3.0) * 0.1, marker_frames=None)


def test_data_patch_equals_callable_patch() -> None:
    seg = _track().subjects.body.segments.seg
    from_callable = seg.patches.from_callable
    from_data = seg.patches.from_data
    # bind-time free-variable resolution yields a byte-identical Face through the whole pipeline
    np.testing.assert_allclose(from_data.points(), from_callable.points(), atol=1e-12)
    np.testing.assert_allclose(from_data.normals(), from_callable.normals(), atol=1e-12)
    np.testing.assert_allclose(from_data.boundary_points(), from_callable.boundary_points(), atol=1e-12)


def test_data_patch_centroid_is_triangle_centroid_transported() -> None:
    # heel/toe/mid form a triangle whose centroid is (1/3, 1/3); poses translate +x by i
    pts = _track().subjects.body.segments.seg.patches.from_data.points()
    np.testing.assert_allclose(pts[0], [1 / 3, 1 / 3, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[2], [2 + 1 / 3, 1 / 3, 0.0], atol=1e-9)


def test_marker_rest_is_a_free_variable_identified_by_the_marker() -> None:
    heel = Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0]))
    toe = Marker(mocap_name="toe", position_segment=np.array([1.0, 0.0, 0.0]))
    face = Face.on(
        Point3Bundle.of([heel.rest, toe.rest, heel.rest]).fit_plane(),
        Region2.rectangle(1.0, 1.0),
    )
    # the Face is genuinely partial until bound, and its frees are the marker identities
    assert face.free_variables() == frozenset({heel, toe})
    with pytest.raises(Exception):  # noqa: B017 - resolving an unbound free must fail
        face.plane().resolve()
