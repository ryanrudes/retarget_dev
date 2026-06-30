"""A patch authored as fungeom DATA over the segment's marker symbols -- a fungeom ``Face`` resolved
at bind time via ``Face.bind(env)`` (>= 0.4.0). Markers drop in directly (``SupportsPoint3``, >= 0.6.0)
or via ``m.rest``; both bind to the same segment-local geometry, transported per-frame by the pose.
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
from retarget.demo.mocap import MocapTrack


@dataclass(frozen=True, slots=True)
class _Markers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker


@dataclass(frozen=True, slots=True)
class _Patches(Patches):
    from_data: Patch
    from_markers: Patch


@dataclass(frozen=True, slots=True)
class _Segs(Segments):
    seg: Segment[_Markers, _Patches]


@dataclass(frozen=True, slots=True)
class _Subs(Subjects):
    body: Subject[_Segs]


def _subjects() -> _Subs:
    heel = Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0]))
    toe = Marker(mocap_name="toe", position_segment=np.array([1.0, 0.0, 0.0]))
    mid = Marker(mocap_name="mid", position_segment=np.array([0.0, 1.0, 0.0]))

    # The data form: a Face over the markers' free-variable rest points -- no strings, no callable.
    # A misspelled symbol (e.g. `hel.rest`) is a NameError here, not a silent bind-time KeyError.
    cloud = Point3Bundle.of([heel.rest, toe.rest, mid.rest])
    plane = cloud.fit_plane()
    data_face = Face.on(plane, Region2.hull(cloud.in_frame(plane)))

    # Same Face, but markers are passed *directly*: a Marker is a fungeom ``SupportsPoint3`` (>= 0.6.0),
    # so it coerces to its ``.rest`` free variable. This must bind identically to ``data_face``.
    markers_cloud = Point3Bundle.of([heel, toe, mid])
    markers_plane = markers_cloud.fit_plane()
    markers_face = Face.on(markers_plane, Region2.hull(markers_cloud.in_frame(markers_plane)))

    return _Subs(
        body=Subject(
            segments=_Segs(
                seg=Segment(
                    markers=_Markers(heel=heel, toe=toe, mid=mid),
                    patches=_Patches(
                        from_data=Patch(label="d", geometry=data_face),
                        from_markers=Patch(label="m", geometry=markers_face),
                    ),
                )
            )
        )
    )


def _track() -> MocapTrack[_Subs]:
    poses = tuple(RigidTransform.from_translation(np.array([float(i), 0.0, 0.0])) for i in range(3))
    state = SceneState(segment_poses={SegmentKey("body", "seg"): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(subjects=_subjects(), state=state, timestamps=np.arange(3.0) * 0.1, marker_frames=None)


def test_data_patch_geometry_is_correct() -> None:
    # heel[0,0,0]/toe[1,0,0]/mid[0,1,0] lie in z=0, so the fitted plane normal is the z-axis and the
    # footprint is the 3-vertex triangle hull -- transported per-frame by the +x-translating pose.
    from_data = _track().subjects.body.segments.seg.patches.from_data
    np.testing.assert_allclose(np.abs(from_data.normals()), [[0.0, 0.0, 1.0]] * 3, atol=1e-9)
    assert from_data.boundary_points().shape == (3, 3, 3)  # (T frames, K=3 hull vertices, 3 coords)


def test_data_patch_from_markers_equals_from_rest() -> None:
    # Passing marker symbols directly (the fungeom ``SupportsPoint3`` coercion = ``.rest``) binds
    # byte-identically to the explicit ``.rest`` data form through the whole query pipeline.
    seg = _track().subjects.body.segments.seg
    from_markers = seg.patches.from_markers
    from_data = seg.patches.from_data
    np.testing.assert_allclose(from_markers.points(), from_data.points(), atol=1e-12)
    np.testing.assert_allclose(from_markers.normals(), from_data.normals(), atol=1e-12)
    np.testing.assert_allclose(from_markers.boundary_points(), from_data.boundary_points(), atol=1e-12)


def test_marker_fungeom_point3_hook_is_rest() -> None:
    # The coercion hook a Marker exposes to fungeom is its ``.rest`` free variable: a free Point3
    # identified by the marker. (FreePoint3 has identity equality, so compare by free-variable set.)
    heel = Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0]))
    assert heel.__fungeom_point3__().free_variables() == frozenset({heel}) == heel.rest.free_variables()


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
