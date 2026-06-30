"""A patch authored with a geometry callable (open algebra) drives the query surface.

The bound patch evaluates its geometry callable to a fungeom Face at bind time, lowers it to
a segment-local rigid frame + boundary, and the existing query methods transport that
per-frame by the segment pose — same contracts as a calibrated patch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fungeom import Face, Region2
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
class _GMarkers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker


@dataclass(frozen=True, slots=True)
class _GPatches(Patches):
    sole: Patch


@dataclass(frozen=True, slots=True)
class _GSegments(Segments):
    seg: Segment[_GMarkers, _GPatches]


@dataclass(frozen=True, slots=True)
class _GSubjects(Subjects):
    body: Subject[_GSegments]


def _sole(seg: SegmentGeometry) -> Face:
    markers = seg.markers["heel", "toe", "mid"]
    plane = markers.fit_plane()
    return Face.on(plane, Region2.hull(markers.in_frame(plane)))


def _geometry_subjects() -> _GSubjects:
    return _GSubjects(
        body=Subject(
            segments=_GSegments(
                seg=Segment(
                    markers=_GMarkers(
                        heel=Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0])),
                        toe=Marker(mocap_name="toe", position_segment=np.array([1.0, 0.0, 0.0])),
                        mid=Marker(mocap_name="mid", position_segment=np.array([0.0, 1.0, 0.0])),
                    ),
                    patches=_GPatches(sole=Patch(label="sole", geometry=_sole)),
                )
            )
        )
    )


def _track_from(poses: tuple[RigidTransform, ...]) -> MocapTrack[_GSubjects]:
    state = SceneState(segment_poses={SegmentKey("body", "seg"): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(
        subjects=_geometry_subjects(),
        state=state,
        timestamps=np.arange(len(poses), dtype=np.float64) * 0.1,
        marker_frames=None,
    )


def _track(num: int = 3) -> MocapTrack[_GSubjects]:
    return _track_from(tuple(RigidTransform.from_translation(np.array([float(i), 0.0, 0.0])) for i in range(num)))


def _sole_patch() -> Patch:
    return _track().subjects.body.segments.seg.patches.sole


def test_geometry_patch_has_geometry() -> None:
    assert _sole_patch().has_geometry() is True


def test_geometry_patch_points_are_region_centroid_transported() -> None:
    # heel/toe/mid form a triangle whose centroid is (1/3, 1/3); poses translate +x by i
    pts = _sole_patch().points()
    np.testing.assert_allclose(pts[0], [1 / 3, 1 / 3, 0.0], atol=1e-9)
    np.testing.assert_allclose(pts[2], [2 + 1 / 3, 1 / 3, 0.0], atol=1e-9)


def test_geometry_patch_normals_are_plane_normal_transported() -> None:
    nrm = _sole_patch().normals()
    # the markers lie in z=0, so the patch normal is +-z (sign is the fit's choice)
    np.testing.assert_allclose(np.abs(nrm[:, 2]), 1.0, atol=1e-9)
    np.testing.assert_allclose(nrm[:, :2], 0.0, atol=1e-9)


def test_geometry_patch_frames_orthonormal_normal_in_col2() -> None:
    frm = _sole_patch().frames()
    assert frm.shape == (3, 3, 3)
    np.testing.assert_allclose(frm[0] @ frm[0].T, np.eye(3), atol=1e-9)  # orthonormal
    np.testing.assert_allclose(np.abs(frm[0][:, 2]), [0.0, 0.0, 1.0], atol=1e-9)  # col 2 = normal


def test_geometry_patch_boundary_is_hull_vertices_transported() -> None:
    bnd = _sole_patch().boundary_points()
    assert bnd.shape == (3, 3, 3)  # T=3, K=3 hull vertices, 3D
    corners0 = bnd[0]  # at frame 0 the pose is identity translation 0
    for expected in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
        assert np.any(np.all(np.isclose(corners0, expected, atol=1e-9), axis=1)), expected


def test_geometry_patch_boundary_transports_under_rotation() -> None:
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    trans = np.array([1.0, 2.0, 3.0])
    rotated = (
        _track_from((RigidTransform.from_rotation_translation(rotation=rot, translation=trans),))
        .subjects.body
        .segments.seg
        .patches.sole
        .boundary_points()[0]
    )
    identity = (
        _track_from((RigidTransform.identity(),)).subjects.body.segments.seg.patches.sole.boundary_points()[0]
    )
    # the rotated-pose boundary is the identity-pose (segment-local) boundary rigidly transported
    np.testing.assert_allclose(rotated, identity @ rot.T + trans, atol=1e-9)
