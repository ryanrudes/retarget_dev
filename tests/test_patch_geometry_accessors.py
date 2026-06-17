"""Tests for the region-typed patch and its oriented-frame/boundary accessors."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    PolygonalRegion,
    RectangularRegion,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    Segments,
    Subject,
    Subjects,
)
from retarget.core.types import TimeEntityVec3, TimeMat3
from retarget.demo.mocap import MocapTrack


class ShoeMarkers(Markers):
    heel: Marker


class ShoePatches(Patches):
    sole: Patch[RectangularRegion]


class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


class ShoeSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


def _track(pose: RigidTransform) -> MocapTrack[ShoeSubjects]:
    subjects = ShoeSubjects(
        left_shoe=Subject(
            segments=ShoeSegments(
                shoe=Segment(
                    markers=ShoeMarkers(heel=Marker(mocap_name="h")),
                    patches=ShoePatches(
                        sole=Patch(
                            label="sole",
                            transform_segment_patch=RigidTransform.identity(),
                            region=RectangularRegion(width=2.0, height=4.0),
                        )
                    ),
                )
            )
        )
    )
    state = SceneState(
        segment_poses={
            SegmentKey("left_shoe", "shoe"): SegmentPoseTrajectory(poses=(pose,))
        }
    )
    return MocapTrack(subjects=subjects, state=state, timestamps=np.array([0.0]))


def _sole(pose: RigidTransform) -> Patch[RectangularRegion]:
    return _track(pose).subjects["left_shoe"].segments["shoe"].patches["sole"]


def test_region_is_statically_rectangular() -> None:
    sole = _sole(RigidTransform.identity())
    assert_type(sole, Patch[RectangularRegion])
    assert_type(sole.region, RectangularRegion)
    assert sole.region.width == 2.0
    assert sole.region.height == 4.0


def test_frames_returns_world_orientation() -> None:
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    pose = RigidTransform.from_rotation_translation(
        rotation=rotation, translation=np.array([1.0, 2.0, 3.0])
    )
    sole = _sole(pose)
    frames = sole.frames()
    assert_type(frames, TimeMat3)
    assert frames.shape == (1, 3, 3)
    # Patch frame is identity, so world frame equals the segment world rotation.
    np.testing.assert_allclose(frames[0], rotation, atol=1e-12)
    # Third column matches normals().
    np.testing.assert_allclose(frames[0][:, 2], sole.normals()[0], atol=1e-12)


def test_boundary_points_are_oriented_rectangle_corners() -> None:
    sole = _sole(RigidTransform.identity())
    corners = sole.boundary_points()
    assert_type(corners, TimeEntityVec3)
    assert corners.shape == (1, 4, 3)
    expected = np.array(
        [
            [-1.0, -2.0, 0.0],
            [1.0, -2.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
        ]
    )
    np.testing.assert_allclose(corners[0], expected, atol=1e-12)


def test_boundary_points_follow_world_pose() -> None:
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    translation = np.array([1.0, 2.0, 3.0])
    pose = RigidTransform.from_rotation_translation(
        rotation=rotation, translation=translation
    )
    sole = _sole(pose)
    corners = sole.boundary_points()[0]
    local = np.array(
        [
            [-1.0, -2.0, 0.0],
            [1.0, -2.0, 0.0],
            [1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
        ]
    )
    expected = local @ rotation.T + translation
    np.testing.assert_allclose(corners, expected, atol=1e-12)


def test_boundary_points_without_region_raises() -> None:
    track = _track(RigidTransform.identity())
    sole = track.subjects["left_shoe"].segments["shoe"].patches["sole"]
    object.__setattr__(sole, "region", None)
    with pytest.raises(ValueError, match="no contact region"):
        sole.boundary_points()


def test_rectangular_region_boundary_vertices() -> None:
    region = RectangularRegion(width=2.0, height=4.0)
    np.testing.assert_allclose(
        region.boundary(),
        np.array([[-1.0, -2.0], [1.0, -2.0], [1.0, 2.0], [-1.0, 2.0]]),
    )


def test_polygonal_region_boundary_is_its_vertices() -> None:
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    region = PolygonalRegion(vertices=vertices)
    np.testing.assert_allclose(region.boundary(), vertices)
