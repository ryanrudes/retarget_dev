from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    Marker,
    MarkerTarget,
    Markers,
    Patch,
    PatchTarget,
    Patches,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    SegmentTarget,
    Segments,
    Subject,
    Subjects,
    bind_scene,
)
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame


class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker


class ShoePatches(Patches):
    sole: Patch


class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


class ShoeSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


def _subjects(*, with_geometry: bool = True) -> ShoeSubjects:
    sole = (
        Patch.rectangular(
            label="sole",
            transform_segment_patch=RigidTransform.identity(),
            width=0.10,
            height=0.25,
            frame="sole_frame",
        )
        if with_geometry
        else Patch(label="sole")
    )
    return ShoeSubjects(
        left_shoe=Subject(
            vicon_name="Left_Shoe_Improved",
            segments=ShoeSegments(
                shoe=Segment(
                    vicon_name="Left_Shoe_Improved",
                    markers=ShoeMarkers(
                        heel=Marker(vicon_name="left_shoe_heel", position_segment=np.zeros(3)),
                        toe=Marker(vicon_name="left_shoe_toe", position_segment=np.array([1.0, 0.0, 0.0])),
                    ),
                    patches=ShoePatches(sole=sole),
                )
            ),
        )
    )


def _track(*, with_geometry: bool = True) -> MocapTrack[ShoeSubjects]:
    state = SceneState(
        segment_poses={
            SegmentKey("left_shoe", "shoe"): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),)
            )
        }
    )
    frame = ViconMarkersFrame(
        stamp_seconds=0.0,
        markers=(
            MarkerObservation(
                marker_name="left_shoe_heel",
                subject_name="Left_Shoe_Improved",
                segment_name="Left_Shoe_Improved",
                position_world=np.array([1.0, 2.0, 3.0]),
                occluded=False,
            ),
        ),
    )
    return MocapTrack(
        subjects=_subjects(with_geometry=with_geometry),
        state=state,
        timestamps=np.array([0.0]),
        marker_frames=(frame,),
    )


def test_bind_scene_exposes_targets_and_external_names() -> None:
    scene = bind_scene(_subjects())
    subject = scene["left_shoe"]
    shoe = subject.segments["shoe"]
    assert subject.external_name == "Left_Shoe_Improved"
    assert subject.vicon_name == "Left_Shoe_Improved"
    assert shoe.vicon_name == "Left_Shoe_Improved"
    assert shoe.segment_target() == SegmentTarget("left_shoe", "shoe")
    assert shoe.marker_target("heel") == MarkerTarget("left_shoe", "shoe", "heel")
    assert shoe.patch_target("sole") == PatchTarget("left_shoe", "shoe", "sole")


def test_bind_scene_unknown_marker_target_raises() -> None:
    shoe = bind_scene(_subjects())["left_shoe"].segments["shoe"]
    with pytest.raises(KeyError, match="has no marker 'missing'"):
        shoe.marker_target("missing")


def test_declaration_only_patch_is_targetable_without_geometry() -> None:
    shoe = bind_scene(_subjects(with_geometry=False))["left_shoe"].segments["shoe"]
    assert shoe.patch_target("sole") == PatchTarget("left_shoe", "shoe", "sole")
    assert shoe.patches["sole"].has_geometry() is False


def test_loaded_track_observed_and_modeled_positions() -> None:
    shoe = _track().subjects["left_shoe"].segments["shoe"]
    np.testing.assert_allclose(shoe.markers["heel"].positions()[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(shoe.markers["heel"].positions(modeled=True)[0], np.zeros(3))
    np.testing.assert_allclose(shoe.patches["sole"].points()[0], np.zeros(3))
    np.testing.assert_allclose(shoe.patches["sole"].normals()[0], np.array([0.0, 0.0, 1.0]))
    assert shoe.markers["heel"].target == MarkerTarget("left_shoe", "shoe", "heel")


def test_declaration_only_patch_points_raise_on_loaded_track() -> None:
    shoe = _track(with_geometry=False).subjects["left_shoe"].segments["shoe"]
    with pytest.raises(ValueError, match="no calibrated geometry"):
        shoe.patches["sole"].points()


def test_bind_scene_rejects_duplicate_vicon_name_within_segment() -> None:
    subjects = ShoeSubjects(
        left_shoe=Subject(
            segments=ShoeSegments(
                shoe=Segment(
                    markers=ShoeMarkers(
                        heel=Marker(vicon_name="dup"),
                        toe=Marker(vicon_name="dup"),
                    ),
                    patches=ShoePatches(sole=Patch(label="sole")),
                )
            )
        )
    )
    with pytest.raises(ValueError, match="Duplicate Marker.vicon_name"):
        bind_scene(subjects)
