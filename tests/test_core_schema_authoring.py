from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from fungeom import Direction3, Face, Plane, Region2

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
from retarget.core.geometry import SegmentGeometry
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame


@dataclass(frozen=True, slots=True)
class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker


@dataclass(frozen=True, slots=True)
class ShoePatches(Patches):
    sole: Patch


@dataclass(frozen=True, slots=True)
class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


@dataclass(frozen=True, slots=True)
class ShoeSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


def _shoe_sole_geometry(seg: SegmentGeometry) -> Face:
    # An axis-aligned patch anchored at the heel marker: plane through heel with a +z normal
    # and a fixed 0.10 x 0.25 footprint (the open-algebra form of an identity patch frame).
    plane = Plane.through(seg.markers["heel"], Direction3.of(0.0, 0.0, 1.0))
    return Face.on(plane, Region2.rectangle(0.10, 0.25))


def _subjects(*, with_geometry: bool = True) -> ShoeSubjects:
    sole = (
        Patch(label="sole", geometry=_shoe_sole_geometry, frame="sole_frame") if with_geometry else Patch(label="sole")
    )
    return ShoeSubjects(
        left_shoe=Subject(
            mocap_name="Left_Shoe_Improved",
            segments=ShoeSegments(
                shoe=Segment(
                    mocap_name="Left_Shoe_Improved",
                    markers=ShoeMarkers(
                        heel=Marker(mocap_name="left_shoe_heel", position_segment=np.zeros(3)),
                        toe=Marker(mocap_name="left_shoe_toe", position_segment=np.array([1.0, 0.0, 0.0])),
                    ),
                    patches=ShoePatches(sole=sole),
                )
            ),
        )
    )


def _track(*, with_geometry: bool = True) -> MocapTrack[ShoeSubjects]:
    state = SceneState(
        segment_poses={SegmentKey("left_shoe", "shoe"): SegmentPoseTrajectory(poses=(RigidTransform.identity(),))}
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
    subject = scene.left_shoe
    shoe = subject.segments.shoe
    assert subject.external_name == "Left_Shoe_Improved"
    assert subject.mocap_name == "Left_Shoe_Improved"
    assert shoe.mocap_name == "Left_Shoe_Improved"
    assert shoe.segment_target() == SegmentTarget("left_shoe", "shoe")
    assert shoe.marker_target("heel") == MarkerTarget("left_shoe", "shoe", "heel")
    assert shoe.patch_target("sole") == PatchTarget("left_shoe", "shoe", "sole")


def test_bind_scene_unknown_marker_target_raises() -> None:
    shoe = bind_scene(_subjects()).left_shoe.segments.shoe
    with pytest.raises(KeyError, match="has no marker 'missing'"):
        shoe.marker_target("missing")


def test_declaration_only_patch_is_targetable_without_geometry() -> None:
    shoe = bind_scene(_subjects(with_geometry=False)).left_shoe.segments.shoe
    assert shoe.patch_target("sole") == PatchTarget("left_shoe", "shoe", "sole")
    assert shoe.patches.sole.has_geometry() is False


def test_loaded_track_observed_and_modeled_positions() -> None:
    shoe = _track().subjects.left_shoe.segments.shoe
    np.testing.assert_allclose(shoe.markers.heel.positions()[0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(shoe.markers.heel.positions(modeled=True)[0], np.zeros(3))
    np.testing.assert_allclose(shoe.patches.sole.points()[0], np.zeros(3))
    np.testing.assert_allclose(shoe.patches.sole.normals()[0], np.array([0.0, 0.0, 1.0]))
    assert shoe.markers.heel.target == MarkerTarget("left_shoe", "shoe", "heel")


def test_declaration_only_patch_points_raise_on_loaded_track() -> None:
    shoe = _track(with_geometry=False).subjects.left_shoe.segments.shoe
    with pytest.raises(ValueError, match="no calibrated geometry"):
        shoe.patches.sole.points()


def _body_model_subjects(*, override_heel: bool = False) -> ShoeSubjects:
    body_model = {
        "left_shoe_heel": np.array([0.5, 0.0, 0.0]),
        "left_shoe_toe": np.array([1.0, 0.0, 0.0]),
    }
    heel = (
        Marker(mocap_name="left_shoe_heel", position_segment=np.zeros(3))
        if override_heel
        else Marker(mocap_name="left_shoe_heel")
    )
    return ShoeSubjects(
        left_shoe=Subject(
            mocap_name="Left_Shoe_Improved",
            body_model=body_model,
            segments=ShoeSegments(
                shoe=Segment(
                    mocap_name="Left_Shoe_Improved",
                    markers=ShoeMarkers(heel=heel, toe=Marker(mocap_name="left_shoe_toe")),
                    patches=ShoePatches(sole=Patch(label="sole")),
                )
            ),
        )
    )


def test_body_model_supplies_marker_segment_positions() -> None:
    shoe = bind_scene(_body_model_subjects()).left_shoe.segments.shoe
    np.testing.assert_allclose(shoe.markers.heel.position_segment, np.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(shoe.markers.toe.position_segment, np.array([1.0, 0.0, 0.0]))


def test_explicit_position_segment_overrides_body_model() -> None:
    shoe = bind_scene(_body_model_subjects(override_heel=True)).left_shoe.segments.shoe
    np.testing.assert_allclose(shoe.markers.heel.position_segment, np.zeros(3))


def test_bind_scene_rejects_duplicate_mocap_name_within_segment() -> None:
    subjects = ShoeSubjects(
        left_shoe=Subject(
            segments=ShoeSegments(
                shoe=Segment(
                    markers=ShoeMarkers(
                        heel=Marker(mocap_name="dup"),
                        toe=Marker(mocap_name="dup"),
                    ),
                    patches=ShoePatches(sole=Patch(label="sole")),
                )
            )
        )
    )
    with pytest.raises(ValueError, match="Duplicate Marker.mocap_name"):
        bind_scene(subjects)
