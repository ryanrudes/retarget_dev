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
    RectangularRegion,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    SegmentTarget,
    Segments,
    Subject,
    Subjects,
    axis_normal,
    bind_scene,
    fixed,
    plane_from,
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
        Patch(
            label="sole",
            transform_segment_patch=RigidTransform.identity(),
            region=RectangularRegion(width=0.10, height=0.25),
            frame="sole_frame",
        )
        if with_geometry
        else Patch(label="sole")
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
    assert subject.mocap_name == "Left_Shoe_Improved"
    assert shoe.mocap_name == "Left_Shoe_Improved"
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
    shoe = bind_scene(_body_model_subjects())["left_shoe"].segments["shoe"]
    np.testing.assert_allclose(
        shoe.markers["heel"].position_segment, np.array([0.5, 0.0, 0.0])
    )
    np.testing.assert_allclose(
        shoe.markers["toe"].position_segment, np.array([1.0, 0.0, 0.0])
    )


def test_explicit_position_segment_overrides_body_model() -> None:
    shoe = bind_scene(_body_model_subjects(override_heel=True))["left_shoe"].segments[
        "shoe"
    ]
    np.testing.assert_allclose(shoe.markers["heel"].position_segment, np.zeros(3))


class _PlaneMarkers(Markers):
    rear: Marker
    inner: Marker
    outer: Marker


class _PlanePatches(Patches):
    sole: Patch


class _PlaneSegments(Segments):
    shoe: Segment[_PlaneMarkers, _PlanePatches]


class _PlaneSubjects(Subjects):
    foot: Subject[_PlaneSegments]


def _plane_subjects(*, with_body_model: bool, normal_offset: float = 0.0) -> _PlaneSubjects:
    positions = {
        "rear": np.array([0.0, 0.0, 0.0]),
        "inner": np.array([1.0, 0.0, 0.0]),
        "outer": np.array([0.0, 1.0, 0.0]),
    }

    def marker(name: str) -> Marker:
        if with_body_model:
            return Marker(mocap_name=name)
        return Marker(mocap_name=name, position_segment=positions[name])

    return _PlaneSubjects(
        foot=Subject(
            body_model=positions if with_body_model else None,
            segments=_PlaneSegments(
                shoe=Segment(
                    markers=_PlaneMarkers(
                        rear=marker("rear"),
                        inner=marker("inner"),
                        outer=marker("outer"),
                    ),
                    patches=_PlanePatches(
                        sole=Patch.planar(
                            label="sole",
                            plane=plane_from("rear", "inner", "outer"),
                            extent=fixed(0.10, 0.25),
                            normal=axis_normal(offset=normal_offset),
                        ),
                    ),
                )
            ),
        )
    )


def test_rectangle_patch_calibrates_from_body_model_at_bind_time() -> None:
    shoe = bind_scene(_plane_subjects(with_body_model=True))["foot"].segments["shoe"]
    sole = shoe.patches["sole"]
    assert sole.has_geometry()
    transform = sole.transform_segment_patch
    assert transform is not None
    # Markers lie in z=0, centroid is their mean, outward normal points +z.
    np.testing.assert_allclose(transform.translation, np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]))
    np.testing.assert_allclose(transform.rotation[:, 2], np.array([0.0, 0.0, 1.0]))


def test_rectangle_patch_calibrates_from_explicit_positions() -> None:
    shoe = bind_scene(_plane_subjects(with_body_model=False))["foot"].segments["shoe"]
    assert shoe.patches["sole"].has_geometry()


def test_rectangle_patch_normal_offset_shifts_origin_along_normal() -> None:
    shoe = bind_scene(_plane_subjects(with_body_model=True, normal_offset=0.1))[
        "foot"
    ].segments["shoe"]
    transform = shoe.patches["sole"].transform_segment_patch
    assert transform is not None
    np.testing.assert_allclose(transform.translation[2], 0.1)


def test_rectangle_patch_without_marker_positions_raises_at_bind_time() -> None:
    subjects = _PlaneSubjects(
        foot=Subject(
            segments=_PlaneSegments(
                shoe=Segment(
                    markers=_PlaneMarkers(
                        rear=Marker(mocap_name="rear"),
                        inner=Marker(mocap_name="inner"),
                        outer=Marker(mocap_name="outer"),
                    ),
                    patches=_PlanePatches(
                        sole=Patch.planar(
                            label="sole",
                            plane=plane_from("rear", "inner", "outer"),
                            extent=fixed(0.1, 0.1),
                        ),
                    ),
                )
            )
        )
    )
    with pytest.raises(ValueError, match="cannot calibrate"):
        bind_scene(subjects)


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
