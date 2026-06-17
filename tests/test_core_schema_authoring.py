from __future__ import annotations

from dataclasses import fields
from typing import TypedDict

import numpy as np
import pytest

from retarget.core import (
    Marker,
    MarkerId,
    MarkerSetSpec,
    MarkerTarget,
    Patch,
    PatchId,
    PatchSpec,
    PatchTarget,
    RectangularRegion,
    SceneSpec,
    SceneState,
    SceneView,
    Segment,
    SegmentId,
    SegmentKey,
    SegmentPoseTrajectory,
    SegmentSpec,
    SegmentTarget,
    Subject,
    SubjectId,
    Subjects,
    Markers,
    Patches,
    Segments,
    build_scene,
    segment_external_name,
    subject_external_name,
    RigidTransform,
    Z_UP_AXES,
)
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame, marker_position


class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker


class ShoePatches(Patches):
    sole: Patch


class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


class ShoeSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


class LegacyMarkerId(MarkerId):
    HEEL = "heel"
    TOE = "toe"


class LegacyPatchId(PatchId):
    SOLE = "sole"


class LegacySegmentId(SegmentId):
    SHOE = "shoe"


class LegacySubjectId(SubjectId):
    LEFT_SHOE = "left_shoe"


class LegacyMarkers(Markers):
    heel: Marker
    toe: Marker


class LegacyPatches(Patches):
    sole: Patch


class LegacySegments(Segments):
    shoe: Segment[LegacyMarkers, LegacyPatches]


class LegacySubjects(Subjects):
    left_shoe: Subject[LegacySegments]


def _authored_scene(*, with_geometry: bool = True) -> SceneSpec:
    patch = (
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
    subjects = ShoeSubjects(
        left_shoe=Subject(
            vicon_name="Left_Shoe_Improved",
            segments=ShoeSegments(
                shoe=Segment(
                    vicon_name="Left_Shoe_Improved",
                    markers=ShoeMarkers(
                        heel=Marker(vicon_name="left_shoe_heel"),
                        toe=Marker(vicon_name="left_shoe_toe"),
                    ),
                    patches=ShoePatches(
                        sole=patch,
                    ),
                )
            )
        )
    )
    return build_scene(subjects)


def _scene_state(subject_id: SubjectId, segment_id: SegmentId) -> SceneState:
    return SceneState(
        segment_poses={
            SegmentKey(subject_id, segment_id): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),),
            )
        }
    )


def test_build_scene_compiles_generated_ids_and_patch_specs() -> None:
    scene = _authored_scene()
    assert isinstance(scene, SceneSpec)

    subject_id = scene.generated_ids.subjects.left_shoe
    assert isinstance(subject_id, SubjectId)
    assert issubclass(scene.generated_ids.subjects, SubjectId)
    assert subject_id.value == "left_shoe"
    assert subject_id.label == "left_shoe"
    assert subject_id.index == 0

    segment_type = scene.generated_ids.segments[subject_id]
    assert issubclass(segment_type, SegmentId)
    segment_id = segment_type.shoe
    assert isinstance(segment_id, SegmentId)
    assert segment_id.value == "shoe"
    assert segment_id.index == 0

    marker_type = scene.generated_ids.markers[SegmentKey(subject_id, segment_id)]
    assert issubclass(marker_type, MarkerId)
    heel_id = marker_type.heel
    assert isinstance(heel_id, MarkerId)
    assert heel_id.value == "heel"
    assert heel_id.index == 0

    patch_type = scene.generated_ids.patches[SegmentKey(subject_id, segment_id)]
    assert issubclass(patch_type, PatchId)
    sole_id = patch_type.sole
    assert isinstance(sole_id, PatchId)
    assert sole_id.value == "sole"
    assert sole_id.index == 0

    subject = scene.subject("left_shoe")
    segment = subject.segment("shoe")
    assert subject.vicon_name == "Left_Shoe_Improved"
    assert subject_external_name(subject) == "Left_Shoe_Improved"
    assert segment.vicon_name == "Left_Shoe_Improved"
    assert segment.subject_vicon_name == "Left_Shoe_Improved"
    assert segment_external_name(segment) == "Left_Shoe_Improved"
    marker_spec = segment.marker_spec(heel_id)
    assert marker_spec.vicon_name == "left_shoe_heel"
    assert segment.marker_external_name(heel_id) == "left_shoe_heel"
    assert segment.marker_from_external_name("left_shoe_heel") == heel_id

    patch_spec = segment.patch_spec(sole_id)
    assert isinstance(patch_spec, PatchSpec)
    assert patch_spec.label == "sole"
    assert patch_spec.frame == "sole_frame"
    assert isinstance(patch_spec.region, RectangularRegion)
    np.testing.assert_allclose(
        patch_spec.transform_segment_patch.rotation,
        np.eye(3, dtype=np.float64),
    )
    np.testing.assert_allclose(
        patch_spec.transform_segment_patch.translation,
        np.zeros(3, dtype=np.float64),
    )

    scene_view = SceneView(spec=scene, state=_scene_state(subject_id, segment_id))
    segment_view = scene_view.subject("left_shoe").segment("shoe")
    segment_target = segment_view.segment_target()
    marker_target = segment_view.marker_target(heel_id)
    patch_target = segment_view.patch_target(sole_id)
    assert isinstance(segment_target, SegmentTarget)
    assert isinstance(marker_target, MarkerTarget)
    assert isinstance(patch_target, PatchTarget)
    assert segment_target.subject == subject_id
    assert segment_target.segment == segment_id
    assert marker_target.subject == subject_id
    assert marker_target.handle.marker == heel_id
    assert patch_target.subject == subject_id
    assert patch_target.handle.patch == sole_id


def test_build_scene_uses_vicon_name_for_marker_lookup() -> None:
    scene = _authored_scene()
    subject_id = scene.generated_ids.subjects.left_shoe
    segment_id = scene.generated_ids.segments[subject_id].shoe
    marker_id = scene.generated_ids.markers[SegmentKey(subject_id, segment_id)].heel
    scene_view = SceneView(spec=scene, state=_scene_state(subject_id, segment_id))
    segment = scene.subject("left_shoe").segment("shoe")
    segment_view = scene_view.subject("left_shoe").segment("shoe")
    frame = ViconMarkersFrame(
        stamp_seconds=0.0,
        markers=(
            MarkerObservation(
                marker_name="left_shoe_heel",
                subject_name="Left_Shoe_Improved",
                segment_name="Left_Shoe_Improved",
                position_world=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                occluded=False,
            ),
        ),
    )
    expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    observed = marker_position(
        frame,
        subject=subject_id,
        segment=segment_view,
        marker=marker_id,
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected)

    track = MocapTrack(
        scene_spec=scene,
        state=_scene_state(subject_id, segment_id),
        timestamps=np.array([0.0], dtype=np.float64),
        marker_frames=(frame,),
    )
    mocap_subject = track.subjects["left_shoe"]
    assert mocap_subject.subject_id is subject_id
    mocap_segment = mocap_subject.segments["shoe"]
    assert mocap_segment.segment_view.segment_id is segment_id
    string_observed = mocap_segment.markers["heel"].positions()
    np.testing.assert_allclose(string_observed, expected.reshape(1, 3))
    np.testing.assert_allclose(
        mocap_segment.patches["sole"].points()[0],
        np.zeros(3, dtype=np.float64),
    )
    np.testing.assert_allclose(
        mocap_segment.patches["sole"].normals()[0],
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    positions = track.observed_marker_positions_for_segment(
        "left_shoe",
        segment,
    )
    np.testing.assert_allclose(positions[0, marker_id.index], expected)


def test_legacy_marker_lookup_still_falls_back_to_labels() -> None:
    segment = SegmentSpec(
        segment=LegacySegmentId.SHOE,
        marker_type=LegacyMarkerId,
        patch_type=LegacyPatchId,
        axis_convention=Z_UP_AXES,
        marker_set=MarkerSetSpec(marker_type=LegacyMarkerId),
    )
    assert segment_external_name(segment) == "shoe"
    frame = ViconMarkersFrame(
        stamp_seconds=0.0,
        markers=(
            MarkerObservation(
                marker_name="heel",
                subject_name="left_shoe",
                segment_name="shoe",
                position_world=np.array([4.0, 5.0, 6.0], dtype=np.float64),
                occluded=False,
            ),
        ),
    )
    expected = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    observed = marker_position(
        frame,
        subject=LegacySubjectId.LEFT_SHOE,
        segment=segment,
        marker=LegacyMarkerId.HEEL,
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected)
    assert segment.marker_external_name(LegacyMarkerId.HEEL) == "heel"
    assert segment.marker_from_external_name("heel") == LegacyMarkerId.HEEL


def test_declared_patches_compile_without_geometry_and_remain_targetable() -> None:
    scene = _authored_scene(with_geometry=False)
    subject_id = scene.generated_ids.subjects.left_shoe
    segment_id = scene.generated_ids.segments[subject_id].shoe
    sole_id = scene.generated_ids.patches[SegmentKey(subject_id, segment_id)].sole

    segment = scene.subject("left_shoe").segment("shoe")
    patch_target = segment.patch_target(sole_id)
    assert patch_target.subject == subject_id
    assert patch_target.handle.patch == sole_id
    assert segment.patch_label(sole_id) == "sole"
    assert segment.patch_frame(sole_id) is None

    with pytest.raises(KeyError, match="declared but has no calibrated geometry"):
        segment.patch_spec(sole_id)

    scene_view = SceneView(spec=scene, state=_scene_state(subject_id, segment_id))
    with pytest.raises(ValueError, match="declared but has no calibrated geometry"):
        scene_view.subject("left_shoe").segment("shoe").patch(sole_id)


def test_generated_id_names_are_sanitized_from_invalid_keys() -> None:
    class WeirdMarkers(Markers):
        heel: Marker

    class WeirdPatches(Patches):
        sole: Patch

    class WeirdSegments(Segments):
        left_shoe: Segment[WeirdMarkers, WeirdPatches]

    WeirdSubjects = TypedDict(
        "WeirdSubjects",
        {
            "left-shoe": Subject[WeirdSegments],
        },
    )

    subjects = WeirdSubjects(
        **{
            "left-shoe": Subject(
                segments=WeirdSegments(
                    left_shoe=Segment(
                        markers=WeirdMarkers(
                            heel=Marker(vicon_name="left-shoe_heel"),
                        ),
                        patches=WeirdPatches(
                            sole=Patch.rectangular(
                                label="sole",
                                transform_segment_patch=RigidTransform.identity(),
                                width=0.1,
                                height=0.2,
                            ),
                        ),
                    )
                )
            )
        }
    )
    scene = build_scene(subjects)
    subject_id = scene.generated_ids.subjects.left_shoe
    assert subject_id.value == "left-shoe"
    assert subject_id.label == "left-shoe"
    assert subject_id.index == 0


def test_generated_id_name_collisions_are_rejected() -> None:
    class WeirdMarkers(Markers):
        heel: Marker

    class WeirdPatches(Patches):
        sole: Patch

    class WeirdSegments(Segments):
        left_shoe: Segment[WeirdMarkers, WeirdPatches]

    WeirdSubjects = TypedDict(
        "WeirdSubjects",
        {
            "left-shoe": Subject[WeirdSegments],
            "left shoe": Subject[WeirdSegments],
        },
    )

    subject = Subject(
        segments=WeirdSegments(
            left_shoe=Segment(
                markers=WeirdMarkers(
                    heel=Marker(vicon_name="left_shoe_heel"),
                ),
                patches=WeirdPatches(
                    sole=Patch.rectangular(
                        label="sole",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.1,
                        height=0.2,
                    ),
                ),
            )
        )
    )
    with pytest.raises(ValueError, match="collide after sanitization"):
        build_scene(
            WeirdSubjects(
                **{
                    "left-shoe": subject,
                    "left shoe": subject,
                }
            )
        )


def test_patch_target_shape_remains_backward_compatible() -> None:
    assert tuple(field.name for field in fields(PatchTarget)) == (
        "subject",
        "handle",
    )
