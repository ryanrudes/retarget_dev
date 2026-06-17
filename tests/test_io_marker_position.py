from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from retarget.core import (
    Marker,
    MarkerId,
    MarkerSetSpec,
    Markers,
    Patch,
    PatchId,
    Patches,
    SceneSpec,
    SceneState,
    SceneView,
    SegmentId,
    SegmentKey,
    Segment,
    SegmentPoseTrajectory,
    SegmentSpec,
    Segments,
    SubjectId,
    Subject,
    SubjectSpec,
    Subjects,
    build_scene,
    Z_UP_AXES,
)
from retarget.core.transform import RigidTransform
from retarget.io import MarkerObservation, ViconMarkersFrame, marker_position


class _SubjectId(SubjectId):
    SUBJECT = "subject"


class _SegmentId(SegmentId):
    SEGMENT = "segment"


class _MarkerId(MarkerId):
    HEEL = "heel"


class _PatchId(PatchId):
    PATCH = "patch"


_SEGMENT_SPEC = SegmentSpec(
    segment=_SegmentId.SEGMENT,
    marker_type=_MarkerId,
    patch_type=_PatchId,
    axis_convention=Z_UP_AXES,
    marker_set=MarkerSetSpec(marker_type=_MarkerId),
)


@dataclass(frozen=True, slots=True)
class _SubjectSpec(SubjectSpec):
    segment_spec: SegmentSpec[_MarkerId, _PatchId]

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        yield self.segment_spec


@dataclass(frozen=True, slots=True)
class _SceneSpec(SceneSpec):
    subject_spec: _SubjectSpec

    def iter_subjects(self) -> Iterable[SubjectSpec]:
        yield self.subject_spec


class _AuthMarkers(Markers):
    heel: Marker


class _AuthPatches(Patches):
    sole: Patch


class _AuthSegments(Segments):
    shoe: Segment[_AuthMarkers, _AuthPatches]


class _AuthSubjects(Subjects):
    left_shoe: Subject[_AuthSegments]


def _authored_scene() -> SceneSpec:
    return build_scene(
        _AuthSubjects(
            left_shoe=Subject(
                vicon_name="Left_Shoe_Improved",
                segments=_AuthSegments(
                    shoe=Segment(
                        vicon_name="Left_Shoe_Improved",
                        markers=_AuthMarkers(
                            heel=Marker(vicon_name="left_shoe_heel"),
                        ),
                        patches=_AuthPatches(
                            sole=Patch(label="sole"),
                        ),
                    )
                ),
            )
        )
    )


def test_marker_position_accepts_segment_view() -> None:
    scene_spec = _SceneSpec(
        subject_spec=_SubjectSpec(
            subject=_SubjectId.SUBJECT,
            segment_spec=_SEGMENT_SPEC,
        )
    )
    state = SceneState(
        segment_poses={
            SegmentKey(_SubjectId.SUBJECT, _SegmentId.SEGMENT): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),),
            )
        }
    )
    scene = SceneView(spec=scene_spec, state=state)
    segment_view = scene.subject(_SubjectId.SUBJECT).segment(_SegmentId.SEGMENT)
    expected = np.array([1.0, 2.0, 3.0])
    marker_frame = ViconMarkersFrame(
        stamp_seconds=0.0,
        markers=(
            MarkerObservation(
                marker_name=_MarkerId.HEEL.label,
                subject_name=_SubjectId.SUBJECT.label,
                segment_name=_SegmentId.SEGMENT.label,
                position_world=expected,
                occluded=False,
            ),
        ),
    )
    observed = marker_position(
        marker_frame,
        segment=segment_view,
        marker=_MarkerId.HEEL,
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected)


def test_marker_position_accepts_segment_spec_and_subject() -> None:
    expected = np.array([1.0, 2.0, 3.0])
    marker_frame = ViconMarkersFrame(
        stamp_seconds=0.0,
        markers=(
            MarkerObservation(
                marker_name=_MarkerId.HEEL.label,
                subject_name=_SubjectId.SUBJECT.label,
                segment_name=_SegmentId.SEGMENT.label,
                position_world=expected,
                occluded=False,
            ),
        ),
    )
    observed = marker_position(
        marker_frame,
        subject=_SubjectId.SUBJECT,
        segment=_SEGMENT_SPEC,
        marker=_MarkerId.HEEL,
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected)


def test_marker_position_requires_subject_for_segment_spec() -> None:
    with pytest.raises(TypeError, match="subject must be provided"):
        marker_position(
            ViconMarkersFrame(stamp_seconds=0.0, markers=()),
            segment=_SEGMENT_SPEC,
            marker=_MarkerId.HEEL,
        )


def test_marker_position_uses_external_subject_segment_names_and_marker_labels() -> None:
    scene = _authored_scene()
    subject_id = scene.generated_ids.subjects.left_shoe
    segment_id = scene.generated_ids.segments[subject_id].shoe
    marker_id = scene.generated_ids.markers[SegmentKey(subject_id, segment_id)].heel

    state = SceneState(
        segment_poses={
            SegmentKey(subject_id, segment_id): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),),
            )
        }
    )
    scene_view = SceneView(spec=scene, state=state)
    segment_view = scene_view.subject("left_shoe").segment("shoe")
    segment_spec = scene.subject("left_shoe").segment("shoe")
    expected = np.array([1.0, 2.0, 3.0])
    marker_frame = ViconMarkersFrame(
        stamp_seconds=0.0,
        markers=(
            MarkerObservation(
                marker_name="left_shoe_heel",
                subject_name="Left_Shoe_Improved",
                segment_name="Left_Shoe_Improved",
                position_world=expected,
                occluded=False,
            ),
        ),
    )

    observed_view = marker_position(
        marker_frame,
        segment=segment_view,
        marker=marker_id,
    )
    observed_spec = marker_position(
        marker_frame,
        subject=subject_id,
        segment=segment_spec,
        marker=marker_id,
    )
    assert observed_view is not None
    assert observed_spec is not None
    np.testing.assert_allclose(observed_view, expected)
    np.testing.assert_allclose(observed_spec, expected)
