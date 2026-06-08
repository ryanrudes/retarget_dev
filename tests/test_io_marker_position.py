from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    MarkerId,
    MarkerSetSpec,
    PatchId,
    RigidTransform,
    SceneSpec,
    SceneState,
    SceneView,
    SegmentId,
    SegmentKey,
    SegmentPoseTrajectory,
    SegmentSpec,
    SubjectId,
    SubjectSpec,
    Z_UP_AXES,
)
from retarget.io import marker_position


class TestSubjectId(SubjectId):
    SUBJECT = "subject"


class TestSegmentId(SegmentId):
    SEGMENT = "segment"


class TestMarkerId(MarkerId):
    HEEL = "heel"


class TestPatchId(PatchId):
    SOLE = "sole"


class _TestSubjectSpec(SubjectSpec):
    def __init__(self) -> None:
        object.__setattr__(self, "subject", TestSubjectId.SUBJECT)

    def iter_segments(self):
        yield test_segment_spec


class _TestSceneSpec(SceneSpec):
    def iter_subjects(self):
        yield _TestSubjectSpec()


test_segment_spec = SegmentSpec(
    segment=TestSegmentId.SEGMENT,
    marker_type=TestMarkerId,
    patch_type=TestPatchId,
    axis_convention=Z_UP_AXES,
    marker_set=MarkerSetSpec(marker_type=TestMarkerId),
    marker_positions_segment={
        TestMarkerId.HEEL: np.array([0.0, 0.0, 0.0]),
    },
)

state = SceneState(
    segment_poses={
        SegmentKey(TestSubjectId.SUBJECT, TestSegmentId.SEGMENT): SegmentPoseTrajectory(
            poses=(RigidTransform.identity(),),
        ),
    }
)
scene_spec = _TestSceneSpec()
expected_position = np.array([1.0, 2.0, 3.0])
marker_frame = object()


def _fake_marker_positions_by_name(
    _marker_frame,
    *,
    subject_name: str,
    segment_name: str,
):
    assert subject_name == TestSubjectId.SUBJECT.label
    assert segment_name == TestSegmentId.SEGMENT.label
    return {TestMarkerId.HEEL.label: expected_position}


def test_marker_position_accepts_segment_spec_and_subject(monkeypatch) -> None:
    monkeypatch.setattr(
        "retarget.io.unbagged.marker_positions_by_name",
        _fake_marker_positions_by_name,
    )
    observed = marker_position(
        marker_frame,  # type: ignore[arg-type]
        subject=TestSubjectId.SUBJECT,
        segment=test_segment_spec,
        marker=TestMarkerId.HEEL,
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected_position)


def test_marker_position_accepts_segment_view(monkeypatch) -> None:
    monkeypatch.setattr(
        "retarget.io.unbagged.marker_positions_by_name",
        _fake_marker_positions_by_name,
    )
    scene = SceneView(spec=scene_spec, state=state)
    segment_view = scene.subject(TestSubjectId.SUBJECT).segment(TestSegmentId.SEGMENT)
    observed = marker_position(
        marker_frame,  # type: ignore[arg-type]
        segment=segment_view,
        marker=TestMarkerId.HEEL,
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected_position)


def test_marker_position_requires_subject_for_segment_spec(monkeypatch) -> None:
    monkeypatch.setattr(
        "retarget.io.unbagged.marker_positions_by_name",
        _fake_marker_positions_by_name,
    )
    with pytest.raises(TypeError, match="subject must be provided"):
        marker_position(
            marker_frame,  # type: ignore[arg-type]
            segment=test_segment_spec,
            marker=TestMarkerId.HEEL,
        )
