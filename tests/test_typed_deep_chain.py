"""Static + runtime checks that the typed deep chain resolves to concrete types.

The ``assert_type`` calls are no-ops at runtime but are verified by the type
checker (``uv run mypy``), guarding the headline feature: literal-key access
through the schema projects to the authored types with no enums and no codegen.
"""

from __future__ import annotations

from typing import assert_type

import numpy as np

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
from retarget.core.types import TimeVec3
from retarget.demo import MocapTrack, Tracks, build_demonstration


class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker


class ShoePatches(Patches):
    sole: Patch


class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


class MySubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


class GroundEstimationTracks(Tracks):
    mocap: MocapTrack[MySubjects]


def _track() -> MocapTrack[MySubjects]:
    subjects = MySubjects(
        left_shoe=Subject(
            segments=ShoeSegments(
                shoe=Segment(
                    markers=ShoeMarkers(
                        heel=Marker(vicon_name="h", position_segment=np.zeros(3)),
                        toe=Marker(vicon_name="t", position_segment=np.array([1.0, 0.0, 0.0])),
                    ),
                    patches=ShoePatches(sole=Patch(label="sole")),
                )
            )
        )
    )
    state = SceneState(
        segment_poses={
            SegmentKey("left_shoe", "shoe"): SegmentPoseTrajectory(
                poses=(RigidTransform.identity(),)
            )
        }
    )
    return MocapTrack(subjects=subjects, state=state, timestamps=np.array([0.0]))


def test_deep_chain_static_and_runtime_types() -> None:
    demo = build_demonstration(GroundEstimationTracks(mocap=_track()))

    mocap = demo.tracks["mocap"]
    assert_type(mocap, MocapTrack[MySubjects])

    subject = mocap.subjects["left_shoe"]
    assert_type(subject, Subject[ShoeSegments])

    segment = subject.segments["shoe"]
    assert_type(segment, Segment[ShoeMarkers, ShoePatches])

    heel = segment.markers["heel"]
    assert_type(heel, Marker)
    assert_type(heel.positions(modeled=True), TimeVec3)

    assert isinstance(heel, Marker)
    assert isinstance(segment.patches["sole"], Patch)
    np.testing.assert_allclose(heel.positions(modeled=True)[0], np.zeros(3))
