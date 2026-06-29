"""Static + runtime checks that the typed deep chain resolves to concrete types.

The ``assert_type`` calls are no-ops at runtime but are verified by the type
checker (``uv run mypy``), guarding the headline feature: attribute access
through the schema projects to the authored types with no enums and no codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type, cast

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
from retarget.core.types import TimeVec3, Vec3
from retarget.demo import Demonstration, MocapTrack, Tracks
from retarget.demo.contact import ContactTrack


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
class MySubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


@dataclass(frozen=True, slots=True)
class GroundEstimationTracks(Tracks):
    mocap: MocapTrack[MySubjects]


def _track() -> MocapTrack[MySubjects]:
    subjects = MySubjects(
        left_shoe=Subject(
            segments=ShoeSegments(
                shoe=Segment(
                    markers=ShoeMarkers(
                        heel=Marker(mocap_name="h", position_segment=cast(Vec3, np.zeros(3))),
                        toe=Marker(mocap_name="t", position_segment=cast(Vec3, np.array([1.0, 0.0, 0.0]))),
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
    demo = Demonstration(GroundEstimationTracks(mocap=_track()))

    mocap = demo.tracks.mocap
    assert_type(mocap, MocapTrack[MySubjects])

    subject = mocap.subjects.left_shoe
    assert_type(subject, Subject[ShoeSegments])

    segment = subject.segments.shoe
    assert_type(segment, Segment[ShoeMarkers, ShoePatches])

    heel = segment.markers.heel
    assert_type(heel, Marker)
    assert_type(heel.positions(modeled=True), TimeVec3)

    assert isinstance(heel, Marker)
    assert isinstance(segment.patches.sole, Patch)
    np.testing.assert_allclose(heel.positions(modeled=True)[0], np.zeros(3))


def test_with_contacts_preserves_typed_deep_chain() -> None:
    track = _track()
    attached = track.with_contacts(ContactTrack(timestamps=track.timestamps, contacts={}))

    # Attaching contacts returns the same parametrized track type, so the typed
    # deep chain still resolves to the authored types.
    assert_type(attached, MocapTrack[MySubjects])
    assert isinstance(attached, MocapTrack)

    segment = attached.subjects.left_shoe.segments.shoe
    assert_type(segment, Segment[ShoeMarkers, ShoePatches])
    assert_type(segment.patches.sole, Patch)
    assert isinstance(segment.patches.sole, Patch)
