"""Shared pytest fixtures and helpers for the typed demonstration layer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    PatchTarget,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    Segments,
    SemanticAxis,
    Subject,
    Subjects,
)
from retarget.demo.contact import ContactTrack
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "process_mocap_data"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

# Stable authored names used throughout the demo-layer tests.
SUBJECT = "subject"
SEGMENT = "segment"

_MARKER_POSITIONS = {
    "heel": np.array([0.0, 0.0, 0.0]),
    "toe": np.array([1.0, 0.0, 0.0]),
    "mid": np.array([0.0, 1.0, 0.0]),
}


class DemoMarkers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker


class DemoPatches(Patches):
    sole: Patch
    toe: Patch


class DemoSegments(Segments):
    segment: Segment[DemoMarkers, DemoPatches]


class DemoSubjects(Subjects):
    subject: Subject[DemoSegments]


def make_demo_subjects() -> DemoSubjects:
    return DemoSubjects(
        subject=Subject(
            segments=DemoSegments(
                segment=Segment(
                    markers=DemoMarkers(
                        heel=Marker(mocap_name="heel", position_segment=_MARKER_POSITIONS["heel"]),
                        toe=Marker(mocap_name="toe", position_segment=_MARKER_POSITIONS["toe"]),
                        mid=Marker(mocap_name="mid", position_segment=_MARKER_POSITIONS["mid"]),
                    ),
                    patches=DemoPatches(
                        sole=Patch.rectangle(
                            label="sole",
                            markers=("heel", "toe", "mid"),
                            width=1.0,
                            height=1.0,
                            outward_axis=SemanticAxis.UP,
                            forward_axis=SemanticAxis.FORWARD,
                        ),
                        toe=Patch(label="toe"),
                    ),
                )
            )
        )
    )


def make_mocap_track(
    *,
    num_timesteps: int = 3,
    include_markers: bool = True,
) -> MocapTrack[DemoSubjects]:
    poses = tuple(
        RigidTransform.from_translation(np.array([float(i), 0.0, 0.0]))
        for i in range(num_timesteps)
    )
    state = SceneState(
        segment_poses={
            SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)
        }
    )
    timestamps = np.arange(num_timesteps, dtype=np.float64) * 0.1
    marker_frames = None
    if include_markers:
        marker_frames = tuple(
            ViconMarkersFrame(
                stamp_seconds=float(timestamps[i]),
                markers=(
                    MarkerObservation(
                        marker_name="heel",
                        subject_name=SUBJECT,
                        segment_name=SEGMENT,
                        position_world=np.array([float(i), 0.0, 0.0]),
                        occluded=False,
                    ),
                ),
            )
            for i in range(num_timesteps)
        )
    return MocapTrack(
        subjects=make_demo_subjects(),
        state=state,
        timestamps=timestamps,
        marker_frames=marker_frames,
    )


def demo_segment(track: MocapTrack[Any]) -> Segment[DemoMarkers, DemoPatches]:
    """Return the canonical demo segment from a (possibly sliced) mocap track."""
    return track.subjects[SUBJECT].segments[SEGMENT]


def make_string_patch_target(
    name: str = "patch",
    subject: str = SUBJECT,
    segment: str = SEGMENT,
) -> PatchTarget:
    return PatchTarget(subject=subject, segment=segment, patch=name)


def make_demo_patch_target(patch: str = "sole") -> PatchTarget:
    return PatchTarget(subject=SUBJECT, segment=SEGMENT, patch=patch)


def make_string_contact_track(
    *,
    timestamps: list[float] | np.ndarray = (0.0, 1.0, 2.0, 3.0),
    target_name: str = "patch",
    contacts: list[bool] | np.ndarray = (False, True, True, False),
    confidences: list[float] | np.ndarray | None = (0.1, 0.2, 0.3, 0.4),
) -> tuple[ContactTrack, PatchTarget]:
    target = make_string_patch_target(target_name)
    track = ContactTrack(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        contacts={target: np.asarray(contacts, dtype=np.bool_)},
        confidences=(
            {target: np.asarray(confidences, dtype=np.float64)}
            if confidences is not None
            else {}
        ),
    )
    return track, target
