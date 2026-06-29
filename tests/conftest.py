"""Shared pytest fixtures and helpers for the typed demonstration layer."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fungeom import Face, Point3, Region2

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
    Subject,
    Subjects,
)
from retarget.core.geometry import SegmentGeometry
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


@dataclass(frozen=True, slots=True)
class DemoMarkers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker


@dataclass(frozen=True, slots=True)
class DemoPatches(Patches):
    sole: Patch
    toe: Patch


@dataclass(frozen=True, slots=True)
class DemoSegments(Segments):
    segment: Segment[DemoMarkers, DemoPatches]


@dataclass(frozen=True, slots=True)
class DemoSubjects(Subjects):
    subject: Subject[DemoSegments]


def _demo_sole_geometry(seg: SegmentGeometry) -> Face:
    """Open-algebra equivalent of the old ``planar(plane_from(...), fixed(1, 1))`` sole.

    Fits the contact plane through the three markers, pins the normal to +z (matching the
    old ``axis_normal`` default), and gives a 1x1 rectangular footprint.
    """
    markers = seg.markers["heel", "toe", "mid"]
    plane = markers.fit_plane().facing(Point3.at(0.0, 0.0, 1.0))
    return Face.on(plane, Region2.rectangle(1.0, 1.0))


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
                        sole=Patch(label="sole", geometry=_demo_sole_geometry),
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
    poses = tuple(RigidTransform.from_translation(np.array([float(i), 0.0, 0.0])) for i in range(num_timesteps))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
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


def make_descending_mocap_track(
    *,
    t_end: float = 3.0,
    hz: float = 100.0,
    descend_until: float = 0.8,
    start_height: float = 0.5,
) -> MocapTrack[DemoSubjects]:
    """A mocap track whose segment (and its sole patch) descends to z=0 then rests.

    Useful for contact-detection tests: the recording is mostly quiet (resting on
    the floor) with a clear descent phase at the start.
    """
    n = int(round(t_end * hz)) + 1
    timestamps = np.linspace(0.0, t_end, n)
    z = np.where(
        timestamps < descend_until,
        start_height * (1.0 - timestamps / descend_until),
        0.0,
    )
    poses = tuple(RigidTransform.from_translation(np.array([0.0, 0.0, float(z[i])])) for i in range(n))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(
        subjects=make_demo_subjects(),
        state=state,
        timestamps=timestamps,
        marker_frames=None,
    )


def make_gliding_mocap_track(
    *,
    glide_speed: float = 0.3,
    t_end: float = 2.0,
    hz: float = 100.0,
    liftoff_at: float | None = None,
) -> MocapTrack[DemoSubjects]:
    """A segment (and its sole patch) translating horizontally at ``glide_speed`` on z=0.

    Models a body riding a moving support (a shoe on a gliding skateboard): the
    world-frame motion is large, yet relative to a support that glides with it the
    patch is at rest. If ``liftoff_at`` is given the segment additionally rises off
    z=0 after that time (a genuine separation from the glider).
    """
    n = int(round(t_end * hz)) + 1
    timestamps = np.linspace(0.0, t_end, n)
    x = glide_speed * timestamps
    z = np.zeros(n)
    if liftoff_at is not None:
        z = 0.3 * np.clip((timestamps - liftoff_at) / 0.3, 0.0, 1.0)
    poses = tuple(RigidTransform.from_translation(np.array([float(x[i]), 0.0, float(z[i])])) for i in range(n))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
    return MocapTrack(
        subjects=make_demo_subjects(),
        state=state,
        timestamps=timestamps,
        marker_frames=None,
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
        confidences=({target: np.asarray(confidences, dtype=np.float64)} if confidences is not None else {}),
    )
    return track, target
