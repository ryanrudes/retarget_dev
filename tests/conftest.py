"""Shared pytest fixtures and helpers for demonstration-layer tests."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from retarget.core import (
    MarkerId,
    MarkerSetSpec,
    PatchCalibrationSpec,
    PatchId,
    RectangularRegion,
    RigidTransform,
    SceneSpec,
    SceneState,
    SegmentId,
    SegmentKey,
    SegmentPoseTrajectory,
    SegmentSpec,
    SubjectId,
    SubjectSpec,
    Z_UP_AXES,
)
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "process_mocap_data"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))


class DemoSubjectId(SubjectId):
    SUBJECT = "subject"


class DemoSegmentId(SegmentId):
    SEGMENT = "segment"


class DemoMarkerId(MarkerId):
    HEEL = "heel"
    TOE = "toe"
    MID = "mid"


class DemoPatchId(PatchId):
    SOLE = "sole"


DEMO_SEGMENT_SPEC = (
    SegmentSpec(
        segment=DemoSegmentId.SEGMENT,
        marker_type=DemoMarkerId,
        patch_type=DemoPatchId,
        axis_convention=Z_UP_AXES,
        marker_set=MarkerSetSpec(marker_type=DemoMarkerId),
        marker_positions_segment={
            DemoMarkerId.HEEL: np.array([0.0, 0.0, 0.0]),
            DemoMarkerId.TOE: np.array([1.0, 0.0, 0.0]),
            DemoMarkerId.MID: np.array([0.0, 1.0, 0.0]),
        },
        patch_calibrations={
            DemoPatchId.SOLE: PatchCalibrationSpec(
                patch=DemoPatchId.SOLE,
                markers=(DemoMarkerId.HEEL, DemoMarkerId.TOE, DemoMarkerId.MID),
                region=RectangularRegion(width=1.0, height=1.0),
            ),
        },
    ).with_built_patches()
)


@dataclass(frozen=True, slots=True)
class DemoSubjectSpec(SubjectSpec):
    segment_spec: SegmentSpec[DemoMarkerId, DemoPatchId]

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        yield self.segment_spec


@dataclass(frozen=True, slots=True)
class DemoSceneSpec(SceneSpec):
    subject_spec: DemoSubjectSpec

    def iter_subjects(self) -> Iterable[SubjectSpec]:
        yield self.subject_spec


def make_mocap_track(
    *,
    num_timesteps: int = 3,
    include_markers: bool = True,
) -> MocapTrack:
    scene_spec = DemoSceneSpec(
        subject_spec=DemoSubjectSpec(
            subject=DemoSubjectId.SUBJECT,
            segment_spec=DEMO_SEGMENT_SPEC,
        )
    )
    poses = tuple(
        RigidTransform.from_translation(np.array([float(i), 0.0, 0.0]))
        for i in range(num_timesteps)
    )
    state = SceneState(
        segment_poses={
            SegmentKey(DemoSubjectId.SUBJECT, DemoSegmentId.SEGMENT): SegmentPoseTrajectory(
                poses=poses,
            )
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
                        marker_name=DemoMarkerId.HEEL.label,
                        subject_name=DemoSubjectId.SUBJECT.label,
                        segment_name=DemoSegmentId.SEGMENT.label,
                        position_world=np.array([float(i), 0.0, 0.0]),
                        occluded=False,
                    ),
                ),
            )
            for i in range(num_timesteps)
        )
    return MocapTrack(
        scene_spec=scene_spec,
        state=state,
        timestamps=timestamps,
        marker_frames=marker_frames,
    )
