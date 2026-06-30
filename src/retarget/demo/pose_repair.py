"""Opt-in repair of short segment-pose gaps from dropped tracking.

Real captures lose the body solve when too many markers occlude; the TF stream
then carries held or drifting poses that the rest of the pipeline would treat as
real motion. :func:`fill_pose_gaps` finds frames where the pose is untrustworthy
(too few markers visible) and interpolates **short** gaps from the bracketing
trusted frames -- linearly for translation, SLERP for rotation -- while leaving
long gaps untouched (so they stay flagged as invalid via ``segment.valid()`` and
can be rendered as ``"unknown"``).

This is deliberately explicit and never silent: you opt in, choose ``max_gap_time``,
and long dropouts are not fabricated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from retarget.core.formats import JumpDetector
from retarget.core.keys import SegmentKey
from retarget.core.schema import Segment, Subject, Subjects
from retarget.core.schema.base import _schema_items
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.transform import RigidTransform
from retarget.demo.mocap import MocapTrack


def fill_pose_gaps[S: Subjects](
    mocap: MocapTrack[S],
    *,
    max_gap_time: float = 0.10,
    min_coverage: float = 0.5,
    jumps: JumpDetector | None = None,
) -> MocapTrack[S]:
    """Return a new track with short untrustworthy pose gaps interpolated.

    A frame is untrustworthy (a gap to fill) when marker coverage is below
    ``min_coverage``, or -- with a ``jumps`` detector -- when its pose is a
    garbage sample. Such samples make bad interpolation anchors, so they are
    skipped and interpolated *across*, never anchored *on*. Long gaps (beyond
    ``max_gap_time`` between trusted neighbours) are left as-is.

    Args:
        mocap: Source track (must carry marker observations to assess coverage).
        max_gap_time: Longest gap (seconds, measured between the bracketing
            trusted frames) to interpolate; longer gaps are left as-is.
        min_coverage: A frame's pose is trusted when at least this fraction of the
            segment's markers are observed.
        jumps: Optional garbage-sample detector (``speed_limit`` / ``robust_jumps``
            / your own); flagged frames are interpolated across too.

    Returns:
        A new ``MocapTrack`` with repaired segment poses. Attached contacts,
        support states, and marker frames are carried over unchanged.
    """
    timestamps = np.asarray(mocap.timestamps, dtype=np.float64)
    repaired: dict[SegmentKey, SegmentPoseTrajectory] = dict(mocap.state.segment_poses)
    pose_filled: dict[SegmentKey, np.ndarray] = {}
    subject: Subject[Any]
    segment: Segment[Any, Any]
    for subject_name, subject in _schema_items(mocap.subjects):
        for segment_name, segment in _schema_items(subject.segments):
            key = SegmentKey(subject_name, segment_name)
            trajectory = mocap.state.segment_poses.get(key)
            if trajectory is None:
                continue
            repaired[key], filled = _repair_trajectory(
                trajectory,
                timestamps,
                segment.valid(min_coverage=min_coverage, jumps=jumps),
                max_gap_time,
            )
            if bool(filled.any()):
                pose_filled[key] = filled
    return MocapTrack(
        subjects=mocap.subjects,
        state=SceneState(segment_poses=repaired),
        timestamps=mocap.timestamps,
        marker_frames=mocap.marker_frames,
        contacts=mocap.contacts,
        nominal_hz_override=mocap.nominal_hz_override,
        support_states=mocap.support_states,
        pose_filled=pose_filled,
    )


def _repair_trajectory(
    trajectory: SegmentPoseTrajectory,
    timestamps: np.ndarray,
    valid: np.ndarray,
    max_gap_time: float,
) -> tuple[SegmentPoseTrajectory, np.ndarray]:
    """Return ``(repaired_trajectory, filled_mask)``; ``filled_mask`` marks synthesized frames."""
    valid_mask = np.asarray(valid, dtype=np.bool_)
    n = len(trajectory.poses)
    filled = np.zeros(n, dtype=np.bool_)
    if n == 0 or valid_mask.all():
        return trajectory, filled

    translations = np.stack([pose.translation for pose in trajectory.poses]).astype(np.float64)
    quaternions = Rotation.from_matrix(
        np.stack([pose.rotation for pose in trajectory.poses]).astype(np.float64)
    ).as_quat()

    invalid = ~valid_mask
    padded = np.r_[False, invalid, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]  # exclusive
    for start_idx, end_idx_excl in zip(starts, ends):
        start = int(start_idx)
        end = int(end_idx_excl) - 1  # inclusive
        if start == 0 or end == n - 1:
            continue  # gap touches a series edge -> no trusted bracket to interpolate from
        lo, hi = start - 1, end + 1
        if float(timestamps[hi] - timestamps[lo]) > max_gap_time:
            continue  # too long to fabricate; leave invalid
        span = timestamps[start : end + 1]
        anchor_t = [float(timestamps[lo]), float(timestamps[hi])]
        for axis in range(3):
            translations[start : end + 1, axis] = np.interp(
                span, anchor_t, [translations[lo, axis], translations[hi, axis]]
            )
        slerp = Slerp(anchor_t, Rotation.from_quat([quaternions[lo], quaternions[hi]]))
        quaternions[start : end + 1] = slerp(span).as_quat()
        filled[start : end + 1] = True

    rotations = Rotation.from_quat(quaternions).as_matrix()
    repaired = SegmentPoseTrajectory(
        tuple(
            RigidTransform.from_rotation_translation(rotations[i], translations[i])
            for i in range(n)
        )
    )
    return repaired, filled
