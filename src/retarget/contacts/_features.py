"""Support-relative features for one patch trajectory against one support.

For a single patch's world-frame point trajectory ``(T, 3)`` and the support's
per-frame clearance/normals, this computes the per-frame feature channels the
scorer fuses: (offset-corrected) clearance, normal speed, tangential speed,
quiet activity/spread, and angular speed. Ported from the reference pipeline,
reduced to the single-patch case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.spatial.transform import Rotation

from retarget.contacts._quiet import QuietResult
from retarget.core.types import FloatArray, Points3, TimeMat3, TimeVec3


@dataclass(frozen=True, slots=True)
class Features:
    """Per-frame ``(T,)`` feature channels for one patch against one support."""

    abs_clearance: FloatArray
    normal_speed: FloatArray
    tangential_speed: FloatArray
    quiet_activity: FloatArray
    quiet_spread: FloatArray
    angular_speed: FloatArray
    quiet_activity_scale: float
    quiet_spread_scale: float
    contact_offset: float


def estimate_contact_offset(
    clearance: FloatArray,
    quiet_mask: np.ndarray,
    quantile: float = 0.20,
    max_abs_offset: float = 0.045,
) -> float:
    """Estimate a clearance bias from quiet, near-contact samples.

    The support plane is fit through sample centroids, so a patch resting on it
    typically has a small constant clearance offset. We estimate that offset as
    the median of the lowest-clearance quiet samples and subtract it, so a
    resting patch reads as zero clearance. Outliers beyond ``max_abs_offset``
    yield ``0.0`` (no correction).
    """
    values = np.asarray(clearance, dtype=np.float64)
    mask = np.asarray(quiet_mask, dtype=np.bool_)
    candidates = values[mask & np.isfinite(values)]
    if candidates.size == 0:
        return 0.0
    cutoff = float(np.quantile(candidates, min(max(quantile, 0.0), 1.0)))
    low = candidates[candidates <= cutoff]
    if low.size == 0:
        low = candidates
    offset = float(np.median(low))
    if abs(offset) > max_abs_offset:
        return 0.0
    return offset


def angular_speed_from_frames(frames: TimeMat3, timestamps: FloatArray) -> FloatArray:
    """Estimate angular speed (rad/s) per frame from a ``(T, 3, 3)`` orientation series."""
    rotations = np.asarray(frames, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    n = len(t)
    speeds = np.zeros(n, dtype=np.float64)
    if n < 2 or rotations.shape[0] != n:
        return cast(FloatArray, speeds)
    rot = Rotation.from_matrix(rotations)
    relative = rot[:-1].inv() * rot[1:]
    angles = relative.magnitude()
    dt = np.diff(t)
    pair_speeds = angles / dt
    speeds[0] = pair_speeds[0]
    speeds[-1] = pair_speeds[-1]
    if n > 2:
        speeds[1:-1] = 0.5 * (pair_speeds[:-1] + pair_speeds[1:])
    return cast(FloatArray, speeds)


def compute_features(
    *,
    clearance: FloatArray,
    normals: Points3,
    velocities: TimeVec3,
    quiet: QuietResult,
    angular_speed: FloatArray,
    contact_offset_max: float,
) -> Features:
    """Build the per-frame feature channels for one patch against one support."""
    raw_clearance = np.asarray(clearance, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)

    offset = estimate_contact_offset(raw_clearance, quiet.mask, max_abs_offset=contact_offset_max)
    adjusted = raw_clearance - offset

    normal_velocity = np.sum(velocity_array * normal_array, axis=1)
    tangential = velocity_array - normal_velocity[:, None] * normal_array
    tangential_speed = np.linalg.norm(tangential, axis=1)

    return Features(
        abs_clearance=cast(FloatArray, np.abs(adjusted)),
        normal_speed=cast(FloatArray, np.abs(normal_velocity)),
        tangential_speed=cast(FloatArray, tangential_speed),
        quiet_activity=quiet.activity,
        quiet_spread=quiet.spread,
        angular_speed=np.asarray(angular_speed, dtype=np.float64),
        quiet_activity_scale=quiet.activity_scale,
        quiet_spread_scale=quiet.spread_scale,
        contact_offset=offset,
    )
