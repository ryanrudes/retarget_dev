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

from retarget.contacts._quiet import QuietResult, local_polynomial_derivative
from retarget.core.types import FloatArray, Points3, TimeMat3, TimeVec3


@dataclass(frozen=True, slots=True)
class Features:
    """Per-frame ``(T,)`` feature channels for one patch against one support.

    ``clearance`` is the *signed* residual ``(measured clearance - resting bias)``:
    positive is a gap (patch above the resting plane), negative is penetration. The
    scorer treats the two sides asymmetrically, so the sign is kept (not ``abs``).
    """

    clearance: FloatArray
    normal_speed: FloatArray
    tangential_speed: FloatArray
    quiet_activity: FloatArray
    quiet_spread: FloatArray
    angular_speed: FloatArray
    resting_bias: float


def estimate_resting_bias(
    clearance: FloatArray,
    quiet_mask: np.ndarray,
    max_resting_bias: float,
) -> float:
    """Estimate the constant resting-clearance bias from quiet, *near-contact* samples.

    A patch resting on a support typically shows a small constant clearance (the fitted
    plane passes through marker centroids, the modeled contact origin may sit a few mm
    off). We estimate it as the median of the quiet frames whose clearance is already
    within ``±max_resting_bias`` of zero -- the candidate resting frames -- and subtract
    it so a true rest reads ~0.

    Restricting to the near-contact band (rather than pooling *all* quiet samples) is
    essential when a patch's relationship to this support *changes* over the clip -- e.g.
    a shoe moved from the floor onto a board: the on-floor frames sit far below the board
    surface, and pooling them would drag the estimated bias away and shift the genuine
    on-board frames into a false gap. The band also guarantees a real (cm-scale) surface
    gap can never be subtracted away.
    """
    values = np.asarray(clearance, dtype=np.float64)
    mask = np.asarray(quiet_mask, dtype=np.bool_)
    band = abs(max_resting_bias)
    candidates = values[mask & np.isfinite(values) & (np.abs(values) <= band)]
    if candidates.size == 0:
        return 0.0
    return float(np.clip(np.median(candidates), -band, band))


def angular_speed_from_frames(
    frames: TimeMat3,
    timestamps: FloatArray,
    window_time: float | None = None,
    degree: int = 1,
) -> FloatArray:
    """Estimate angular speed (rad/s) per frame from a ``(T, 3, 3)`` orientation series.

    With ``window_time=None`` this is a raw central finite difference of the inter-frame
    rotation angle. With a positive ``window_time`` the estimate is smoothed exactly like
    the translation channel (:func:`local_polynomial_derivative`): the inter-frame body
    rotation vectors are accumulated into a rotation-vector path and that *vector* path is
    polynomial-differentiated, then its magnitude taken. Smoothing the vector path (rather
    than the rectified per-frame angle) removes zero-mean orientation noise without the
    upward bias that differencing noisy per-frame angles incurs.
    """
    rotations = np.asarray(frames, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    n = len(t)
    speeds = np.zeros(n, dtype=np.float64)
    if n < 2 or rotations.shape[0] != n:
        return cast(FloatArray, speeds)
    rot = Rotation.from_matrix(rotations)
    increments = (rot[:-1].inv() * rot[1:]).as_rotvec()  # (n-1, 3): body-frame steps

    if window_time is not None and window_time > 0.0:
        # Integrate the steps into a rotation-vector path, then smooth-differentiate it as
        # a vector and take the magnitude -- the rotational twin of the translation channel.
        path = np.zeros((n, 3), dtype=np.float64)
        path[1:] = np.cumsum(increments, axis=0)
        omega = np.asarray(local_polynomial_derivative(path, t, window_time, degree), dtype=np.float64)
        return cast(FloatArray, np.linalg.norm(omega, axis=1))

    angles = np.linalg.norm(increments, axis=1)
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
    max_resting_bias: float,
) -> Features:
    """Build the per-frame feature channels for one patch against one support.

    The resting bias is estimated and subtracted from clearance (σ-bounded), leaving a
    *signed* clearance residual. Motion channels are magnitudes (speed perpendicular to
    and along the support, quiet activity/spread, angular speed).
    """
    raw_clearance = np.asarray(clearance, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    velocity_array = np.asarray(velocities, dtype=np.float64)

    bias = estimate_resting_bias(raw_clearance, quiet.mask, max_resting_bias)
    residual = raw_clearance - bias

    normal_velocity = np.sum(velocity_array * normal_array, axis=1)
    tangential = velocity_array - normal_velocity[:, None] * normal_array
    tangential_speed = np.linalg.norm(tangential, axis=1)

    return Features(
        clearance=residual,
        normal_speed=cast(FloatArray, np.abs(normal_velocity)),
        tangential_speed=cast(FloatArray, tangential_speed),
        quiet_activity=quiet.activity,
        quiet_spread=quiet.spread,
        angular_speed=np.asarray(angular_speed, dtype=np.float64),
        resting_bias=bias,
    )
