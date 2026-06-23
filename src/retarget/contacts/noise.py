"""Measured sensor-noise scales for contact detection.

Contact detection scores each signal by how many noise standard deviations it sits
from "resting contact". Those per-channel noise scales (σ) should reflect the actual
measurement error, not be hand-tuned. :func:`estimate_noise` derives them from a
*static calibration recording* (the rig held still): every motion channel should read
~0, so its observed magnitude is the noise floor, and the contact point's positional
jitter is the clearance noise. When no calibration is supplied, :data:`DEFAULT_NOISE`
(typical optical-mocap σ) is used.

Noise is measured directly on ``patch.points()`` (the segment-solver pose applied to
the calibrated contact origin) -- the quantity detection actually scores -- rather than
propagated from raw marker covariance (the segment pose is solver-provided).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from retarget.contacts._features import angular_speed_from_frames
from retarget.contacts._quiet import QuietParams, compute_quiet_activity_and_spread, local_polynomial_derivative
from retarget.contacts._scope import Scope, normalize_scope
from retarget.core.targets import PatchTarget
from retarget.core.types import FloatArray


@dataclass(frozen=True, slots=True)
class ChannelNoise:
    """Per-channel noise scale (σ) for one patch, in physical units.

    Each value is the standard deviation / typical resting magnitude of a feature
    channel under genuine "at rest in contact" conditions. Detection divides each
    feature by its σ, so a feature one σ from rest contributes ~1 to the χ² penalty.
    """

    clearance_sigma: float = 0.0015
    """Positional jitter of the contact point along the up/normal axis (m). This is the
    tight *discriminator* channel -- which surface a patch rests on is decided here."""
    normal_speed_sigma: float = 0.08
    """Speed-along-normal noise scale (m/s). Generous: differentiation amplifies position
    noise, and resting contact on a *moving* support shows small residual relative speed.
    A genuine fly-through is m/s, far above this, so generosity costs no discrimination."""
    tangential_speed_sigma: float = 0.08
    """In-plane speed noise scale (m/s); generous, as above."""
    quiet_activity_sigma: float = 0.08
    """Windowed-RMS speed (quiet activity) noise scale (m/s); generous, as above."""
    quiet_spread_sigma: float = 0.01
    """Windowed position-range (quiet spread) noise scale (m); generous, as above."""
    angular_speed_sigma: float = 0.3
    """Angular-speed noise scale (rad/s); generous (a rolling/settling support tilts)."""


# Typical optical mocap (~100-250 Hz). Deliberately conservative (slightly large) so
# uncalibrated detection is not hair-trigger; calibration tightens these.
DEFAULT_NOISE = ChannelNoise()


@dataclass(frozen=True, slots=True)
class NoiseCalibration:
    """Per-patch measured noise scales, with a fallback for uncalibrated patches."""

    per_patch: Mapping[PatchTarget, ChannelNoise] = field(default_factory=dict)
    default: ChannelNoise = DEFAULT_NOISE

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_patch", MappingProxyType(dict(self.per_patch)))

    def for_patch(self, target: PatchTarget) -> ChannelNoise:
        """Measured noise for ``target``; the ``default`` when absent."""
        return self.per_patch.get(target, self.default)


def _robust_std(values: FloatArray, floor: float) -> float:
    """Robust standard deviation via the median absolute deviation (×1.4826)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return floor
    mad = float(np.median(np.abs(finite - np.median(finite))))
    return max(1.4826 * mad, floor)


def _resting_scale(magnitude: FloatArray, floor: float) -> float:
    """Noise scale for a non-negative magnitude channel: its resting RMS level."""
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return floor
    return max(float(np.sqrt(np.mean(finite**2))), floor)


def _channel_noise(
    points: FloatArray,
    frames: FloatArray,
    timestamps: FloatArray,
    up_axis: int,
    quiet: QuietParams,
    floors: ChannelNoise,
) -> ChannelNoise:
    up = np.zeros(3, dtype=np.float64)
    up[up_axis] = 1.0
    height = points @ up
    velocity = local_polynomial_derivative(
        points, timestamps, quiet.derivative_window_time, quiet.derivative_poly_degree
    )
    normal_speed = np.abs(velocity @ up)
    tangential = velocity - (velocity @ up)[:, None] * up
    tangential_speed = np.linalg.norm(tangential, axis=1)
    activity, spread = compute_quiet_activity_and_spread(points, timestamps, quiet)
    angular = angular_speed_from_frames(
        frames, timestamps, quiet.derivative_window_time, quiet.derivative_poly_degree
    )
    return ChannelNoise(
        clearance_sigma=_robust_std(height, floors.clearance_sigma),
        normal_speed_sigma=_resting_scale(normal_speed, floors.normal_speed_sigma),
        tangential_speed_sigma=_resting_scale(tangential_speed, floors.tangential_speed_sigma),
        quiet_activity_sigma=_resting_scale(activity, floors.quiet_activity_sigma),
        quiet_spread_sigma=_resting_scale(spread, floors.quiet_spread_sigma),
        angular_speed_sigma=_resting_scale(angular, floors.angular_speed_sigma),
    )


def estimate_noise(
    scope: Scope,
    *,
    up_axis: int = 2,
    quiet: QuietParams = QuietParams(),
    floor_fraction: float = 0.2,
) -> NoiseCalibration:
    """Measure per-patch noise scales from a *static* calibration recording.

    ``scope`` is bound mocap held still (a ``Patch``/``Segment``/``Subject``/
    ``MocapTrack`` or patches). For each geometry-bearing patch, the contact point's
    positional jitter gives the clearance σ and every motion channel's resting
    magnitude gives its σ. Each σ is floored at ``floor_fraction`` of the corresponding
    :data:`DEFAULT_NOISE` value, so calibration can *refine* the physical defaults (down
    to a sane fraction) but a degenerately clean clip (an over-smoothed rigid body) can't
    yield an absurdly tiny, hair-trigger scale.
    """
    floors = ChannelNoise(
        clearance_sigma=floor_fraction * DEFAULT_NOISE.clearance_sigma,
        normal_speed_sigma=floor_fraction * DEFAULT_NOISE.normal_speed_sigma,
        tangential_speed_sigma=floor_fraction * DEFAULT_NOISE.tangential_speed_sigma,
        quiet_activity_sigma=floor_fraction * DEFAULT_NOISE.quiet_activity_sigma,
        quiet_spread_sigma=floor_fraction * DEFAULT_NOISE.quiet_spread_sigma,
        angular_speed_sigma=floor_fraction * DEFAULT_NOISE.angular_speed_sigma,
    )
    per_patch: dict[PatchTarget, ChannelNoise] = {}
    for patch in normalize_scope(scope):
        runtime = patch._runtime()
        timestamps = np.asarray(runtime.timestamps, dtype=np.float64)
        points = np.asarray(patch.points(), dtype=np.float64)
        frames = np.asarray(patch.frames(), dtype=np.float64)
        per_patch[patch.target] = _channel_noise(points, frames, timestamps, up_axis, quiet, floors)
    default = _median_channel_noise(list(per_patch.values()))
    return NoiseCalibration(per_patch=per_patch, default=default)


def _median_channel_noise(noises: list[ChannelNoise]) -> ChannelNoise:
    if not noises:
        return DEFAULT_NOISE
    return ChannelNoise(
        clearance_sigma=float(np.median([n.clearance_sigma for n in noises])),
        normal_speed_sigma=float(np.median([n.normal_speed_sigma for n in noises])),
        tangential_speed_sigma=float(np.median([n.tangential_speed_sigma for n in noises])),
        quiet_activity_sigma=float(np.median([n.quiet_activity_sigma for n in noises])),
        quiet_spread_sigma=float(np.median([n.quiet_spread_sigma for n in noises])),
        angular_speed_sigma=float(np.median([n.angular_speed_sigma for n in noises])),
    )
