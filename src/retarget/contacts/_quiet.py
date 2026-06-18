"""Quiet-interval detection on scalar or vector position signals.

A signal is "quiet" where both its local *activity* (smoothed speed, windowed
RMS) and its local *spread* (windowed position range) fall below adaptive
hysteresis thresholds. This is the substrate the contact detector uses to (a)
find quiet candidate samples for support-surface fitting and (b) feed
quiet activity/spread as features into contact scoring.

Ported and slimmed from the reference ``contact_detection`` pipeline: only the
scalar and vector-position (norm-reduced) paths are kept; quaternion and
Mahalanobis variants are intentionally omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

from retarget.core.types import FloatArray, TimeBool
from retarget.utils.intervals import (
    clean_mask_by_time,
    intervals_from_mask,
    mask_from_intervals,
)


@dataclass(frozen=True, slots=True)
class QuietParams:
    """Smoothing windows, derivative settings, and threshold floors for quiet detection."""

    pos_smooth_time: float = 0.03
    vel_smooth_time: float = 0.02
    quiet_window_time: float = 0.10
    spread_window_time: float = 0.12
    derivative_window_time: float | None = 0.05
    derivative_poly_degree: int = 1
    min_interval_time: float = 0.18
    max_gap_time: float = 0.10
    min_blip_time: float = 0.15
    min_activity_on_eps: float = 0.003
    min_activity_off_eps: float = 0.010
    on_percentile: float = 45.0
    off_percentile: float = 65.0
    use_time_gaussian_smoothing: bool = True
    use_time_windows: bool = True


@dataclass(frozen=True, slots=True)
class QuietResult:
    """Per-frame quiet detection output.

    Attributes:
        mask: Boolean quiet mask with shape ``(T,)`` (gap/blip cleaned, min-duration enforced).
        activity: Smoothed activity (windowed RMS speed) with shape ``(T,)``.
        spread: Local position spread (windowed range) with shape ``(T,)``.
        activity_scale: Adaptive activity off-threshold used to normalize the activity feature.
        spread_scale: Adaptive spread off-threshold used to normalize the spread feature.
    """

    mask: TimeBool
    activity: FloatArray
    spread: FloatArray
    activity_scale: float = 1.0
    spread_scale: float = 1.0


def _ensure_2d(x: FloatArray) -> tuple[FloatArray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 1
    if scalar:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("signal must have shape (T,) or (T, D)")
    return arr, scalar


def _odd_window_samples(timestamps: FloatArray, window_time: float) -> int:
    dt_med = float(np.median(np.diff(timestamps)))
    samples = max(3, int(round(window_time / dt_med)))
    if samples % 2 == 0:
        samples += 1
    return samples


def time_gaussian_smooth(
    x: FloatArray,
    timestamps: FloatArray,
    sigma_time: float,
    radius_sigma: float = 3.0,
) -> FloatArray:
    """Smooth a scalar/vector signal with a Gaussian kernel in real time units."""
    arr, scalar = _ensure_2d(x)
    t = np.asarray(timestamps, dtype=np.float64)
    n = len(t)
    if n == 0 or sigma_time <= 0.0:
        return arr[:, 0].copy() if scalar else arr.copy()

    out = np.empty_like(arr)
    radius = radius_sigma * sigma_time
    left = 0
    right = -1
    for i in range(n):
        center = t[i]
        while left < n and t[left] < center - radius:
            left += 1
        while right + 1 < n and t[right + 1] <= center + radius:
            right += 1
        idx = np.arange(left, right + 1)
        if idx.size == 0:
            out[i] = arr[i]
            continue
        dt = t[idx] - center
        weights = np.exp(-0.5 * (dt / sigma_time) ** 2)
        weight_sum = float(np.sum(weights))
        out[i] = arr[i] if weight_sum <= 0.0 else weights @ arr[idx] / weight_sum
    return out[:, 0] if scalar else out


def _smooth(x: FloatArray, timestamps: FloatArray, smooth_time: float, use_time_gaussian: bool) -> FloatArray:
    if use_time_gaussian:
        return time_gaussian_smooth(x, timestamps, sigma_time=smooth_time)
    from scipy.ndimage import gaussian_filter1d

    arr, scalar = _ensure_2d(x)
    dt_med = float(np.median(np.diff(timestamps)))
    out = gaussian_filter1d(arr, sigma=smooth_time / dt_med, axis=0)
    return cast(FloatArray, out[:, 0] if scalar else out)


def local_polynomial_derivative(
    x: FloatArray,
    timestamps: FloatArray,
    window_time: float | None = 0.05,
    degree: int = 1,
) -> FloatArray:
    """Estimate ``dx/dt`` with a local (optionally weighted) polynomial fit per sample."""
    arr, scalar = _ensure_2d(x)
    t = np.asarray(timestamps, dtype=np.float64)
    if len(t) != len(arr):
        raise ValueError("signal and timestamps must have the same length")
    if len(t) < 2:
        out = np.zeros_like(arr)
        return out[:, 0] if scalar else out
    if degree < 1:
        raise ValueError("degree must be at least 1")

    fallback = np.gradient(arr, t, axis=0)
    if window_time is None or window_time <= 0.0:
        return cast(FloatArray, fallback[:, 0] if scalar else fallback)

    out = np.empty_like(arr)
    min_points = degree + 1
    half_window = 0.5 * window_time
    n = len(t)
    for i, center in enumerate(t):
        idx = np.where((t >= center - half_window) & (t <= center + half_window))[0]
        if idx.size < min_points:
            order = np.argsort(np.abs(t - center))
            idx = np.sort(order[: min(n, min_points)])
        if idx.size < min_points:
            out[i] = fallback[i]
            continue
        dt = t[idx] - center
        vandermonde = np.vander(dt, N=degree + 1, increasing=True)
        if idx.size > min_points:
            sigma = max(window_time / 4.0, 1e-12)
            weights = np.sqrt(np.exp(-0.5 * (dt / sigma) ** 2))
            a = vandermonde * weights[:, None]
            b = arr[idx] * weights[:, None]
        else:
            a = vandermonde
            b = arr[idx]
        try:
            coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
            out[i] = coeffs[1]
        except np.linalg.LinAlgError:
            out[i] = fallback[i]
    return out[:, 0] if scalar else out


def _time_window_slices(timestamps: FloatArray, window_time: float) -> list[slice]:
    t = np.asarray(timestamps, dtype=np.float64)
    radius = 0.5 * window_time
    n = len(t)
    left = 0
    right = -1
    slices: list[slice] = []
    for i in range(n):
        center = t[i]
        while left < n and t[left] < center - radius:
            left += 1
        while right + 1 < n and t[right + 1] <= center + radius:
            right += 1
        slices.append(slice(i, i + 1) if left > right else slice(left, right + 1))
    return slices


def _time_moving_rms(x: FloatArray, timestamps: FloatArray, window_time: float) -> FloatArray:
    out = np.empty(len(x), dtype=np.float64)
    for i, sl in enumerate(_time_window_slices(timestamps, window_time)):
        window = x[sl]
        out[i] = np.sqrt(np.mean(window * window))
    return cast(FloatArray, out)


def _time_moving_range(x: FloatArray, timestamps: FloatArray, window_time: float) -> FloatArray:
    out = np.empty(len(x), dtype=np.float64)
    for i, sl in enumerate(_time_window_slices(timestamps, window_time)):
        window = x[sl]
        out[i] = np.max(window) - np.min(window)
    return cast(FloatArray, out)


def _moving_rms(x: FloatArray, window_samples: int) -> FloatArray:
    return cast(FloatArray, np.sqrt(uniform_filter1d(x * x, size=window_samples, mode="nearest")))


def _moving_range(x: FloatArray, window_samples: int) -> FloatArray:
    hi = maximum_filter1d(x, size=window_samples, mode="nearest")
    lo = minimum_filter1d(x, size=window_samples, mode="nearest")
    return cast(FloatArray, hi - lo)


def adaptive_hysteresis_thresholds(
    activity: FloatArray,
    min_on: float = 0.003,
    min_off: float = 0.010,
    on_percentile: float = 45.0,
    off_percentile: float = 65.0,
) -> tuple[float, float]:
    """Estimate low-activity on/off thresholds from the activity distribution."""
    finite = activity[np.isfinite(activity)]
    if finite.size == 0:
        return min_on, min_off
    on_eps = max(min_on, float(np.percentile(finite, on_percentile)))
    off_eps = max(min_off, float(np.percentile(finite, off_percentile)), 1.25 * on_eps)
    return on_eps, off_eps


def adaptive_range_thresholds(
    spread: FloatArray,
    min_on: float = 0.0,
    min_off: float = 0.0,
    on_percentile: float = 45.0,
    off_percentile: float = 65.0,
) -> tuple[float, float]:
    """Estimate spread on/off thresholds from the spread distribution."""
    finite = spread[np.isfinite(spread)]
    if finite.size == 0:
        return float(min_on), float(min_off)
    on_eps = max(float(min_on), float(np.percentile(finite, on_percentile)), 1e-12)
    off_eps = max(float(min_off), float(np.percentile(finite, off_percentile)), 1.25 * on_eps)
    return on_eps, off_eps


def dual_hysteresis_mask(
    activity: FloatArray,
    activity_on: float,
    activity_off: float,
    spread: FloatArray,
    spread_on: float,
    spread_off: float,
) -> TimeBool:
    """Quiet mask using hysteresis on both activity and local spread."""
    mask = np.zeros(len(activity), dtype=np.bool_)
    in_quiet = False
    for i in range(len(activity)):
        a = float(activity[i])
        s = float(spread[i])
        if in_quiet:
            if a >= activity_off or s >= spread_off:
                in_quiet = False
        elif a <= activity_on and s <= spread_on:
            in_quiet = True
        mask[i] = in_quiet
    return mask


def score_hysteresis_mask(score: FloatArray, on_threshold: float, off_threshold: float) -> TimeBool:
    """High-score hysteresis mask (True while the score stays above thresholds)."""
    s = np.asarray(score, dtype=np.float64)
    mask = np.zeros(len(s), dtype=np.bool_)
    active = False
    for i in range(len(s)):
        val = float(s[i])
        if active:
            if val <= off_threshold:
                active = False
        elif val >= on_threshold:
            active = True
        mask[i] = active
    return mask


def compute_quiet_activity_and_spread(
    x: FloatArray,
    timestamps: FloatArray,
    params: QuietParams,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(activity, spread)`` per-frame metrics for a scalar/vector signal."""
    arr, _ = _ensure_2d(x)
    t = np.asarray(timestamps, dtype=np.float64)

    smoothed = _smooth(arr, t, params.pos_smooth_time, params.use_time_gaussian_smoothing)
    smoothed_2d, _ = _ensure_2d(smoothed)
    velocity = local_polynomial_derivative(
        smoothed_2d, t, params.derivative_window_time, params.derivative_poly_degree
    )
    velocity = _smooth(velocity, t, params.vel_smooth_time, params.use_time_gaussian_smoothing)
    velocity_2d, _ = _ensure_2d(velocity)
    speed = cast(FloatArray, np.linalg.norm(velocity_2d, axis=1))

    if params.use_time_windows:
        activity = _time_moving_rms(speed, t, params.quiet_window_time)
        component_ranges = np.column_stack(
            [_time_moving_range(smoothed_2d[:, c], t, params.spread_window_time) for c in range(smoothed_2d.shape[1])]
        )
    else:
        n_activity = _odd_window_samples(t, params.quiet_window_time)
        n_spread = _odd_window_samples(t, params.spread_window_time)
        activity = _moving_rms(speed, n_activity)
        component_ranges = np.column_stack(
            [_moving_range(smoothed_2d[:, c], n_spread) for c in range(smoothed_2d.shape[1])]
        )
    spread = cast(FloatArray, np.linalg.norm(component_ranges, axis=1))
    return activity, spread


def detect_quiet(
    x: FloatArray,
    timestamps: FloatArray,
    params: QuietParams = QuietParams(),
) -> QuietResult:
    """Detect intervals where ``x`` is locally quiet in both activity and spread.

    Args:
        x: Scalar ``(T,)`` or vector ``(T, D)`` motion samples.
        timestamps: Strictly increasing timestamps with shape ``(T,)``.
        params: Smoothing/threshold parameters.

    Returns:
        A :class:`QuietResult` with the cleaned quiet mask and activity/spread traces.
    """
    t = np.asarray(timestamps, dtype=np.float64)
    arr, _ = _ensure_2d(np.asarray(x, dtype=np.float64))
    if len(t) < 3:
        # Too short to estimate windows/derivatives; treat the whole clip as quiet.
        zeros = np.zeros(len(t), dtype=np.float64)
        return QuietResult(mask=cast(TimeBool, np.ones(len(t), dtype=np.bool_)), activity=zeros, spread=zeros)

    activity, spread = compute_quiet_activity_and_spread(arr, t, params)

    activity_on, activity_off = adaptive_hysteresis_thresholds(
        activity,
        min_on=params.min_activity_on_eps,
        min_off=params.min_activity_off_eps,
        on_percentile=params.on_percentile,
        off_percentile=params.off_percentile,
    )
    spread_on, spread_off = adaptive_range_thresholds(
        spread, on_percentile=params.on_percentile, off_percentile=params.off_percentile
    )

    mask = dual_hysteresis_mask(activity, activity_on, activity_off, spread, spread_on, spread_off)
    mask = clean_mask_by_time(t, mask, max_gap_time=params.max_gap_time, min_blip_time=params.min_blip_time)
    mask = mask_from_intervals(t, intervals_from_mask(t, mask, min_duration=params.min_interval_time))
    return QuietResult(
        mask=mask,
        activity=activity,
        spread=spread,
        activity_scale=activity_off if activity_off > 1e-12 else 1.0,
        spread_scale=spread_off if spread_off > 1e-12 else 1.0,
    )
