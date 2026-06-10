"""Alignment primitives for cross-track timeline synchronization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retarget.core.enums import TrackId


@dataclass(frozen=True, slots=True)
class EnergySignal:
    """Scalar or vector energy signal sampled on a timeline."""

    timestamps: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        if len(timestamps) != values.shape[0]:
            raise ValueError("values first dimension must match timestamps length")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("EnergySignal timestamps must be strictly increasing")

    @property
    def scalar_values(self) -> np.ndarray:
        if self.values.ndim == 1:
            return self.values
        return np.linalg.norm(self.values, axis=1)


@dataclass(frozen=True, slots=True)
class TimelineTransform:
    """Affine mapping from source-track time to reference-track time.

    ``TimelineTransform`` maps source time to reference time::

        t_reference = scale * t_source + offset

    Example (``scale=1``): if the reference peak is at 1.0 s and the
    corresponding source peak is at 1.3 s, then ``offset = -0.3`` so that
    ``t_reference = t_source - 0.3``.
    """

    scale: float = 1.0
    offset: float = 0.0

    def to_reference(self, t_source: np.ndarray) -> np.ndarray:
        return self.scale * t_source + self.offset

    def to_source(self, t_reference: np.ndarray) -> np.ndarray:
        return (t_reference - self.offset) / self.scale


@dataclass(frozen=True, slots=True)
class TrackAlignment[K: TrackId]:
    """Recorded alignment between two demonstration tracks."""

    source: K
    reference: K
    transform: TimelineTransform
    score: float
    method: str = "cross_correlation"


def estimate_alignment_from_signals(
    *,
    reference: EnergySignal,
    source: EnergySignal,
    max_lag_seconds: float,
) -> tuple[TimelineTransform, float]:
    """
    Estimate an offset-only timeline transform via normalized cross-correlation.

    Both signals are resampled to a common uniform grid before correlation.
    """
    if max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be non-negative")

    ref_times = reference.timestamps
    src_times = source.timestamps
    if len(ref_times) < 2 or len(src_times) < 2:
        return TimelineTransform(scale=1.0, offset=0.0), 0.0

    start = max(float(ref_times[0]), float(src_times[0]))
    stop = min(float(ref_times[-1]), float(src_times[-1]))
    if stop <= start:
        return TimelineTransform(scale=1.0, offset=0.0), 0.0

    dts = np.concatenate([
        np.diff(ref_times),
        np.diff(src_times),
    ])
    dts = dts[dts > 0]
    if len(dts) == 0:
        raise ValueError("Cannot estimate alignment without positive timestamp spacing")
    median_dt = float(np.median(dts))

    grid = np.arange(start, stop, median_dt)
    if len(grid) < 2:
        return TimelineTransform(scale=1.0, offset=0.0), 0.0

    ref_values = _normalize(_resample_to_grid(reference, grid))
    src_values = _normalize(_resample_to_grid(source, grid))

    max_lag_samples = max(1, int(round(max_lag_seconds / median_dt)))
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            ref_slice = ref_values[-lag:]
            src_slice = src_values[: len(ref_slice)]
        elif lag > 0:
            ref_slice = ref_values[: len(ref_values) - lag]
            src_slice = src_values[lag : lag + len(ref_slice)]
        else:
            ref_slice = ref_values
            src_slice = src_values
        if len(ref_slice) < 2:
            continue
        score = float(np.dot(ref_slice, src_slice) / len(ref_slice))
        if score > best_score:
            best_score = score
            best_lag = lag

    offset = -best_lag * median_dt
    return TimelineTransform(scale=1.0, offset=offset), best_score


def _resample_to_grid(signal: EnergySignal, grid: np.ndarray) -> np.ndarray:
    scalar = signal.scalar_values
    return np.interp(grid, signal.timestamps, scalar)


def _normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values)
    std = float(np.std(centered))
    if std <= 1e-12:
        raise ValueError("Cannot estimate alignment from a constant energy signal")
    return centered / std
