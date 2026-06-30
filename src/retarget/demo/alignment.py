"""Alignment primitives for cross-track timeline synchronization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from retarget.core.types import FloatArray1D, TimeLike
from retarget.demo.tracks import Track


type SignalExtractor = Callable[[Track], EnergySignal]
"""A function that extracts a scalar signal from a track."""


@dataclass(frozen=True, slots=True)
class EnergySignal:
    """Scalar alignment signal sampled on a timeline.

    ``timestamps`` and ``values`` must both have shape ``(T,)``. If an energy
    source starts as a vector-valued or spatial signal, callers should collapse
    it to this scalar representation before constructing ``EnergySignal``.
    """

    timestamps: FloatArray1D = field(repr=False)
    values: FloatArray1D = field(repr=False)
    name: str | None = None

    def __post_init__(self) -> None:
        name = None if self.name is None else str(self.name)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)

        if timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        if values.ndim != 1:
            raise ValueError("EnergySignal values must be a 1D scalar signal")
        if timestamps.shape != values.shape:
            raise ValueError("EnergySignal timestamps and values must have matching shape")
        if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("EnergySignal timestamps must be strictly increasing")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("EnergySignal timestamps must be finite")
        if not np.all(np.isfinite(values)):
            raise ValueError("EnergySignal values must be finite")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "values", values)

    def __repr__(self) -> str:
        result = "EnergySignal("
        if self.name is not None:
            result += f"name={self.name!r}, "
        start = float(self.timestamps[0]) if len(self.timestamps) else None
        stop = float(self.timestamps[-1]) if len(self.timestamps) else None
        result += (
            f"samples={len(self.timestamps)}, "
            f"start={start!r}, "
            f"stop={stop!r}, "
        )
        if len(self.values) > 0:
            mean = float(np.mean(self.values))
            std = float(np.std(self.values))
            vmin = float(np.min(self.values))
            vmax = float(np.max(self.values))
            result += f"mean={mean:.6f}, std={std:.6f}, min={vmin:.6f}, max={vmax:.6f}"
        return result + ")"


@dataclass(frozen=True, slots=True)
class TimelineTransform:
    """Affine mapping from source-track time to reference-track time.

    ``TimelineTransform`` maps source time to reference time::

        t_reference = scale * t_source + offset

    Example with ``scale=1``: if a reference event happens at 1.0 s and the
    corresponding source event happens at 1.3 s, then ``offset = -0.3`` maps
    ``1.3 -> 1.0``.
    """

    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        scale = float(self.scale)
        offset = float(self.offset)
        if not np.isfinite(scale):
            raise ValueError("TimelineTransform scale must be finite")
        if not np.isfinite(offset):
            raise ValueError("TimelineTransform offset must be finite")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)

    @classmethod
    def identity(cls) -> TimelineTransform:
        """Return the identity timeline transform."""
        return cls(scale=1.0, offset=0.0)

    def to_reference(self, source_time: TimeLike) -> TimeLike:
        """Map source-track time into reference-track time."""
        return self.scale * source_time + self.offset

    def to_source(self, reference_time: TimeLike) -> TimeLike:
        """Map reference-track time back into source-track time."""
        if self.scale == 0.0:
            raise ZeroDivisionError("cannot invert a zero-scale TimelineTransform")
        return (reference_time - self.offset) / self.scale

    def inverse(self) -> TimelineTransform:
        """Return the inverse transform."""
        if self.scale == 0.0:
            raise ZeroDivisionError("cannot invert a zero-scale TimelineTransform")
        return TimelineTransform(
            scale=1.0 / self.scale,
            offset=-self.offset / self.scale,
        )

    def then(self, other: TimelineTransform) -> TimelineTransform:
        """Compose this transform with another transform.

        The returned transform applies ``self`` first, then ``other``.
        """
        return TimelineTransform(
            scale=other.scale * self.scale,
            offset=other.scale * self.offset + other.offset,
        )


@dataclass(frozen=True, slots=True)
class TrackAlignment:
    """Recorded alignment from one track timeline into another.

    ``source`` and ``reference`` are authored track names (strings).
    """

    source: str
    reference: str
    transform: TimelineTransform
    score: float | None = None

    def __post_init__(self) -> None:
        if self.score is not None and not np.isfinite(float(self.score)):
            raise ValueError("TrackAlignment score must be finite when provided")


def estimate_alignment_from_signals(
    *,
    reference: EnergySignal,
    source: EnergySignal,
    max_lag_seconds: float,
) -> tuple[TimelineTransform, float]:
    """Estimate an offset-only transform using normalized cross-correlation.

    Both signals are resampled to a common uniform grid before correlation.
    The returned transform maps source time into reference time.
    """
    if max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be non-negative")

    ref_times = reference.timestamps
    src_times = source.timestamps
    if len(ref_times) < 2 or len(src_times) < 2:
        return TimelineTransform.identity(), 0.0

    start = max(float(ref_times[0]), float(src_times[0]))
    stop = min(float(ref_times[-1]), float(src_times[-1]))
    if stop <= start:
        return TimelineTransform.identity(), 0.0

    median_dt = _median_positive_spacing([ref_times, src_times])
    grid = np.arange(start, stop, median_dt)
    if len(grid) < 2:
        return TimelineTransform.identity(), 0.0

    ref_values = _normalize(_resample_to_grid(reference, grid))
    src_values = _normalize(_resample_to_grid(source, grid))

    max_lag_samples = int(round(max_lag_seconds / median_dt))
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        ref_slice, src_slice = _lagged_slices(ref_values, src_values, lag)
        if len(ref_slice) < 2:
            continue
        score = float(np.dot(ref_slice, src_slice) / len(ref_slice))
        if score > best_score:
            best_score = score
            best_lag = lag

    if not np.isfinite(best_score):
        return TimelineTransform.identity(), 0.0

    offset = -best_lag * median_dt
    return TimelineTransform(scale=1.0, offset=offset), best_score


def _median_positive_spacing(timelines: Sequence[FloatArray1D]) -> float:
    dts = np.concatenate([np.diff(timeline) for timeline in timelines])
    dts = dts[dts > 0]
    if len(dts) == 0:
        raise ValueError("Cannot estimate alignment without positive timestamp spacing")
    return float(np.median(dts))


def _lagged_slices(
    reference_values: FloatArray1D,
    source_values: FloatArray1D,
    lag: int,
) -> tuple[FloatArray1D, FloatArray1D]:
    if lag < 0:
        reference_slice = reference_values[-lag:]
        source_slice = source_values[: len(reference_slice)]
    elif lag > 0:
        reference_slice = reference_values[: len(reference_values) - lag]
        source_slice = source_values[lag : lag + len(reference_slice)]
    else:
        reference_slice = reference_values
        source_slice = source_values
    return reference_slice, source_slice


def _resample_to_grid(signal: EnergySignal, grid: FloatArray1D) -> FloatArray1D:
    return np.interp(grid, signal.timestamps, signal.values)


def _normalize(values: FloatArray1D) -> FloatArray1D:
    centered = values - np.mean(values)
    std = float(np.std(centered))
    if std <= 1e-12:
        raise ValueError("Cannot estimate alignment from a constant energy signal")
    return centered / std
