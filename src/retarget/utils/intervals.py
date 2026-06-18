"""Boolean-mask / run-length interval utilities.

These convert between per-sample boolean masks and ``(start, end)`` time
intervals, and perform temporal cleanup (fill brief gaps, drop brief blips).
They are generic array helpers with no dependency on the scene/demo layers and
are shared by contact detection and quiet-interval detection.

All functions treat intervals as closed ``[start, end]`` in the same time units
as ``timestamps``.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from retarget.core.types import FloatArray, TimeBool

type IntervalList = list[tuple[float, float]]
"""A list of closed ``(start, end)`` time intervals."""


def _validate_mask(timestamps: FloatArray, mask: TimeBool) -> tuple[FloatArray, TimeBool]:
    t = np.asarray(timestamps, dtype=np.float64)
    m = np.asarray(mask, dtype=np.bool_)
    if t.ndim != 1 or m.ndim != 1:
        raise ValueError("timestamps and mask must be 1D arrays")
    if len(t) != len(m):
        raise ValueError("timestamps and mask must have the same length")
    return t, cast(TimeBool, m)


def _runs(mask: TimeBool, value: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (start_idx, end_idx_exclusive) arrays for contiguous ``mask == value`` runs."""
    value_mask = mask == value
    padded = np.r_[False, value_mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return starts, ends


def intervals_from_mask(
    timestamps: FloatArray,
    mask: TimeBool,
    min_duration: float = 0.0,
) -> IntervalList:
    """Convert a boolean mask into ``(start, end)`` time intervals.

    Args:
        timestamps: Timestamps with shape ``(N,)``.
        mask: Boolean mask with shape ``(N,)``.
        min_duration: Drop intervals shorter than this many seconds.

    Returns:
        Contiguous runs where ``mask`` is True, in the time units of ``timestamps``.
    """
    t, m = _validate_mask(timestamps, mask)
    if len(t) == 0:
        return []
    starts, ends = _runs(m, True)
    intervals: IntervalList = []
    for start, end in zip(starts, ends):
        start_time = float(t[start])
        end_time = float(t[end - 1])
        if end_time - start_time >= min_duration:
            intervals.append((start_time, end_time))
    return intervals


def mask_from_intervals(timestamps: FloatArray, intervals: IntervalList) -> TimeBool:
    """Build a per-sample mask that is True inside any of the given intervals.

    Args:
        timestamps: Timestamps with shape ``(N,)``.
        intervals: Closed intervals ``(start, end)`` in the units of ``timestamps``.

    Returns:
        Boolean mask with shape ``(N,)``.
    """
    t = np.asarray(timestamps, dtype=np.float64)
    mask = np.zeros(len(t), dtype=np.bool_)
    for start, end in intervals:
        mask |= (t >= start) & (t <= end)
    return mask


def _run_durations(timestamps: FloatArray, mask: TimeBool, value: bool) -> list[tuple[int, int, float]]:
    """Return contiguous runs of ``mask == value`` as ``(start_idx, end_idx, duration)``."""
    t, m = _validate_mask(timestamps, mask)
    if len(t) == 0:
        return []
    starts, ends = _runs(m, value)
    runs: list[tuple[int, int, float]] = []
    for start, end in zip(starts, ends):
        start_idx = int(start)
        end_idx = int(end) - 1
        duration = float(t[end_idx] - t[start_idx])
        runs.append((start_idx, end_idx, duration))
    return runs


def fill_short_false_runs(timestamps: FloatArray, mask: TimeBool, max_gap_time: float) -> TimeBool:
    """Fill brief interior False gaps surrounded by True runs (hole filling).

    Runs touching either series edge are never filled (their true extent is
    unknown beyond the recording).

    Args:
        timestamps: Timestamps with shape ``(N,)``.
        mask: Boolean mask with shape ``(N,)``.
        max_gap_time: Maximum interior False-run duration to fill (seconds).

    Returns:
        Copy of ``mask`` with qualifying interior False runs set to True.
    """
    cleaned = np.asarray(mask, dtype=np.bool_).copy()
    if max_gap_time <= 0.0:
        return cast(TimeBool, cleaned)
    for start, end, duration in _run_durations(timestamps, cast(TimeBool, cleaned), False):
        touches_edge = start == 0 or end == len(cleaned) - 1
        if not touches_edge and duration <= max_gap_time:
            cleaned[start : end + 1] = True
    return cast(TimeBool, cleaned)


def remove_short_true_runs(timestamps: FloatArray, mask: TimeBool, min_duration: float) -> TimeBool:
    """Remove brief True blips shorter than ``min_duration`` seconds.

    Args:
        timestamps: Timestamps with shape ``(N,)``.
        mask: Boolean mask with shape ``(N,)``.
        min_duration: True runs shorter than this are cleared.

    Returns:
        Copy of ``mask`` with short True blips removed.
    """
    cleaned = np.asarray(mask, dtype=np.bool_).copy()
    if min_duration <= 0.0:
        return cast(TimeBool, cleaned)
    for start, end, duration in _run_durations(timestamps, cast(TimeBool, cleaned), True):
        if duration < min_duration:
            cleaned[start : end + 1] = False
    return cast(TimeBool, cleaned)


def clean_mask_by_time(
    timestamps: FloatArray,
    mask: TimeBool,
    max_gap_time: float = 0.10,
    min_blip_time: float = 0.15,
) -> TimeBool:
    """Fill short gaps then remove short True blips.

    Args:
        timestamps: Timestamps with shape ``(N,)``.
        mask: Boolean mask with shape ``(N,)``.
        max_gap_time: Passed to :func:`fill_short_false_runs`.
        min_blip_time: Passed to :func:`remove_short_true_runs`.

    Returns:
        Temporally cleaned boolean mask with shape ``(N,)``.
    """
    cleaned = fill_short_false_runs(timestamps, mask, max_gap_time)
    return remove_short_true_runs(timestamps, cleaned, min_blip_time)
