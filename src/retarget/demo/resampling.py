from __future__ import annotations

from enum import StrEnum

import numpy as np


class ResampleMethod(StrEnum):
    """Time-axis resampling policy."""

    NEAREST = "nearest"
    """Resample to the nearest source timestamp."""

    PREVIOUS = "previous"
    """Resample to the previous source timestamp."""


def nearest_indices(
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    insertion_points = np.searchsorted(source_timestamps, target_timestamps)
    right = np.clip(insertion_points, 0, len(source_timestamps) - 1)
    left = np.clip(insertion_points - 1, 0, len(source_timestamps) - 1)

    choose_right = (
        np.abs(source_timestamps[right] - target_timestamps)
        < np.abs(target_timestamps - source_timestamps[left])
    )
    return np.where(choose_right, right, left).astype(np.intp, copy=False)


def previous_indices(
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(source_timestamps, target_timestamps, side="right") - 1
    return np.clip(indices, 0, len(source_timestamps) - 1).astype(np.intp, copy=False)


def resample_indices(
    *,
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    method: ResampleMethod | str,
) -> np.ndarray:
    method = ResampleMethod(method)

    if len(source_timestamps) == 0:
        if len(target_timestamps) == 0:
            return np.empty((0,), dtype=np.intp)
        raise ValueError(
            "cannot resample from empty source timestamps "
            "to non-empty target timestamps"
        )

    if method is ResampleMethod.NEAREST:
        return nearest_indices(source_timestamps, target_timestamps)
    if method is ResampleMethod.PREVIOUS:
        return previous_indices(source_timestamps, target_timestamps)
    raise ValueError(f"unsupported contact resample method: {method!r}")


def validate_strictly_increasing_timestamps(timestamps: np.ndarray) -> None:
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")


def validate_resample_timestamps(timestamps: np.ndarray) -> np.ndarray:
    target_timestamps = np.asarray(timestamps, dtype=np.float64)
    if target_timestamps.ndim != 1:
        raise ValueError("resample timestamps must be a 1D array")
    if not np.all(np.isfinite(target_timestamps)):
        raise ValueError("resample timestamps must be finite")
    validate_strictly_increasing_timestamps(target_timestamps)
    return target_timestamps