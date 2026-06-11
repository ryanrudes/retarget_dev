import numpy as np

from retarget.core.types import FloatArray


def estimate_nominal_hz(timestamps: FloatArray) -> float:
    """Estimate nominal sampling rate from timestamps.

    Uses the median positive timestamp difference, which is robust to small
    jitter and occasional dropped frames.

    Args:
        timestamps: Native timestamps, shape (T,).

    Returns:
        Estimated sampling rate in Hz.

    Raises:
        ValueError: If timestamps are not 1D, have fewer than two samples,
            or have no positive timestamp differences.
    """
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")

    if timestamps.size < 2:
        raise ValueError("at least two timestamps are required to estimate nominal_hz")

    diffs = np.diff(timestamps)
    positive_diffs = diffs[diffs > 0.0]

    if positive_diffs.size == 0:
        raise ValueError("timestamps must contain at least one positive difference")

    median_dt = float(np.median(positive_diffs))
    if median_dt <= 0.0:
        raise ValueError("median positive timestamp difference must be positive")

    return 1.0 / median_dt


def validate_nominal_hz(nominal_hz: float | None) -> float | None:
    """Validate an optional explicit nominal sampling rate."""
    if nominal_hz is None:
        return None

    hz = float(nominal_hz)
    if hz <= 0.0:
        raise ValueError("nominal_hz must be positive")

    return hz