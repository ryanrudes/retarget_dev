from __future__ import annotations

import numpy as np
import pytest

from retarget.demo.resampling import (
    ResampleMethod,
    nearest_indices,
    previous_indices,
    resample_indices,
    validate_resample_timestamps,
)


def test_validate_resample_timestamps_accepts_strictly_increasing_1d() -> None:
    timestamps = validate_resample_timestamps([0.0, 0.5, 1.0])

    np.testing.assert_array_equal(timestamps, np.array([0.0, 0.5, 1.0]))
    assert timestamps.dtype == np.float64


def test_validate_resample_timestamps_allows_empty_array() -> None:
    timestamps = validate_resample_timestamps([])

    np.testing.assert_array_equal(timestamps, np.array([], dtype=np.float64))


def test_validate_resample_timestamps_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="1D array"):
        validate_resample_timestamps(np.array([[0.0, 1.0]]))


def test_validate_resample_timestamps_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_resample_timestamps([0.0, np.inf])


def test_validate_resample_timestamps_rejects_non_increasing_values() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_resample_timestamps([0.0, 0.5, 0.5])


def test_nearest_indices_picks_closest_source_samples() -> None:
    source = np.array([0.0, 1.0, 2.0, 4.0])
    target = np.array([0.1, 0.9, 2.6, 3.7])

    indices = nearest_indices(source, target)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2, 3], dtype=np.intp))


def test_nearest_indices_breaks_ties_toward_left_sample() -> None:
    source = np.array([0.0, 1.0, 2.0])
    target = np.array([0.5, 1.5])

    indices = nearest_indices(source, target)

    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.intp))


def test_nearest_indices_clamps_outside_source_range() -> None:
    source = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 4.0])

    indices = nearest_indices(source, target)

    np.testing.assert_array_equal(indices, np.array([0, 2], dtype=np.intp))


def test_previous_indices_picks_previous_source_samples() -> None:
    source = np.array([0.0, 1.0, 2.0, 4.0])
    target = np.array([0.1, 1.0, 2.6, 4.0])

    indices = previous_indices(source, target)

    np.testing.assert_array_equal(indices, np.array([0, 1, 2, 3], dtype=np.intp))


def test_previous_indices_clamps_outside_source_range() -> None:
    source = np.array([1.0, 2.0, 3.0])
    target = np.array([0.0, 4.0])

    indices = previous_indices(source, target)

    np.testing.assert_array_equal(indices, np.array([0, 2], dtype=np.intp))


def test_resample_indices_dispatches_nearest_method() -> None:
    indices = resample_indices(
        source_timestamps=np.array([0.0, 1.0, 2.0]),
        target_timestamps=np.array([0.8, 1.6]),
        method=ResampleMethod.NEAREST,
    )

    np.testing.assert_array_equal(indices, np.array([1, 2], dtype=np.intp))


def test_resample_indices_dispatches_previous_method() -> None:
    indices = resample_indices(
        source_timestamps=np.array([0.0, 1.0, 2.0]),
        target_timestamps=np.array([0.8, 1.6]),
        method=ResampleMethod.PREVIOUS,
    )

    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.intp))


def test_resample_indices_accepts_string_method() -> None:
    indices = resample_indices(
        source_timestamps=np.array([0.0, 1.0, 2.0]),
        target_timestamps=np.array([0.8, 1.6]),
        method="previous",
    )

    np.testing.assert_array_equal(indices, np.array([0, 1], dtype=np.intp))


def test_resample_indices_returns_empty_for_empty_source_and_empty_target() -> None:
    indices = resample_indices(
        source_timestamps=np.array([], dtype=np.float64),
        target_timestamps=np.array([], dtype=np.float64),
        method=ResampleMethod.NEAREST,
    )

    np.testing.assert_array_equal(indices, np.array([], dtype=np.intp))


def test_resample_indices_rejects_empty_source_with_non_empty_target() -> None:
    with pytest.raises(ValueError, match="empty source"):
        resample_indices(
            source_timestamps=np.array([], dtype=np.float64),
            target_timestamps=np.array([0.0]),
            method=ResampleMethod.NEAREST,
        )