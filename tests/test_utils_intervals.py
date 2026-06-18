"""Tests for boolean-mask / run-length interval utilities."""

from __future__ import annotations

import numpy as np
import pytest

from retarget.utils.intervals import (
    clean_mask_by_time,
    fill_short_false_runs,
    intervals_from_mask,
    mask_from_intervals,
    remove_short_true_runs,
)


def test_intervals_from_mask_extracts_contiguous_runs() -> None:
    t = np.arange(6, dtype=np.float64)
    mask = np.array([False, True, True, False, True, False], dtype=np.bool_)
    assert intervals_from_mask(t, mask) == [(1.0, 2.0), (4.0, 4.0)]


def test_intervals_from_mask_min_duration_drops_short() -> None:
    t = np.arange(6, dtype=np.float64)
    mask = np.array([False, True, True, False, True, False], dtype=np.bool_)
    # The single-sample run at index 4 has zero duration and is dropped.
    assert intervals_from_mask(t, mask, min_duration=0.5) == [(1.0, 2.0)]


def test_mask_from_intervals_round_trips() -> None:
    t = np.arange(6, dtype=np.float64)
    mask = np.array([False, True, True, False, True, False], dtype=np.bool_)
    intervals = intervals_from_mask(t, mask)
    np.testing.assert_array_equal(mask_from_intervals(t, intervals), mask)


def test_empty_inputs() -> None:
    empty = np.empty((0,), dtype=np.float64)
    empty_mask = np.empty((0,), dtype=np.bool_)
    assert intervals_from_mask(empty, empty_mask) == []
    np.testing.assert_array_equal(
        mask_from_intervals(empty, []), np.empty((0,), dtype=np.bool_)
    )


def test_fill_short_false_runs_fills_interior_not_edges() -> None:
    t = np.arange(7, dtype=np.float64)
    # interior gap of length 1 at index 3; leading gap at index 0.
    mask = np.array([False, True, True, False, True, True, False], dtype=np.bool_)
    filled = fill_short_false_runs(t, mask, max_gap_time=1.0)
    np.testing.assert_array_equal(
        filled, np.array([False, True, True, True, True, True, False], dtype=np.bool_)
    )
    # A trailing/leading False run is never filled even if short.
    assert not filled[0]
    assert not filled[-1]


def test_fill_short_false_runs_respects_max_gap() -> None:
    t = np.arange(7, dtype=np.float64)
    mask = np.array([True, True, False, False, False, True, True], dtype=np.bool_)
    # gap is 3 samples (indices 2..4), spanning 2.0s; too long to fill.
    np.testing.assert_array_equal(fill_short_false_runs(t, mask, max_gap_time=1.0), mask)


def test_remove_short_true_runs() -> None:
    t = np.arange(6, dtype=np.float64)
    mask = np.array([True, False, True, True, True, False], dtype=np.bool_)
    # First True run is a single sample (0s duration) -> removed at any positive min.
    cleaned = remove_short_true_runs(t, mask, min_duration=0.5)
    np.testing.assert_array_equal(
        cleaned, np.array([False, False, True, True, True, False], dtype=np.bool_)
    )


def test_clean_mask_by_time_fills_then_removes() -> None:
    t = np.arange(14, dtype=np.float64)
    mask = np.array(
        [True, True, True, True, False, True, True, True, True, False, False, False, True, False],
        dtype=np.bool_,
    )
    cleaned = clean_mask_by_time(t, mask, max_gap_time=1.0, min_blip_time=1.5)
    # The 1-sample gap at index 4 is filled (interior, <=1s) merging runs 0..8;
    # the 3-sample gap at 9..11 (2s) is too long to fill, leaving the lone True at
    # index 12 as a short blip that is then removed.
    np.testing.assert_array_equal(
        cleaned,
        np.array(
            [True, True, True, True, True, True, True, True, True, False, False, False, False, False],
            dtype=np.bool_,
        ),
    )


def test_does_not_mutate_input() -> None:
    t = np.arange(5, dtype=np.float64)
    mask = np.array([True, False, True, True, False], dtype=np.bool_)
    original = mask.copy()
    fill_short_false_runs(t, mask, max_gap_time=1.0)
    remove_short_true_runs(t, mask, min_duration=1.0)
    np.testing.assert_array_equal(mask, original)


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        intervals_from_mask(np.arange(3, dtype=np.float64), np.array([True, False], dtype=np.bool_))
