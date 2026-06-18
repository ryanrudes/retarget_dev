"""Tests for quiet-interval detection."""

from __future__ import annotations

import numpy as np

from retarget.contacts._quiet import (
    QuietParams,
    detect_quiet,
    local_polynomial_derivative,
    score_hysteresis_mask,
)


def _lift_in_middle_signal() -> tuple[np.ndarray, np.ndarray]:
    """A 3D point that rests at the ends and lifts (triangular) during [1, 2]s.

    The recording is mostly quiet, matching the detector's adaptive-threshold
    assumption; the foot clearly moves through the middle window.
    """
    t = np.linspace(0.0, 3.0, 301)
    z = np.zeros_like(t)
    up = (t > 1.0) & (t <= 1.5)
    down = (t > 1.5) & (t < 2.0)
    z[up] = 0.5 * (t[up] - 1.0)
    z[down] = 0.5 * (2.0 - t[down])
    points = np.column_stack([np.zeros_like(t), np.zeros_like(t), z])
    return t, points


def test_detect_quiet_marks_rest_intervals() -> None:
    t, points = _lift_in_middle_signal()
    result = detect_quiet(points, t)

    assert result.mask.shape == (t.shape[0],)
    assert result.activity.shape == (t.shape[0],)
    assert result.spread.shape == (t.shape[0],)

    # The resting windows at both ends are quiet.
    assert result.mask[(t >= 0.2) & (t <= 0.8)].all()
    assert result.mask[(t >= 2.2) & (t <= 2.8)].all()

    # The lift is clearly moving and not quiet.
    assert not result.mask[np.argmin(np.abs(t - 1.25))]
    assert not result.mask[np.argmin(np.abs(t - 1.75))]


def test_detect_quiet_short_clip_is_all_quiet() -> None:
    t = np.array([0.0, 0.1], dtype=np.float64)
    points = np.zeros((2, 3), dtype=np.float64)
    result = detect_quiet(points, t)
    assert result.mask.all()


def test_local_polynomial_derivative_matches_gradient_without_window() -> None:
    t = np.linspace(0.0, 1.0, 50)
    x = np.sin(2.0 * np.pi * t)
    deriv = local_polynomial_derivative(x, t, window_time=None)
    np.testing.assert_allclose(deriv, np.gradient(x, t), rtol=1e-6)


def test_local_polynomial_derivative_recovers_linear_slope() -> None:
    t = np.linspace(0.0, 2.0, 80)
    slope = 3.0
    x = slope * t + 0.5
    deriv = local_polynomial_derivative(x, t, window_time=0.2, degree=1)
    np.testing.assert_allclose(deriv, np.full_like(t, slope), atol=1e-6)


def test_score_hysteresis_mask_enter_and_exit() -> None:
    score = np.array([0.0, 0.5, 0.7, 0.5, 0.45, 0.3, 0.5, 0.7], dtype=np.float64)
    mask = score_hysteresis_mask(score, on_threshold=0.6, off_threshold=0.4)
    # Enters at index 2 (>=0.6), stays through 0.5/0.45 (>0.4), exits at 0.3 (<=0.4),
    # re-enters at index 7.
    expected = np.array([False, False, True, True, True, False, False, True], dtype=np.bool_)
    np.testing.assert_array_equal(mask, expected)


def test_quiet_params_are_frozen() -> None:
    params = QuietParams(quiet_window_time=0.2)
    assert params.quiet_window_time == 0.2
    assert params.derivative_poly_degree == 1
