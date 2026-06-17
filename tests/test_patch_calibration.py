from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    BodyFrameTranslation,
    RectangularRegion,
    SemanticAxis,
    calibrate_patch_transform,
)

_POSITIONS = {
    "a": np.array([0.0, 0.0, 0.0]),
    "b": np.array([1.0, 0.0, 0.0]),
    "c": np.array([0.0, 1.0, 0.0]),
    "d": np.array([0.0, 0.0, 1.0]),
}


def test_calibration_fits_centroid_for_planar_markers() -> None:
    transform = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        markers=("a", "b", "c"),
    )
    np.testing.assert_allclose(
        transform.translation,
        np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    )
    np.testing.assert_allclose(abs(transform.rotation[:, 2][2]), 1.0)


def test_normal_offset_moves_origin_along_fitted_normal() -> None:
    base = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        markers=("a", "b", "c"),
    )
    offset = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        markers=("a", "b", "c"),
        normal_offset=0.1,
    )
    expected = base.translation + 0.1 * base.rotation[:, 2]
    np.testing.assert_allclose(offset.translation, expected)


def test_body_frame_marker_translations_applied_before_fitting() -> None:
    transform = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        markers=("a", "b", "c"),
        marker_translations={"a": BodyFrameTranslation(np.array([0.0, 0.0, 0.3]))},
    )
    # Lifting marker 'a' by 0.3 raises the centroid by 0.1 in z.
    np.testing.assert_allclose(transform.translation[2], 0.1)


def test_semantic_axis_marker_translations_supported() -> None:
    transform = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        markers=("a", "b", "c"),
        marker_translations={"a": 0.3 * SemanticAxis.UP},
    )
    np.testing.assert_allclose(transform.translation[2], 0.1)


def test_calibration_requires_at_least_three_markers() -> None:
    with pytest.raises(ValueError, match="at least three markers"):
        calibrate_patch_transform(
            marker_positions_segment=_POSITIONS,
            markers=("a", "b"),
        )


def test_rectangular_region_contains() -> None:
    region = RectangularRegion(width=1.0, height=2.0)
    assert region.contains(np.array([0.4, 0.9]))
    assert not region.contains(np.array([0.6, 0.0]))
