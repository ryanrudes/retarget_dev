from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    BodyFrameTranslation,
    MarkerId,
    MarkerSetSpec,
    PatchCalibrationSpec,
    PatchId,
    RectangularRegion,
    SegmentId,
    SegmentSpec,
    SemanticAxis,
    Z_UP_AXES,
)


class TestMarkerId(MarkerId):
    A = "a"
    B = "b"
    C = "c"
    D = "d"


class TestPatchId(PatchId):
    SURFACE = "surface"


class TestSegmentId(SegmentId):
    BODY = "body"


def make_segment(
    calibration: PatchCalibrationSpec[TestMarkerId, TestPatchId],
) -> SegmentSpec[TestMarkerId, TestPatchId]:
    return SegmentSpec(
        segment=TestSegmentId.BODY,
        marker_type=TestMarkerId,
        patch_type=TestPatchId,
        axis_convention=Z_UP_AXES,
        marker_set=MarkerSetSpec(marker_type=TestMarkerId),
        marker_positions_segment={
            TestMarkerId.A: np.array([0.0, 0.0, 0.0]),
            TestMarkerId.B: np.array([1.0, 0.0, 0.0]),
            TestMarkerId.C: np.array([0.0, 1.0, 0.0]),
            TestMarkerId.D: np.array([0.0, 0.0, 1.0]),
        },
        patch_calibrations={
            TestPatchId.SURFACE: calibration,
        },
    )


def test_patch_calibration_accepts_markers_without_translations() -> None:
    calibration = PatchCalibrationSpec(
        patch=TestPatchId.SURFACE,
        markers=(TestMarkerId.A, TestMarkerId.B, TestMarkerId.C),
        region=RectangularRegion(width=1.0, height=1.0),
    )
    segment = make_segment(calibration).with_built_patches()
    patch = segment.patch_spec(TestPatchId.SURFACE)
    np.testing.assert_allclose(
        patch.transform_segment_patch.translation,
        np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    )


def test_normal_offset_moves_patch_origin_along_fitted_normal() -> None:
    calibration = PatchCalibrationSpec(
        patch=TestPatchId.SURFACE,
        markers=(TestMarkerId.A, TestMarkerId.B, TestMarkerId.C),
        normal_offset=0.1,
        region=RectangularRegion(width=1.0, height=1.0),
    )
    segment = make_segment(calibration).with_built_patches()
    patch = segment.patch_spec(TestPatchId.SURFACE)
    normal = patch.transform_segment_patch.rotation[:, 2]
    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]) + 0.1 * normal
    np.testing.assert_allclose(
        patch.transform_segment_patch.translation,
        expected,
    )


def test_sparse_marker_translations_are_applied_before_fitting() -> None:
    calibration = PatchCalibrationSpec(
        patch=TestPatchId.SURFACE,
        markers=(TestMarkerId.A, TestMarkerId.B, TestMarkerId.C),
        marker_translations={
            TestMarkerId.A: 0.1 * SemanticAxis.UP,
        },
        region=RectangularRegion(width=1.0, height=1.0),
    )
    surface_points = calibration.surface_points(
        marker_positions_segment={
            TestMarkerId.A: np.array([0.0, 0.0, 0.0]),
            TestMarkerId.B: np.array([1.0, 0.0, 0.0]),
            TestMarkerId.C: np.array([0.0, 1.0, 0.0]),
        },
        segment=make_segment(calibration),
    )
    np.testing.assert_allclose(
        surface_points[0],
        np.array([0.0, 0.0, 0.1]),
    )
    np.testing.assert_allclose(
        surface_points[1],
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        surface_points[2],
        np.array([0.0, 1.0, 0.0]),
    )


def test_body_frame_marker_translation_supports_cad_offsets() -> None:
    calibration = PatchCalibrationSpec(
        patch=TestPatchId.SURFACE,
        markers=(TestMarkerId.A, TestMarkerId.B, TestMarkerId.C),
        marker_translations={
            TestMarkerId.A: BodyFrameTranslation(
                np.array([0.0, 0.0, -0.012])
            ),
            TestMarkerId.B: BodyFrameTranslation(
                np.array([0.0, 0.0, -0.010])
            ),
            TestMarkerId.C: BodyFrameTranslation(
                np.array([0.0, 0.0, -0.011])
            ),
        },
        region=RectangularRegion(width=1.0, height=1.0),
    )
    surface_points = calibration.surface_points(
        marker_positions_segment={
            TestMarkerId.A: np.array([0.0, 0.0, 0.0]),
            TestMarkerId.B: np.array([1.0, 0.0, 0.0]),
            TestMarkerId.C: np.array([0.0, 1.0, 0.0]),
        },
        segment=make_segment(calibration),
    )
    np.testing.assert_allclose(
        surface_points,
        np.array(
            [
                [0.0, 0.0, -0.012],
                [1.0, 0.0, -0.010],
                [0.0, 1.0, -0.011],
            ]
        ),
    )


def test_marker_translations_must_be_subset_of_markers() -> None:
    with pytest.raises(ValueError, match="not listed in markers"):
        PatchCalibrationSpec(
            patch=TestPatchId.SURFACE,
            markers=(TestMarkerId.A, TestMarkerId.B, TestMarkerId.C),
            marker_translations={
                TestMarkerId.D: 0.1 * SemanticAxis.UP,
            },
            region=RectangularRegion(width=1.0, height=1.0),
        )


def test_patch_calibration_rejects_too_few_markers() -> None:
    with pytest.raises(ValueError, match="at least three markers"):
        PatchCalibrationSpec(
            patch=TestPatchId.SURFACE,
            markers=(TestMarkerId.A, TestMarkerId.B),
            region=RectangularRegion(width=1.0, height=1.0),
        )


def test_patch_calibration_rejects_duplicate_markers() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        PatchCalibrationSpec(
            patch=TestPatchId.SURFACE,
            markers=(TestMarkerId.A, TestMarkerId.A, TestMarkerId.C),
            region=RectangularRegion(width=1.0, height=1.0),
        )
