from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    BodyFrameTranslation,
    MarkerId,
    MarkerSetSpec,
    PatchCalibrationSpec,
    PatchId,
    PatchSpec,
    RectangularRegion,
    SegmentId,
    SegmentSpec,
    SemanticAxis,
    RigidTransform,
    Z_UP_AXES,
)


class _MarkerId(MarkerId):
    A = "a"
    B = "b"
    C = "c"
    D = "d"


class _PatchId(PatchId):
    SURFACE = "surface"


class _SegmentId(SegmentId):
    BODY = "body"


def make_segment(
    calibration: PatchCalibrationSpec[_MarkerId, _PatchId],
) -> SegmentSpec[_MarkerId, _PatchId]:
    return SegmentSpec(
        segment=_SegmentId.BODY,
        marker_type=_MarkerId,
        patch_type=_PatchId,
        axis_convention=Z_UP_AXES,
        marker_set=MarkerSetSpec(marker_type=_MarkerId),
        marker_positions_segment={
            _MarkerId.A: np.array([0.0, 0.0, 0.0]),
            _MarkerId.B: np.array([1.0, 0.0, 0.0]),
            _MarkerId.C: np.array([0.0, 1.0, 0.0]),
            _MarkerId.D: np.array([0.0, 0.0, 1.0]),
        },
        patch_calibrations={
            _PatchId.SURFACE: calibration,
        },
    )


def test_patch_calibration_accepts_markers_without_translations() -> None:
    calibration = PatchCalibrationSpec(
        patch=_PatchId.SURFACE,
        markers=(_MarkerId.A, _MarkerId.B, _MarkerId.C),
        region=RectangularRegion(width=1.0, height=1.0),
    )
    segment = make_segment(calibration).with_built_patches()
    patch = segment.patch_spec(_PatchId.SURFACE)
    np.testing.assert_allclose(
        patch.transform_segment_patch.translation,
        np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
    )


def test_normal_offset_moves_patch_origin_along_fitted_normal() -> None:
    calibration = PatchCalibrationSpec(
        patch=_PatchId.SURFACE,
        markers=(_MarkerId.A, _MarkerId.B, _MarkerId.C),
        normal_offset=0.1,
        region=RectangularRegion(width=1.0, height=1.0),
    )
    segment = make_segment(calibration).with_built_patches()
    patch = segment.patch_spec(_PatchId.SURFACE)
    normal = patch.transform_segment_patch.rotation[:, 2]
    expected = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]) + 0.1 * normal
    np.testing.assert_allclose(
        patch.transform_segment_patch.translation,
        expected,
    )


def test_sparse_marker_translations_are_applied_before_fitting() -> None:
    calibration = PatchCalibrationSpec(
        patch=_PatchId.SURFACE,
        markers=(_MarkerId.A, _MarkerId.B, _MarkerId.C),
        marker_translations={
            _MarkerId.A: 0.1 * SemanticAxis.UP,
        },
        region=RectangularRegion(width=1.0, height=1.0),
    )
    surface_points = calibration.surface_points(
        marker_positions_segment={
            _MarkerId.A: np.array([0.0, 0.0, 0.0]),
            _MarkerId.B: np.array([1.0, 0.0, 0.0]),
            _MarkerId.C: np.array([0.0, 1.0, 0.0]),
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
        patch=_PatchId.SURFACE,
        markers=(_MarkerId.A, _MarkerId.B, _MarkerId.C),
        marker_translations={
            _MarkerId.A: BodyFrameTranslation(
                np.array([0.0, 0.0, -0.012])
            ),
            _MarkerId.B: BodyFrameTranslation(
                np.array([0.0, 0.0, -0.010])
            ),
            _MarkerId.C: BodyFrameTranslation(
                np.array([0.0, 0.0, -0.011])
            ),
        },
        region=RectangularRegion(width=1.0, height=1.0),
    )
    surface_points = calibration.surface_points(
        marker_positions_segment={
            _MarkerId.A: np.array([0.0, 0.0, 0.0]),
            _MarkerId.B: np.array([1.0, 0.0, 0.0]),
            _MarkerId.C: np.array([0.0, 1.0, 0.0]),
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
            patch=_PatchId.SURFACE,
            markers=(_MarkerId.A, _MarkerId.B, _MarkerId.C),
            marker_translations={
                _MarkerId.D: 0.1 * SemanticAxis.UP,
            },
            region=RectangularRegion(width=1.0, height=1.0),
        )


def test_patch_calibration_rejects_too_few_markers() -> None:
    with pytest.raises(ValueError, match="at least three markers"):
        PatchCalibrationSpec(
            patch=_PatchId.SURFACE,
            markers=(_MarkerId.A, _MarkerId.B),
            region=RectangularRegion(width=1.0, height=1.0),
        )


def test_patch_calibration_rejects_duplicate_markers() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        PatchCalibrationSpec(
            patch=_PatchId.SURFACE,
            markers=(_MarkerId.A, _MarkerId.A, _MarkerId.C),
            region=RectangularRegion(width=1.0, height=1.0),
        )


def test_direct_segment_spec_patches_remain_calibrated() -> None:
    segment = SegmentSpec(
        segment=_SegmentId.BODY,
        marker_type=_MarkerId,
        patch_type=_PatchId,
        axis_convention=Z_UP_AXES,
        marker_set=MarkerSetSpec(marker_type=_MarkerId),
        patches={
            _PatchId.SURFACE: PatchSpec(
                patch=_PatchId.SURFACE,
                transform_segment_patch=RigidTransform.identity(),
                region=RectangularRegion(width=1.0, height=1.0),
            ),
        },
    )
    assert segment.patch(_PatchId.SURFACE).patch == _PatchId.SURFACE
    patch = segment.patch_spec(_PatchId.SURFACE)
    np.testing.assert_allclose(
        patch.transform_segment_patch.translation,
        np.zeros(3, dtype=np.float64),
    )
