from __future__ import annotations

import math

import numpy as np
import pytest

from retarget.core import (
    BodyFrameTranslation,
    Chirality,
    Patch,
    RectangularRegion,
    SemanticAxis,
    WindingDirection,
    along_axis,
    axis_normal,
    bounding_box,
    calibrate_patch_transform,
    fixed,
    min_area_rectangle,
    plane_from,
    side,
    toward,
    winding,
)
from retarget.core.schema.patch import resolve_patch_calibration

_POSITIONS = {
    "a": np.array([0.0, 0.0, 0.0]),
    "b": np.array([1.0, 0.0, 0.0]),
    "c": np.array([0.0, 1.0, 0.0]),
    "d": np.array([0.0, 0.0, 1.0]),  # above the z=0 plane of a/b/c
}


def _fit(**kwargs):
    return calibrate_patch_transform(
        marker_positions_segment=_POSITIONS, plane=plane_from("a", "b", "c"), **kwargs
    )


# -- plane fit --------------------------------------------------------------


def test_calibration_fits_centroid_for_planar_markers() -> None:
    transform, _ = _fit()
    np.testing.assert_allclose(transform.translation, [1.0 / 3.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(abs(transform.rotation[:, 2][2]), 1.0)


def test_plane_from_requires_at_least_three_markers() -> None:
    with pytest.raises(ValueError, match="at least three markers"):
        plane_from("a", "b")


def test_body_frame_translations_applied_before_fitting() -> None:
    transform, _ = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        plane=plane_from("a", "b", "c", translations={"a": BodyFrameTranslation(np.array([0.0, 0.0, 0.3]))}),
    )
    # Lifting marker 'a' by 0.3 raises the plane centroid by 0.1 in z.
    np.testing.assert_allclose(transform.translation[2], 0.1)


def test_semantic_axis_translations_supported() -> None:
    transform, _ = calibrate_patch_transform(
        marker_positions_segment=_POSITIONS,
        plane=plane_from("a", "b", "c", translations={"a": 0.3 * SemanticAxis.UP}),
    )
    np.testing.assert_allclose(transform.translation[2], 0.1)


# -- normal resolvers -------------------------------------------------------


def test_axis_normal_default_points_up() -> None:
    transform, _ = _fit()
    np.testing.assert_allclose(transform.rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)


def test_axis_normal_offset_moves_origin_along_normal() -> None:
    base, _ = _fit()
    offset, _ = _fit(normal=axis_normal(offset=0.1))
    np.testing.assert_allclose(offset.translation, base.translation + 0.1 * base.rotation[:, 2])


def test_defines_must_be_the_up_axis() -> None:
    with pytest.raises(ValueError, match="can only define the UP axis"):
        axis_normal(SemanticAxis.FORWARD)


def test_side_anchor_outward_and_inward() -> None:
    # 'd' is above the plane: outward side -> +z; declared inward -> -z.
    out, _ = _fit(normal=side("d"))
    inw, _ = _fit(normal=side("d", defines=-SemanticAxis.UP))
    np.testing.assert_allclose(out.rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(inw.rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-12)


def test_side_offset_sign_is_intuitive() -> None:
    out, _ = _fit(normal=side("d", offset=0.1))
    inw, _ = _fit(normal=side("d", defines=-SemanticAxis.UP, offset=0.1))
    assert out.translation[2] > 0.0   # +offset along outward (toward 'd', up)
    assert inw.translation[2] < 0.0   # +offset along outward (away from 'd', down)
    np.testing.assert_allclose(out.translation[2], -inw.translation[2])


def test_side_anchor_in_plane_errors() -> None:
    # 'a' is one of the plane markers -> lies in the plane -> cannot pick a side.
    with pytest.raises(ValueError, match="lies in the patch plane"):
        _fit(normal=side("a"))


def test_winding_resolves_normal_sign() -> None:
    # order a,b,c is counterclockwise about +z.
    ccw, _ = _fit(normal=winding(["a", "b", "c"], direction=WindingDirection.COUNTERCLOCKWISE))
    cw, _ = _fit(normal=winding(["a", "b", "c"], direction=WindingDirection.CLOCKWISE))
    np.testing.assert_allclose(ccw.rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(cw.rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-12)


def test_winding_left_handed_flips() -> None:
    rh, _ = _fit(normal=winding(["a", "b", "c"], direction=WindingDirection.COUNTERCLOCKWISE))
    lh, _ = _fit(
        normal=winding(["a", "b", "c"], direction=WindingDirection.COUNTERCLOCKWISE,
                       chirality=Chirality.LEFT_HANDED)
    )
    np.testing.assert_allclose(rh.rotation[:, 2], -lh.rotation[:, 2], atol=1e-12)


def test_winding_requires_three_markers() -> None:
    with pytest.raises(ValueError, match="at least three"):
        winding(["a", "b"], direction=WindingDirection.COUNTERCLOCKWISE)


def test_winding_rejects_self_intersecting_order() -> None:
    pos = {
        "a": np.array([0.0, 0.0, 0.0]),
        "b": np.array([3.0, 0.0, 0.0]),
        "c": np.array([3.0, 1.0, 0.0]),
        "d": np.array([0.0, 2.0, 0.0]),
    }
    # a, c, b, d is a bow-tie ordering of this convex quad -> self-intersecting.
    with pytest.raises(ValueError, match="simple convex loop"):
        calibrate_patch_transform(
            marker_positions_segment=pos, plane=plane_from("a", "b", "c", "d"),
            normal=winding(["a", "c", "b", "d"], direction=WindingDirection.COUNTERCLOCKWISE),
        )


def test_side_and_winding_agree_on_normal() -> None:
    # A convex CCW footprint with an anchor on the +z side -> both pick +z.
    pos = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),
        "c": np.array([-1.0, 0.0, 0.0]),
        "d": np.array([0.0, -1.0, 0.0]),
        "top": np.array([0.0, 0.0, 0.5]),
    }
    by_side, _ = calibrate_patch_transform(
        marker_positions_segment=pos, plane=plane_from("a", "b", "c", "d"), normal=side("top")
    )
    by_wind, _ = calibrate_patch_transform(
        marker_positions_segment=pos, plane=plane_from("a", "b", "c", "d"),
        normal=winding(["a", "b", "c", "d"], direction=WindingDirection.COUNTERCLOCKWISE),
    )
    np.testing.assert_allclose(by_side.rotation[:, 2], by_wind.rotation[:, 2], atol=1e-9)


# -- tangential orientation -------------------------------------------------


def test_min_area_rectangle_orients_and_sizes_to_tight_box() -> None:
    # A 0.40 x 0.20 rectangle rotated 30 deg in the z=0 plane; +x aligns to the long edge.
    a, b, ang = 0.20, 0.10, math.radians(30.0)
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    corners = np.array([[a, b], [-a, b], [-a, -b], [a, -b]]) @ rot.T
    positions = {f"c{i}": np.array([corners[i, 0], corners[i, 1], 0.0]) for i in range(4)}
    names = tuple(positions)

    transform, region = calibrate_patch_transform(
        marker_positions_segment=positions, plane=plane_from(*names),
        tangential=min_area_rectangle(*names), extent=bounding_box(*names),
    )
    np.testing.assert_allclose(transform.rotation[:, 0], [math.cos(ang), math.sin(ang), 0.0], atol=1e-6)
    np.testing.assert_allclose([region.width, region.height], [2 * a, 2 * b], atol=1e-6)

    _, default_region = calibrate_patch_transform(
        marker_positions_segment=positions, plane=plane_from(*names), extent=bounding_box(*names),
    )
    assert region.width * region.height < default_region.width * default_region.height


def test_toward_marker_orients_x_at_marker() -> None:
    positions = {**_POSITIONS, "m": np.array([1.0, 1.0, 0.0])}
    transform, _ = calibrate_patch_transform(
        marker_positions_segment=positions, plane=plane_from("a", "b", "c"),
        tangential=toward("m"),
    )
    # +x points from the plane centroid (1/3, 1/3, 0) toward m -> (1, 1, 0)/sqrt(2).
    np.testing.assert_allclose(transform.rotation[:, 0], [math.sqrt(0.5), math.sqrt(0.5), 0.0], atol=1e-6)


def test_along_axis_matches_default_forward() -> None:
    base, _ = _fit()
    explicit, _ = _fit(tangential=along_axis(SemanticAxis.FORWARD))
    np.testing.assert_allclose(base.rotation, explicit.rotation, atol=1e-12)


# -- builder threading + validation -----------------------------------------


def test_planar_threads_resolvers_through_calibration() -> None:
    positions = {**_POSITIONS, "above": np.array([0.0, 0.0, 1.0])}
    patch = Patch.planar(
        label="p", plane=plane_from("a", "b", "c"), extent=fixed(1.0, 1.0),
        normal=side("above", defines=-SemanticAxis.UP),
    )
    assert patch.calibration is not None
    transform, _ = resolve_patch_calibration(
        patch.calibration, positions, subject="s", segment="seg", patch_name="p"
    )
    # 'above' declared inward -> normal points the other way (down).
    np.testing.assert_allclose(transform.rotation[:, 2], [0.0, 0.0, -1.0], atol=1e-12)


def test_resolve_validates_normal_resolver_markers() -> None:
    patch = Patch.planar(
        label="p", plane=plane_from("a", "b", "c"), extent=fixed(1.0, 1.0),
        normal=side("missing"),
    )
    assert patch.calibration is not None
    with pytest.raises(ValueError, match="missing"):
        resolve_patch_calibration(
            patch.calibration, _POSITIONS, subject="s", segment="seg", patch_name="p"
        )


def test_resolve_validates_tangential_markers() -> None:
    patch = Patch.planar(
        label="p", plane=plane_from("a", "b", "c"), extent=fixed(1.0, 1.0),
        tangential=min_area_rectangle("a", "b", "missing"),
    )
    assert patch.calibration is not None
    with pytest.raises(ValueError, match="missing"):
        resolve_patch_calibration(
            patch.calibration, _POSITIONS, subject="s", segment="seg", patch_name="p"
        )


def test_rectangular_region_contains() -> None:
    region = RectangularRegion(width=1.0, height=2.0)
    assert region.contains(np.array([0.4, 0.9]))
    assert not region.contains(np.array([0.6, 0.0]))
