"""Tests for support-surface models and fitters."""

from __future__ import annotations

import numpy as np

from retarget.contacts.supports import (
    HeightmapSupportModel,
    PlaneSupportModel,
    SupportModel,
    SupportModelType,
    SupportParams,
    TimeIndexedSupport,
    fit_best_support_surface,
    ground_plane,
    infer_support_surface,
    plane_support,
)


def _planar_cloud(height: float = 0.2, noise: float = 0.002, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = np.full(n, height) + rng.normal(0.0, noise, size=n)
    return np.column_stack([xy, z])


def test_plane_support_fit_recovers_up_normal() -> None:
    cloud = _planar_cloud(height=0.2)
    plane = PlaneSupportModel.fit(cloud, SupportParams())
    np.testing.assert_allclose(np.abs(plane.normal), [0.0, 0.0, 1.0], atol=1e-2)
    # Clearance is near zero on the fitted plane.
    assert float(np.mean(np.abs(plane.clearance(cloud)))) < 0.01
    # A point 0.1 above the plane has clearance ~0.1.
    above = np.array([[0.0, 0.0, 0.3]])
    assert abs(float(plane.clearance(above)[0]) - 0.1) < 0.02


def test_plane_support_satisfies_protocol() -> None:
    plane = ground_plane(0.0)
    assert isinstance(plane, SupportModel)


def test_ground_plane_clearance_is_height_above_floor() -> None:
    floor = ground_plane(0.5)
    points = np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 0.9], [3.0, -1.0, 0.2]])
    np.testing.assert_allclose(floor.clearance(points), [0.0, 0.4, -0.3], atol=1e-9)
    normals = floor.normals_at(points)
    np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (3, 1)))


def test_plane_support_normalizes_normal() -> None:
    plane = plane_support(normal=np.array([0.0, 0.0, 5.0]), point=np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0])
    assert abs(float(plane.clearance(np.array([[0.0, 0.0, 2.0]]))[0]) - 1.0) < 1e-9


def test_fit_best_support_surface_prefers_plane_when_coplanar() -> None:
    cloud = _planar_cloud(height=0.1)
    model = fit_best_support_surface(cloud, SupportParams())
    assert isinstance(model, PlaneSupportModel)


def test_fit_best_support_surface_falls_back_to_heightmap_when_stepped() -> None:
    rng = np.random.default_rng(1)
    low = np.column_stack([rng.uniform(-1.0, -0.5, size=(120, 2)), np.full(120, 0.0)])
    high = np.column_stack([rng.uniform(0.5, 1.0, size=(120, 2)), np.full(120, 0.3)])
    cloud = np.vstack([low, high])
    model = fit_best_support_surface(cloud, SupportParams())
    assert isinstance(model, HeightmapSupportModel)


def test_explicit_model_type_heightmap() -> None:
    cloud = _planar_cloud(height=0.0)
    model = fit_best_support_surface(cloud, SupportParams(model_type=SupportModelType.HEIGHTMAP))
    assert isinstance(model, HeightmapSupportModel)


def test_infer_support_surface_prefers_quiet_points() -> None:
    # Quiet points sit near z=0; noisy "moving" points are high and would bias a naive fit.
    quiet = _planar_cloud(height=0.0, n=120)
    moving = np.column_stack([np.zeros((40, 2)), np.full(40, 1.0)])
    cloud = np.vstack([quiet, moving])
    mask = np.concatenate([np.ones(120, dtype=bool), np.zeros(40, dtype=bool)])
    model = infer_support_surface(cloud, mask, SupportParams())
    # The fitted plane sits near the quiet height, not pulled up by the movers.
    assert abs(float(model.clearance(np.array([[0.0, 0.0, 0.0]]))[0])) < 0.05


def test_time_indexed_support_at_returns_per_frame_plane() -> None:
    origins = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    support = TimeIndexedSupport(origins=origins, normals=normals)
    assert len(support) == 2
    plane0 = support.at(0)
    plane1 = support.at(1)
    assert abs(float(plane0.clearance(np.array([[0.0, 0.0, 0.2]]))[0]) - 0.2) < 1e-9
    assert abs(float(plane1.clearance(np.array([[0.0, 0.0, 0.2]]))[0]) - (-0.3)) < 1e-9
