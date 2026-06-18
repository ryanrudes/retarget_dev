"""Support surfaces a patch can be "in contact" with.

A support is anything exposing :meth:`clearance` (signed distance above the
surface) and :meth:`normals_at` (unit outward normal). Built-ins:

* :class:`PlaneSupportModel` -- a fixed plane (RANSAC + SVD fit, or built from a
  known height/normal via :func:`ground_plane` / :func:`plane_support`);
* :class:`HeightmapSupportModel` / :class:`LocalPercentileHeightmap` -- gridded /
  local-percentile height fields for uneven terrain (power-user);
* :class:`TimeIndexedSupport` -- a moving support whose plane is recomputed per
  frame (e.g. a tracked skateboard deck); the driver evaluates it frame-by-frame.

The math is ported from the reference ``contact_detection`` pipeline; the plane
SVD refinement reuses :func:`retarget.utils.geometry.fit_patch_frame`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from scipy.spatial import cKDTree

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.types import FloatArray, FloatArray1D, Points3, TimeVec3, Vec3
from retarget.utils.geometry import fit_patch_frame


class SupportModelType(StrEnum):
    """Support-surface geometry selected when fitting from candidate points."""

    AUTO = "auto"
    """Pick plane vs heightmap from the data."""
    PLANE = "plane"
    """Single global plane."""
    HEIGHTMAP = "heightmap"
    """Gridded heightmap over the capture volume."""
    LOCAL_HEIGHTMAP = "local_heightmap"
    """Local percentile heightmap around each query."""


@dataclass(frozen=True, slots=True)
class SupportParams:
    """Parameters for fitting a support surface from candidate points."""

    model_type: SupportModelType = SupportModelType.AUTO
    plane_residual_tolerance: float = 0.025
    coplanarity_ratio_threshold: float = 0.85
    heightmap_cell_size: float = 0.10
    heightmap_quantile: float = 0.20
    local_heightmap_radius: float = 0.25
    local_heightmap_quantile: float = 0.20
    local_heightmap_min_neighbors: int = 3
    ransac_iterations: int = 128
    random_seed: int = 17
    up_axis: int = 2
    min_support_points: int = 8


@runtime_checkable
class SupportModel(Protocol):
    """Geometry exposing clearance and surface normals for query points.

    ``name`` is a short label used as the categorical support name in
    :class:`~retarget.contacts.state.SupportStateTrack`.
    """

    @property
    def name(self) -> str:
        """Short categorical support label (e.g. ``"ground"``, ``"skateboard"``)."""
        ...

    def clearance(self, points: Points3) -> FloatArray1D:
        """Signed distance above the surface (positive = above), shape ``(M,)``."""
        ...

    def normals_at(self, points: Points3) -> Points3:
        """Unit outward normal at each query point, shape ``(M, 3)``."""
        ...


def _up_vector(up_axis: int) -> Vec3:
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    v = np.zeros(3, dtype=np.float64)
    v[up_axis] = 1.0
    return cast(Vec3, v)


def _orient_normal_up(normal: np.ndarray, up_axis: int) -> Vec3:
    arr = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("normal must be non-zero")
    arr = arr / norm
    if arr[up_axis] < 0.0:
        arr = -arr
    return cast(Vec3, arr)


def _horizontal_axes(up_axis: int) -> list[int]:
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    return [axis for axis in range(3) if axis != up_axis]


def _validate_point_cloud(points: Points3) -> Points3:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (M, 3)")
    return cast(Points3, arr)


@dataclass(frozen=True, slots=True)
class PlaneSupportModel:
    """Planar support surface defined by a unit normal and a point on the plane."""

    normal: Vec3
    origin: Vec3
    name: str = "plane"

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=np.float64).reshape(3))
        normal = np.asarray(self.normal, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("plane normal must be non-zero")
        object.__setattr__(self, "normal", normal / norm)

    def clearance(self, points: Points3) -> FloatArray1D:
        arr = np.asarray(points, dtype=np.float64)
        return cast(FloatArray1D, (arr - self.origin) @ self.normal)

    def normals_at(self, points: Points3) -> Points3:
        arr = np.asarray(points, dtype=np.float64)
        return cast(Points3, np.broadcast_to(self.normal, arr.shape).copy())

    @classmethod
    def fit(cls, points: Points3, params: SupportParams) -> PlaneSupportModel:
        """Fit a plane with RANSAC, then refine on inliers via SVD."""
        cloud = _validate_point_cloud(points)
        if len(cloud) < 3:
            raise ValueError("need at least 3 points to fit a plane")

        rng = np.random.default_rng(params.random_seed)
        best_inliers: FloatArray | None = None
        best_count = -1
        for _ in range(params.ransac_iterations):
            sample_idx = rng.choice(len(cloud), size=3, replace=False)
            p0, p1, p2 = cloud[sample_idx]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = float(np.linalg.norm(normal))
            if norm <= 1e-12:
                continue
            normal = normal / norm
            residuals = np.abs((cloud - p0[None, :]) @ normal)
            inliers = residuals <= params.plane_residual_tolerance
            count = int(np.sum(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or int(np.sum(best_inliers)) < 3:
            inlier_cloud = cloud
        else:
            inlier_cloud = cloud[best_inliers]

        origin, normal_vec = _refine_plane(inlier_cloud, params.up_axis)
        return cls(normal=normal_vec, origin=origin)


def _refine_plane(points: Points3, up_axis: int) -> tuple[Vec3, Vec3]:
    """Return ``(origin, unit_normal)`` for a plane fit to ``points`` (normal up-oriented)."""
    up = _up_vector(up_axis)
    try:
        frame = fit_patch_frame(points, outward_hint_segment=up)
    except ValueError:
        origin = cast(Vec3, np.asarray(points, dtype=np.float64).mean(axis=0))
        return origin, up
    origin = frame.translation
    normal = _orient_normal_up(frame.rotation[:, 2], up_axis)
    return origin, normal


@dataclass
class HeightmapSupportModel:
    """Piecewise height field on a regular horizontal grid (power-user)."""

    cell_size: float
    cells_xy: Points3
    heights: FloatArray1D
    up_axis: int = 2
    name: str = "heightmap"
    _tree: Any = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def fit(cls, points: Points3, params: SupportParams) -> HeightmapSupportModel:
        cloud = _validate_point_cloud(points)
        if len(cloud) < 1:
            raise ValueError("need at least 1 point to fit a heightmap")
        cell_size = float(params.heightmap_cell_size)
        if cell_size <= 0.0:
            raise ValueError("heightmap_cell_size must be positive")
        horizontal = _horizontal_axes(params.up_axis)
        xy = cloud[:, horizontal]
        z = cloud[:, params.up_axis]
        cell_ids = np.floor(xy / cell_size).astype(int)
        buckets: dict[tuple[int, int], list[float]] = {}
        for cell_id, height in zip(cell_ids, z):
            buckets.setdefault((int(cell_id[0]), int(cell_id[1])), []).append(float(height))
        cells: list[FloatArray] = []
        heights: list[float] = []
        for cell_id, values in sorted(buckets.items()):
            center = (np.asarray(cell_id, dtype=np.float64) + 0.5) * cell_size
            cells.append(center)
            heights.append(float(np.quantile(values, params.heightmap_quantile)))
        return cls(
            cell_size=cell_size,
            cells_xy=cast(Points3, np.asarray(cells, dtype=np.float64)),
            heights=cast(FloatArray1D, np.asarray(heights, dtype=np.float64)),
            up_axis=params.up_axis,
        )

    def _height_at_xy(self, xy: FloatArray) -> FloatArray1D:
        if self._tree is None:
            self._tree = cKDTree(self.cells_xy)
        _, idx = self._tree.query(xy, k=1)
        return cast(FloatArray1D, self.heights[idx])

    def clearance(self, points: Points3) -> FloatArray1D:
        arr = np.asarray(points, dtype=np.float64)
        horizontal = _horizontal_axes(self.up_axis)
        support_height = self._height_at_xy(arr[:, horizontal])
        return cast(FloatArray1D, arr[:, self.up_axis] - support_height)

    def normals_at(self, points: Points3) -> Points3:
        arr = np.asarray(points, dtype=np.float64)
        normals = np.zeros_like(arr)
        normals[:, self.up_axis] = 1.0
        return cast(Points3, normals)


@dataclass
class LocalPercentileHeightmap:
    """Neighborhood percentile height field for uneven terrain (power-user)."""

    points: Points3
    radius: float
    quantile: float
    min_neighbors: int
    up_axis: int = 2
    name: str = "local_heightmap"
    _tree: Any = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def fit(cls, points: Points3, params: SupportParams) -> LocalPercentileHeightmap:
        cloud = _validate_point_cloud(points)
        if len(cloud) < 1:
            raise ValueError("need at least 1 point to fit a local heightmap")
        radius = float(params.local_heightmap_radius)
        if radius <= 0.0:
            raise ValueError("local_heightmap_radius must be positive")
        return cls(
            points=cloud.copy(),
            radius=radius,
            quantile=float(params.local_heightmap_quantile),
            min_neighbors=int(params.local_heightmap_min_neighbors),
            up_axis=params.up_axis,
        )

    def _height_at_xy(self, xy: FloatArray) -> FloatArray1D:
        if self._tree is None:
            self._tree = cKDTree(self.points[:, _horizontal_axes(self.up_axis)])
        heights = np.empty(len(xy), dtype=np.float64)
        support_z = self.points[:, self.up_axis]
        k = min(max(self.min_neighbors, 1), len(self.points))
        for i, query in enumerate(xy):
            neighbor_idx = self._tree.query_ball_point(query, r=self.radius)
            if len(neighbor_idx) < self.min_neighbors:
                _, nearest_idx = self._tree.query(query, k=k)
                neighbor_idx = np.atleast_1d(nearest_idx).astype(int).tolist()
            values = support_z[neighbor_idx]
            if values.size == 0:
                _, nearest_idx = self._tree.query(query, k=1)
                values = np.asarray([support_z[int(nearest_idx)]])
            heights[i] = float(np.quantile(values, self.quantile))
        return cast(FloatArray1D, heights)

    def clearance(self, points: Points3) -> FloatArray1D:
        arr = np.asarray(points, dtype=np.float64)
        horizontal = _horizontal_axes(self.up_axis)
        support_height = self._height_at_xy(arr[:, horizontal])
        return cast(FloatArray1D, arr[:, self.up_axis] - support_height)

    def normals_at(self, points: Points3) -> Points3:
        arr = np.asarray(points, dtype=np.float64)
        normals = np.zeros_like(arr)
        normals[:, self.up_axis] = 1.0
        return cast(Points3, normals)


@dataclass(frozen=True, slots=True)
class TimeIndexedSupport:
    """A moving support: a plane recomputed per frame from a tracked body's pose.

    ``origins`` and ``normals`` are world-frame ``(T, 3)`` arrays aligned with the
    scope's timeline. The driver evaluates a query point at frame ``t`` against
    :meth:`at` ``(t)``.
    """

    origins: TimeVec3
    normals: TimeVec3
    name: str = "tracked_plane"

    def __post_init__(self) -> None:
        origins = np.asarray(self.origins, dtype=np.float64)
        normals = np.asarray(self.normals, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins must have shape (T, 3)")
        if normals.shape != origins.shape:
            raise ValueError("normals must have the same shape as origins")
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "normals", normals)

    def __len__(self) -> int:
        return len(self.origins)

    def at(self, timestep: int) -> PlaneSupportModel:
        """Return the per-frame plane support at ``timestep``."""
        return PlaneSupportModel(normal=cast(Vec3, self.normals[timestep]), origin=cast(Vec3, self.origins[timestep]), name=self.name)


def fit_best_support_surface(points: Points3, params: SupportParams = SupportParams()) -> SupportModel:
    """Choose plane vs heightmap support geometry from point coplanarity.

    With ``model_type=AUTO`` a plane is fit first and kept when the fraction of
    near-planar points meets ``coplanarity_ratio_threshold``; otherwise a gridded
    heightmap is used.
    """
    cloud = _validate_point_cloud(points)
    if params.model_type is SupportModelType.PLANE:
        return PlaneSupportModel.fit(cloud, params)
    if params.model_type is SupportModelType.HEIGHTMAP:
        return HeightmapSupportModel.fit(cloud, params)
    if params.model_type is SupportModelType.LOCAL_HEIGHTMAP:
        return LocalPercentileHeightmap.fit(cloud, params)

    plane = PlaneSupportModel.fit(cloud, params)
    residuals = np.abs(plane.clearance(cloud))
    coplanar_ratio = float(np.mean(residuals <= params.plane_residual_tolerance))
    if coplanar_ratio >= params.coplanarity_ratio_threshold:
        return plane
    return HeightmapSupportModel.fit(cloud, params)


def infer_support_surface(
    points: Points3,
    quiet_mask: np.ndarray,
    params: SupportParams = SupportParams(),
) -> SupportModel:
    """Bootstrap a support surface from quiet candidate points.

    Quiet samples are preferred; if too few are available, the lowest points
    along the up axis are used as a fallback so a surface can always be fit.
    """
    cloud = _validate_point_cloud(points)
    mask = np.asarray(quiet_mask, dtype=np.bool_)
    candidates = cloud[mask]
    if len(candidates) < params.min_support_points:
        count = min(max(params.min_support_points, 3), len(cloud))
        order = np.argsort(cloud[:, params.up_axis])
        candidates = cloud[order[:count]]
    return fit_best_support_surface(cast(Points3, candidates), params)


@dataclass(frozen=True, slots=True)
class InferredGround:
    """Marker support meaning "fit the surface from the data" at detection time.

    It is not itself a :class:`SupportModel` (it has no geometry yet); the detector
    recognizes it and fits a real surface from the scope's quiet samples via
    :func:`infer_support_surface`. Use :func:`infer_ground` to construct one.
    """

    name: str = "ground"


def infer_ground() -> InferredGround:
    """A support that is fitted from the motion -- the readable form of "auto floor".

    Pass it where you would name a support, e.g.
    ``against={"ground": infer_ground(), "board": deck_patch}``. Pairs with the
    explicit :func:`ground_plane` (a known height).
    """
    return InferredGround()


def ground_plane(
    height: float = 0.0,
    *,
    up: SemanticAxis = SemanticAxis.UP,
    axis_convention: AxisConvention = Z_UP_AXES,
    name: str = "ground",
) -> PlaneSupportModel:
    """A static floor plane at ``height`` along the convention's up axis."""
    normal = axis_convention.vector(up)
    origin = cast(Vec3, float(height) * normal)
    return PlaneSupportModel(normal=normal, origin=origin, name=name)


def plane_support(*, normal: Vec3, point: Vec3, name: str = "plane") -> PlaneSupportModel:
    """A static plane through ``point`` with the given (outward) ``normal``."""
    return PlaneSupportModel(normal=normal, origin=point, name=name)
