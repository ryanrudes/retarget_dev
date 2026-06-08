from __future__ import annotations

from typing import cast

import numpy as np

from retarget.core.types import Vec2, Vec3, Mat3, Points2, Points3
from retarget.core.transform import RigidTransform


def point_in_polygon(
    point: Vec2,
    vertices: Points2,
) -> bool:
    """Check if a point is in a polygon.

    Args:
        point: The point to check.
        vertices: The vertices of the polygon.

    Returns:
        True if the point is in the polygon, False otherwise.
    """
    x, y = point
    inside = False

    n = vertices.shape[0]

    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        crosses_y = (y1 > y) != (y2 > y)

        if crosses_y:
            x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x <= x_intersect:
                inside = not inside

    return inside


def fit_patch_frame(
    surface_points_segment: Points3,
    *,
    outward_hint_segment: Vec3 | None = None,
    x_axis_hint_segment: Vec3 | None = None,
) -> RigidTransform:
    """
    Fit a patch frame from surface points in the segment frame.

    Patch convention:
        +z is the outward normal.
        x and y span the contact plane.
        origin is the centroid.
    """
    if surface_points_segment.shape[0] < 3:
        raise ValueError("Need at least three points to fit a patch frame")

    centroid = cast(Vec3, surface_points_segment.mean(axis=0))
    centered = surface_points_segment - centroid

    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("Surface points must not be collinear")

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    if outward_hint_segment is not None:
        if np.dot(normal, outward_hint_segment) < 0:
            normal = -normal

    if x_axis_hint_segment is not None:
        x_axis = x_axis_hint_segment - np.dot(x_axis_hint_segment, normal) * normal

        norm = np.linalg.norm(x_axis)
        if norm < 1e-9:
            raise ValueError("x_axis_hint_segment is parallel to fitted normal")

        x_axis = x_axis / norm
    else:
        x_axis = vh[0]
        x_axis = x_axis - np.dot(x_axis, normal) * normal
        x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, normal)
    x_axis = x_axis / np.linalg.norm(x_axis)

    rotation_segment_patch = cast(Mat3, np.column_stack([x_axis, y_axis, normal]))

    return RigidTransform.from_rotation_translation(
        rotation=rotation_segment_patch,
        translation=centroid,
    )