"""Backend helper to calibrate a patch frame from segment-local marker positions.

This produces a ``transform_segment_patch`` suitable for authoring a calibrated
:class:`~retarget.core.schema.Patch`. It is the low-level primitive that
``Patch.planar(plane=...)`` calls at bind time; backend loaders may also call it
directly when they already hold a marker-position mapping.

Each independent aspect of the definition is one pluggable resolver (see
:mod:`retarget.core.resolvers`): the ``plane`` source, the ``normal`` resolver
(which side is outward + the contact-surface offset it carries), the ``tangential``
orientation (in-plane +x), the ``origin`` (in-plane placement), and the ``extent``
(rectangle size).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.contact_region import RectangularRegion
from retarget.core.resolvers import (
    FittedPlane,
    ExtentResolver,
    NormalResolver,
    OriginResolver,
    PlaneResolver,
    TangentialResolver,
    along_axis,
    axis_normal,
    fixed,
)
from retarget.core.transform import RigidTransform
from retarget.core.translation import AxisResolvable
from retarget.core.types import Mat3, Vec3
from retarget.utils.geometry import fit_patch_frame


class _AxisResolver(AxisResolvable):
    def __init__(self, convention: AxisConvention) -> None:
        self._convention = convention

    def axis(self, axis: SemanticAxis) -> Vec3:
        return self._convention.vector(axis)


def _orient_to_outward(rotation: Mat3, outward_segment: Vec3) -> Mat3:
    """Sign-flip the fitted normal to agree with ``outward_segment``, re-orthonormalized."""
    rot = np.asarray(rotation, dtype=np.float64)
    normal = rot[:, 2].copy()
    if float(np.dot(normal, np.asarray(outward_segment, dtype=np.float64))) < 0.0:
        normal = -normal
    x = rot[:, 0]
    x = x - np.dot(x, normal) * normal
    x = x / np.linalg.norm(x)
    y = np.cross(normal, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, normal)
    x = x / np.linalg.norm(x)
    return cast(Mat3, np.column_stack([x, y, normal]))


def _apply_tangential(
    tangential: TangentialResolver,
    rotation: Mat3,
    reference: Vec3,
    marker_positions_segment: Mapping[str, Vec3],
    axis_convention: AxisConvention,
) -> Mat3:
    """Rebuild the patch rotation with an in-plane +x from ``tangential`` (normal kept)."""
    normal = np.asarray(rotation, dtype=np.float64)[:, 2]
    desired = np.asarray(
        tangential.tangential_x(
            normal=cast(Vec3, normal),
            reference=reference,
            marker_positions_segment=marker_positions_segment,
            axis_convention=axis_convention,
        ),
        dtype=np.float64,
    )
    x = desired - np.dot(desired, normal) * normal
    norm = float(np.linalg.norm(x))
    if norm < 1e-9:
        raise ValueError("the tangential resolver produced an axis parallel to the patch normal")
    x = x / norm
    y = np.cross(normal, x)
    y = y / np.linalg.norm(y)
    x = np.cross(y, normal)
    x = x / np.linalg.norm(x)
    return cast(Mat3, np.column_stack([x, y, normal]))


def calibrate_patch_transform(
    *,
    marker_positions_segment: Mapping[str, Vec3],
    plane: PlaneResolver,
    extent: ExtentResolver = fixed(1.0, 1.0),
    normal: NormalResolver = axis_normal(),
    tangential: TangentialResolver = along_axis(SemanticAxis.FORWARD),
    origin: OriginResolver | None = None,
    axis_convention: AxisConvention = Z_UP_AXES,
) -> tuple[RigidTransform, RectangularRegion]:
    """Fit a segment->patch transform + rectangle from calibration markers.

    Every aspect is one resolver:

    1. ``plane`` supplies the (optionally pre-translated) segment-frame points;
    2. fit the patch plane through them;
    3. ``normal`` fixes the +normal (outward) sign and contributes the contact-surface
       ``offset`` applied along it (+ = outward);
    4. ``tangential`` sets the in-plane +x (default: ``along_axis(FORWARD)``);
    5. ``origin`` places the in-plane origin (default: ``extent``'s default / the
       plane-marker centroid), then the offset is applied along the oriented normal;
    6. ``extent`` sizes the rectangle (default: a degenerate 1x1 -- pass a real
       ``extent`` such as :func:`~retarget.core.resolvers.bounding_box`).
    """
    resolver = _AxisResolver(axis_convention)
    surface_points = plane.surface_points(marker_positions_segment, resolver)

    fitted = fit_patch_frame(surface_points, outward_hint_segment=None, x_axis_hint_segment=None)
    rotation = np.asarray(fitted.rotation, dtype=np.float64)
    reference = np.asarray(fitted.translation, dtype=np.float64)

    resolution = normal.resolve(
        plane_normal_segment=cast(Vec3, rotation[:, 2]),
        plane_reference=cast(Vec3, reference),
        marker_positions_segment=marker_positions_segment,
        axis_convention=axis_convention,
    )
    rotation = np.asarray(_orient_to_outward(cast(Mat3, rotation), resolution.outward_segment), dtype=np.float64)
    rotation = np.asarray(
        _apply_tangential(
            tangential, cast(Mat3, rotation), cast(Vec3, reference), marker_positions_segment, axis_convention
        ),
        dtype=np.float64,
    )

    plane_ctx = FittedPlane(
        rotation=cast(Mat3, rotation),
        reference=cast(Vec3, reference),
        marker_positions=marker_positions_segment,
    )

    if origin is None:
        origin = extent.default_origin()  # e.g. bounding_box -> its bbox center
    if origin is None:
        offset_x, offset_y = 0.0, 0.0
    else:
        offset_x, offset_y = (float(v) for v in origin.locate(plane_ctx))
    translation = (
        reference
        + rotation[:, 0] * offset_x
        + rotation[:, 1] * offset_y
        + rotation[:, 2] * resolution.offset
    )
    transform = RigidTransform.from_rotation_translation(
        rotation=cast(Mat3, rotation), translation=cast(Vec3, translation)
    )

    region = extent.fit(plane_ctx)
    return transform, region
