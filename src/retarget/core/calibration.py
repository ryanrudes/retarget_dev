"""Backend helper to calibrate a patch frame from segment-local marker positions.

This produces a ``transform_segment_patch`` suitable for authoring a calibrated
:class:`~retarget.core.schema.Patch`. It is the low-level primitive that
``Patch.rectangle(markers=...)`` calls at bind time; backend loaders may also
call it directly when they already hold a marker-position mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.contact_region import RectangularRegion
from retarget.core.patch_frame import FittedPlane, PatchExtent, PatchOrigin, fixed
from retarget.core.transform import RigidTransform
from retarget.core.translation import AxisResolvable, MarkerTranslation
from retarget.core.types import Vec3
from retarget.utils.geometry import fit_patch_frame


class _AxisResolver(AxisResolvable):
    def __init__(self, convention: AxisConvention) -> None:
        self._convention = convention

    def axis(self, axis: SemanticAxis) -> Vec3:
        return self._convention.vector(axis)


def calibrate_patch_transform(
    *,
    marker_positions_segment: Mapping[str, Vec3],
    markers: Sequence[str],
    axis_convention: AxisConvention = Z_UP_AXES,
    marker_translations: Mapping[str, MarkerTranslation] | None = None,
    normal_offset: float = 0.0,
    outward_axis: SemanticAxis = SemanticAxis.UP,
    forward_axis: SemanticAxis = SemanticAxis.FORWARD,
    origin: PatchOrigin | None = None,
    extent: PatchExtent | None = None,
) -> tuple[RigidTransform, RectangularRegion]:
    """Fit a segment->patch transform + rectangle from calibration markers.

    1. take the listed plane ``markers``' segment-frame positions;
    2. apply optional sparse ``marker_translations`` before fitting;
    3. fit the patch plane (normal + in-plane axes);
    4. place the in-plane origin via ``origin`` (default: the plane-marker centroid),
       then apply ``normal_offset`` along the fitted normal;
    5. size the rectangle via ``extent`` (default: a degenerate 1x1 -- pass a real
       ``extent`` such as :func:`~retarget.core.patch_frame.bounding_box`).
    """
    if len(markers) < 3:
        raise ValueError("Patch calibration requires at least three markers")
    resolver = _AxisResolver(axis_convention)
    translations = dict(marker_translations or {})

    points: list[np.ndarray] = []
    for marker in markers:
        position = np.asarray(marker_positions_segment[marker], dtype=np.float64)
        translation = translations.get(marker)
        if translation is None:
            points.append(position)
        else:
            points.append(position + translation.resolve(resolver))
    surface_points = np.stack(points)

    fitted = fit_patch_frame(
        surface_points,
        outward_hint_segment=axis_convention.vector(outward_axis),
        x_axis_hint_segment=axis_convention.vector(forward_axis),
    )
    rotation = fitted.rotation
    reference = fitted.translation
    plane = FittedPlane(
        rotation=rotation, reference=reference, marker_positions=marker_positions_segment
    )

    if origin is None and extent is not None:
        origin = extent.default_origin()  # e.g. bounding_box -> its bbox center
    if origin is None:
        offset_x, offset_y = 0.0, 0.0
    else:
        offset_x, offset_y = (float(v) for v in origin.locate(plane))
    translation = (
        reference
        + rotation[:, 0] * offset_x
        + rotation[:, 1] * offset_y
        + rotation[:, 2] * normal_offset
    )
    transform = RigidTransform.from_rotation_translation(
        rotation=rotation, translation=translation
    )

    region = (extent if extent is not None else fixed(1.0, 1.0)).fit(plane)
    return transform, region
