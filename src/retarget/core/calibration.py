"""Backend helper to calibrate a patch frame from segment-local marker positions.

This produces a ``transform_segment_patch`` suitable for authoring a calibrated
:class:`~retarget.core.schema.Patch`. It is a low-level backend utility used by
loaders that derive patch geometry from VSK/measured marker positions; ordinary
typed authoring sets ``Patch.rectangular(transform_segment_patch=...)`` directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
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
    x_axis: SemanticAxis = SemanticAxis.FORWARD,
) -> RigidTransform:
    """Fit a segment->patch transform from calibration markers.

    1. take the listed markers' segment-frame positions;
    2. apply optional sparse ``marker_translations`` before fitting;
    3. fit the patch frame;
    4. apply ``normal_offset`` after fitting, along the fitted normal.
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

    transform = fit_patch_frame(
        surface_points,
        outward_hint_segment=axis_convention.vector(outward_axis),
        x_axis_hint_segment=axis_convention.vector(x_axis),
    )
    if normal_offset != 0.0:
        normal = transform.rotation[:, 2]
        transform = RigidTransform.from_rotation_translation(
            rotation=transform.rotation,
            translation=transform.translation + normal_offset * normal,
        )
    return transform
