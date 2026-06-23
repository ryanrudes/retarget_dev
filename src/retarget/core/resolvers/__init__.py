"""Pluggable resolvers for each aspect of a planar patch's definition.

Every geometric aspect of a patch is one composable resolver (an ABC + frozen
implementations + a lowercase factory), so ``Patch.planar(...)`` is a uniform
composition rather than a pile of coupled keyword arguments:

* :mod:`~retarget.core.resolvers.plane` -- which markers define the contact plane;
* :mod:`~retarget.core.resolvers.normal` -- which side is outward (+ contact offset);
* :mod:`~retarget.core.resolvers.tangential` -- the in-plane +x axis;
* :mod:`~retarget.core.resolvers.origin` -- the in-plane origin;
* :mod:`~retarget.core.resolvers.extent` -- the rectangle size.

:class:`~retarget.core.resolvers.fitted_plane.FittedPlane` is the resolved-plane
context handed to the origin/extent resolvers.
"""

from __future__ import annotations

from retarget.core.resolvers.extent import (
    BoundingBox,
    Fixed,
    ExtentResolver,
    bounding_box,
    fixed,
)
from retarget.core.resolvers.fitted_plane import FittedPlane
from retarget.core.resolvers.normal import (
    Chirality,
    NormalResolution,
    AxisNormal,
    NormalResolver,
    SideNormal,
    WindingNormal,
    WindingDirection,
    axis_normal,
    side,
    winding,
)
from retarget.core.resolvers.origin import (
    AtMarker,
    BoundingBoxCenter,
    Centroid,
    OriginResolver,
    at_marker,
    bounding_box_center,
    centroid,
)
from retarget.core.resolvers.plane import PlaneFromMarkers, PlaneResolver, plane_from
from retarget.core.resolvers.tangential import (
    AlongAxis,
    MinAreaRectangle,
    TangentialResolver,
    TowardMarker,
    along_axis,
    min_area_rectangle,
    toward,
)

__all__ = [
    # shared context
    "FittedPlane",
    # plane
    "PlaneResolver",
    "PlaneFromMarkers",
    "plane_from",
    # normal
    "NormalResolver",
    "NormalResolution",
    "AxisNormal",
    "SideNormal",
    "WindingNormal",
    "WindingDirection",
    "Chirality",
    "axis_normal",
    "side",
    "winding",
    # tangential
    "TangentialResolver",
    "AlongAxis",
    "TowardMarker",
    "MinAreaRectangle",
    "along_axis",
    "toward",
    "min_area_rectangle",
    # origin
    "OriginResolver",
    "BoundingBoxCenter",
    "Centroid",
    "AtMarker",
    "bounding_box_center",
    "centroid",
    "at_marker",
    # extent
    "ExtentResolver",
    "BoundingBox",
    "Fixed",
    "bounding_box",
    "fixed",
]
