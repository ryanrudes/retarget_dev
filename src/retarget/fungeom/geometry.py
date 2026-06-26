"""Back-compat re-export.

The segment-geometry view graduated into :mod:`retarget.core.geometry` — it is a **core**
authoring concept (the open patch algebra), evaluated by the binding, not an adapter add-on.
This module re-exports it so ``from retarget.fungeom import SegmentGeometry`` keeps working.
"""

from __future__ import annotations

from retarget.core.geometry import (
    MarkerGeometry,
    SegmentGeometry,
    patch_face,
    segment_geometry,
)

__all__ = [
    "MarkerGeometry",
    "SegmentGeometry",
    "patch_face",
    "segment_geometry",
]
