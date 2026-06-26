"""Resolve a detection *scope* to the geometry-bearing patches to operate on.

A *scope* is a ``Patch``/``Segment``/``Subject``/``MocapTrack`` or an iterable of
patches. Both the detector and the noise-calibration step walk a scope the same
way, so the logic lives here to avoid an import cycle between them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from retarget.core.schema.patch import Patch
from retarget.core.schema.segment import Segment
from retarget.core.schema.subject import Subject
from retarget.demo.mocap import MocapTrack

Scope = Patch | Segment[Any, Any] | Subject[Any] | MocapTrack[Any] | Iterable[Patch]


def normalize_scope(scope: Scope) -> list[Patch]:
    """Resolve a scope to the geometry-bearing patches to operate on."""
    if isinstance(scope, Patch):
        if not scope.has_geometry():
            raise ValueError("the given patch has no calibrated geometry to detect on")
        return [scope]
    if isinstance(scope, Segment):
        patches: list[Patch] = list(cast(Mapping[str, Patch], scope.patches).values())
    elif isinstance(scope, Subject):
        patches = [
            patch
            for segment in cast(Mapping[str, Segment[Any, Any]], scope.segments).values()
            for patch in cast(Mapping[str, Patch], segment.patches).values()
        ]
    elif isinstance(scope, MocapTrack):
        patches = [
            patch
            for subject in cast(Mapping[str, Subject[Any]], scope.subjects).values()
            for segment in cast(Mapping[str, Segment[Any, Any]], subject.segments).values()
            for patch in cast(Mapping[str, Patch], segment.patches).values()
        ]
    else:
        patches = list(scope)
        for patch in patches:
            if not isinstance(patch, Patch):
                raise TypeError(f"scope iterable must contain Patch objects; got {type(patch).__name__}")
    geometry = [patch for patch in patches if patch.has_geometry()]
    if not geometry:
        raise ValueError("scope contains no geometry-bearing patches to detect on")
    return geometry
