"""Marker schema declaration and bound time-series queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from fungeom import Point3

from retarget.core.formats import finite_difference_velocity, speed_from_velocity
from retarget.core.schema.base import _Schema
from retarget.core.targets import MarkerTarget
from retarget.core.types import TimeVec3, Vec3

if TYPE_CHECKING:
    from retarget.core.schema.segment import _SegmentRuntime


@dataclass(frozen=True, slots=True)
class Markers(_Schema):
    """Base class for typed marker schema declarations."""


@dataclass(slots=True, eq=False)
class _MarkerBinding:
    subject: str
    segment: str
    marker: str
    runtime: _SegmentRuntime | None


@dataclass(frozen=True, slots=True)
class Marker:
    """A marker: authoring metadata plus bound time-series queries."""

    mocap_name: str
    position_segment: Vec3 | None = field(default=None, compare=False)
    _binding: _MarkerBinding | None = field(default=None, init=False, compare=False, repr=False)

    @property
    def rest(self) -> Point3:
        """This marker's segment-frame rest position as a fungeom **free variable**.

        Reference it to author patch geometry as data over typed marker symbols -- no string keys,
        no callable::

            cloud = Point3Bundle.of([heel.rest, toe.rest, mid.rest])
            Patch(label="sole", geometry=Face.on(cloud.fit_plane(), Region2.hull(...)))

        The free variable is identified by the marker itself, so a misspelled symbol is a
        ``NameError`` (statically a mypy error) rather than a silent string. At bind time it
        resolves to the marker's ``position_segment`` (its segment-frame rest position).
        """
        return Point3.free(self)

    def positions(self, *, modeled: bool = False) -> TimeVec3:
        """World-frame marker positions with shape ``(T, 3)``.

        ``modeled=False`` returns observed positions (NaN where unobserved);
        ``modeled=True`` rigidly transforms the segment-frame position.
        """
        runtime = self._runtime()
        if modeled:
            if self.position_segment is None:
                raise ValueError(
                    f"Marker {self._binding.marker!r} has no segment-frame "  # type: ignore[union-attr]
                    "position for modeled queries"
                )
            local = np.asarray(self.position_segment, dtype=np.float64)
            world = np.einsum("tij,j->ti", runtime.rotations, local)
            return cast(TimeVec3, world + runtime.translations)
        try:
            return cast(TimeVec3, runtime.observed_markers[self.mocap_name])
        except KeyError as exc:
            raise ValueError("Observed marker positions require marker_frames on the mocap track") from exc

    def velocities(self, *, modeled: bool = False) -> TimeVec3:
        """World-frame marker velocities with shape ``(T, 3)``."""
        runtime = self._runtime()
        return cast(
            TimeVec3,
            finite_difference_velocity(self.positions(modeled=modeled), runtime.timestamps),
        )

    def speed(self, *, modeled: bool = False) -> np.ndarray:
        """World-frame marker speed with shape ``(T,)``."""
        return speed_from_velocity(self.velocities(modeled=modeled))

    @property
    def target(self) -> MarkerTarget:
        """Stable scene-level identity for this marker."""
        binding = self._require_binding()
        return MarkerTarget(
            subject=binding.subject,
            segment=binding.segment,
            marker=binding.marker,
        )

    def _require_binding(self) -> _MarkerBinding:
        if self._binding is None:
            from retarget.core.schema.segment import _UNBOUND_MESSAGE

            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Marker"))
        return self._binding

    def _runtime(self) -> _SegmentRuntime:
        from retarget.core.schema.segment import _UNBOUND_MESSAGE

        binding = self._require_binding()
        if binding.runtime is None:
            raise RuntimeError(_UNBOUND_MESSAGE.format(what="Marker"))
        return binding.runtime
