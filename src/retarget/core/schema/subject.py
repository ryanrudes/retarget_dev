"""Subject schema declaration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from retarget.core.schema.segment import Segment, Segments
from retarget.core.types import Vec3


class Subjects(TypedDict):
    """Base class for typed subject schema declarations."""


@dataclass(slots=True, eq=False)
class _SubjectBinding:
    subject: str


@dataclass(frozen=True, slots=True)
class Subject[SegmentsT: Segments]:
    """A subject: a typed mapping of named segments.

    ``body_model`` is an optional subject-level source of segment-frame marker
    rest positions keyed by ``Marker.mocap_name`` (e.g. from a VSK). At bind
    time, any marker without an explicit ``position_segment`` inherits its
    segment-frame position from this mapping, so callers need not repeat
    ``position_segment`` on every marker. A per-marker ``position_segment``
    always overrides the body model.
    """

    segments: SegmentsT
    mocap_name: str | None = None
    body_model: Mapping[str, Vec3] | None = field(default=None, compare=False)
    _binding: _SubjectBinding | None = field(
        default=None, init=False, compare=False, repr=False
    )

    def segment_external_name(self, segment: str) -> str:
        """External (Vicon) name used by IO for one of this subject's segments."""
        seg = cast(Mapping[str, Segment[Any, Any]], self.segments)[segment]
        return seg.mocap_name or segment

    @property
    def external_name(self) -> str | None:
        """External (Vicon) subject name, if any."""
        return self.mocap_name
