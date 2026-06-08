from __future__ import annotations

from dataclasses import dataclass

from retarget.core.enums import SegmentId, SubjectId


@dataclass(frozen=True, slots=True)
class SegmentKey:
    """Globally unique runtime identity for one segment instance in a scene."""

    subject: SubjectId
    segment: SegmentId
