from __future__ import annotations

from dataclasses import dataclass

from retarget.core.enums import MarkerId, PatchId, SegmentId, SubjectId
from retarget.core.handles import MarkerHandle, PatchHandle
from retarget.core.keys import SegmentKey


@dataclass(frozen=True, slots=True)
class SegmentTarget:
    """Scene-level reference to a concrete segment on a concrete subject."""

    subject: SubjectId
    segment: SegmentId

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.segment)

    @property
    def label(self) -> str:
        return self.segment.label

    @property
    def index(self) -> int:
        return self.segment.index


@dataclass(frozen=True, slots=True)
class MarkerTarget[M: MarkerId]:
    """Scene-level reference to a concrete marker on a concrete subject."""

    subject: SubjectId
    handle: MarkerHandle[M]

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.handle.segment)

    @property
    def label(self) -> str:
        return self.handle.label

    @property
    def index(self) -> int:
        return self.handle.index


@dataclass(frozen=True, slots=True)
class PatchTarget[P: PatchId]:
    """Scene-level reference to a concrete patch on a concrete subject."""

    subject: SubjectId
    handle: PatchHandle[P]

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.handle.segment)
