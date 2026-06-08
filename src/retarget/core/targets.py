from __future__ import annotations

from dataclasses import dataclass

from retarget.core.enums import PatchId, SubjectId
from retarget.core.handles import PatchHandle
from retarget.core.keys import SegmentKey


@dataclass(frozen=True, slots=True)
class PatchTarget[P: PatchId]:
    """Scene-level reference to a concrete patch on a concrete subject."""

    subject: SubjectId
    handle: PatchHandle[P]

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.handle.segment)
