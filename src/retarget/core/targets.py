"""Stable, hashable scene-level identity keys.

Targets are the canonical keys for runtime data structures (contact tracks,
marker observation maps, pose maps). They use the authored string names of the
subject/segment/marker/patch they reference, so they remain stable across
slicing, resampling, and serialization without depending on identifier enums.
"""

from __future__ import annotations

from dataclasses import dataclass

from retarget.core.keys import SegmentKey


@dataclass(frozen=True, slots=True)
class SegmentTarget:
    """Scene-level reference to a concrete segment on a concrete subject."""

    subject: str
    segment: str

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.segment)


@dataclass(frozen=True, slots=True)
class MarkerTarget:
    """Scene-level reference to a concrete marker on a concrete subject."""

    subject: str
    segment: str
    marker: str

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.segment)


@dataclass(frozen=True, slots=True)
class PatchTarget:
    """Scene-level reference to a concrete patch on a concrete subject."""

    subject: str
    segment: str
    patch: str

    @property
    def segment_key(self) -> SegmentKey:
        return SegmentKey(self.subject, self.segment)
