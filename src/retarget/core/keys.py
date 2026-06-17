from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentKey:
    """Globally unique runtime identity for one segment instance in a scene.

    ``subject`` and ``segment`` are the authored (compiled) string names.
    """

    subject: str
    segment: str
