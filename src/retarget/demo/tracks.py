"""Generic track and time primitives for the demonstration layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class TimeRange:
    """Half-open time interval ``[start, stop)``, in seconds."""

    start: float
    stop: float

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError("TimeRange stop must be >= start")

    def contains(self, time: float) -> bool:
        return self.start <= time < self.stop


class TimeTrack(Protocol[T]):
    """Minimal protocol for time-indexed demonstration tracks."""

    @property
    def timestamps(self) -> np.ndarray: ...

    def __len__(self) -> int: ...

    def slice_time(self, start: float, stop: float) -> T: ...

    def nearest_index(self, time: float) -> int: ...
