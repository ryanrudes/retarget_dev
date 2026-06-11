from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


from retarget.core.types import FloatArray
from retarget.utils.sampler import (
    estimate_nominal_hz,
    validate_nominal_hz,
)


@dataclass(frozen=True, slots=True)
class TrackSyncInfo:
    """Synchronization metadata for a time-indexed track."""

    can_be_reference: bool = True
    can_be_aligned: bool = True
    nominal_hz: float | None = None


class Track(ABC):
    """Base class for time-indexed demonstration tracks.

    Concrete subclasses should expose an optional `nominal_hz_override`
    attribute. If present and non-None, it is used as the nominal sampling
    rate. Otherwise the rate is estimated from `timestamps`.

    Subclasses may also override `nominal_hz` directly if they need custom
    behavior.
    """

    @property
    @abstractmethod
    def timestamps(self) -> FloatArray:
        """Native timestamps for this track, shape (T,)."""

    @abstractmethod
    def slice_time(self, start: float, stop: float) -> Track:
        """Return a cheap time-sliced view over [start, stop)."""

    @abstractmethod
    def nearest_index(self, time: float) -> int:
        """Return the nearest native timestep index.

        Raises:
            IndexError: If the track is empty.
        """

    @property
    def nominal_hz_override(self) -> float | None:
        """Optional explicit nominal sampling rate.

        Dataclass subclasses can implement this by defining a field named
        `nominal_hz_override`.
        """
        return None

    @property
    def nominal_hz(self) -> float:
        """Nominal sampling rate in Hz.

        Uses `nominal_hz_override` when provided; otherwise estimates from
        timestamps.
        """
        override = validate_nominal_hz(self.nominal_hz_override)
        if override is not None:
            return override

        return estimate_nominal_hz(self.timestamps)

    @property
    def sync_info(self) -> TrackSyncInfo:
        """Synchronization metadata for this track."""
        return TrackSyncInfo(
            can_be_reference=True,
            can_be_aligned=True,
            nominal_hz=self.nominal_hz,
        )


@dataclass(frozen=True, slots=True)
class TrackView[SourceTrack: Track](Track):
    """Base class for cheap time views over a source track.

    A TrackView exposes a time-selected view of an existing source track. It
    should not copy the source data unless a concrete implementation has a
    specific reason to do so.

    Views inherit the resolved nominal sampling rate and synchronization
    eligibility from their source track. This avoids re-estimating frequency
    from short slices, where the selected interval may contain too few samples
    to estimate a rate.
    """

    source: SourceTrack

    @property
    def nominal_hz(self) -> float:
        """Resolved nominal sampling rate inherited from the source track."""
        return self.source.nominal_hz

    @property
    def sync_info(self) -> TrackSyncInfo:
        """Synchronization metadata inherited from the source track."""
        source_info = self.source.sync_info
        return TrackSyncInfo(
            can_be_reference=source_info.can_be_reference,
            can_be_aligned=source_info.can_be_aligned,
            nominal_hz=self.nominal_hz,
        )