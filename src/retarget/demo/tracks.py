from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

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


def indices_for_time_range(
    timestamps: FloatArray,
    start: float,
    stop: float,
) -> tuple[int, ...]:
    """Return indices whose timestamps lie in [start, stop)."""
    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array")

    mask = (timestamps >= start) & (timestamps < stop)
    return tuple(int(index) for index in np.nonzero(mask)[0])


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

    def slice_time(self, start: float, stop: float) -> Track:
        """Return a cheap time-sliced view over [start, stop)."""
        return self._view_from_indices(
            indices_for_time_range(self.timestamps, start, stop)
        )

    def nearest_index(self, time: float) -> int:
        """Return the nearest native timestep index.

        Raises:
            IndexError: If the track is empty.
        """
        timestamps = self.timestamps
        if timestamps.size == 0:
            raise IndexError(
                f"cannot query nearest_index on an empty {type(self).__name__}"
            )
        return int(np.argmin(np.abs(timestamps - time)))

    def resample_to(self, timestamps: FloatArray) -> "Track":
        """Return this track resampled onto the requested timestamps.

        Concrete track types define their own interpolation semantics. For
        example, continuous mocap data may use interpolation, while discrete
        contact data may use nearest-neighbor or step-wise sampling.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement resample_to"
        )

    @classmethod
    def _view_type(cls) -> type[TrackView[Any]]:
        """Return the concrete view class for this owning track type."""
        raise TypeError(
            f"{cls.__name__} does not define `_view_type()`; "
            "override it to return the corresponding TrackView class."
        )

    def _view_from_indices(self, indices: tuple[int, ...]) -> Track:
        """Construct a view over this track using indices in this track's time basis."""
        return self._view_type()(source=self, indices=indices)


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
    indices: tuple[int, ...]

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

    def _view_from_indices(self, indices: tuple[int, ...]) -> TrackView[SourceTrack]:
        """Construct a same-type view from indices relative to this view."""
        source_indices = tuple(self.indices[index] for index in indices)
        return type(self)(source=self.source, indices=source_indices)