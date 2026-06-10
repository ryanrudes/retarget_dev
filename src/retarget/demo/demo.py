"""Generic demonstration container and sliced views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeVar, cast

from retarget.core.enums import TrackId
from retarget.demo.alignment import (
    EnergySignal,
    TrackAlignment,
    estimate_alignment_from_signals,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Demonstration[K: TrackId]:
    """Multimodal demonstration keyed by typed track identifiers."""

    tracks: Mapping[K, object]
    alignments: tuple[TrackAlignment[K], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", MappingProxyType(dict(self.tracks)))

    def track(self, track: K) -> object:
        return self.tracks[track]

    def typed_track[T](
        self,
        track: K,
        expected_type: type[T] | tuple[type[Any], ...],
    ) -> T:
        value = self.track(track)
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Track {track!r} is not of expected type {expected_type}; "
                f"got {type(value).__name__}"
            )
        return cast(T, value)

    def slice_time(self, start: float, stop: float) -> DemonstrationView[K]:
        sliced_tracks: dict[K, object] = {}
        for track_id, value in self.tracks.items():
            sliced_tracks[track_id] = _slice_track(value, start, stop)
        return DemonstrationView(
            source=self,
            tracks=sliced_tracks,
            alignments=self.alignments,
        )

    def with_track(self, track: K, value: object) -> Demonstration[K]:
        updated = dict(self.tracks)
        updated[track] = value
        return Demonstration(tracks=updated, alignments=self.alignments)

    def with_alignment(self, alignment: TrackAlignment[K]) -> Demonstration[K]:
        return Demonstration(
            tracks=self.tracks,
            alignments=(*self.alignments, alignment),
        )

    def align(
        self,
        *,
        reference: K,
        source: K,
        reference_signal: EnergySignal,
        source_signal: EnergySignal,
        max_lag_seconds: float,
    ) -> Demonstration[K]:
        transform, score = estimate_alignment_from_signals(
            reference=reference_signal,
            source=source_signal,
            max_lag_seconds=max_lag_seconds,
        )
        alignment = TrackAlignment(
            source=source,
            reference=reference,
            transform=transform,
            score=score,
        )
        return self.with_alignment(alignment)


@dataclass(frozen=True, slots=True)
class DemonstrationView[K: TrackId]:
    """Time-sliced view over a demonstration."""

    source: Demonstration[K]
    tracks: Mapping[K, object]
    alignments: tuple[TrackAlignment[K], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", MappingProxyType(dict(self.tracks)))

    def track(self, track: K) -> object:
        return self.tracks[track]

    def typed_track[T](
        self,
        track: K,
        expected_type: type[T] | tuple[type[Any], ...],
    ) -> T:
        value = self.track(track)
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Track {track!r} is not of expected type {expected_type}; "
                f"got {type(value).__name__}"
            )
        return cast(T, value)

    def slice_time(self, start: float, stop: float) -> DemonstrationView[K]:
        sliced_tracks: dict[K, object] = {}
        for track_id, value in self.tracks.items():
            sliced_tracks[track_id] = _slice_track(value, start, stop)
        return DemonstrationView(
            source=self.source,
            tracks=sliced_tracks,
            alignments=self.alignments,
        )

    def resample_to(self, reference: K) -> DemonstrationView[K]:
        raise NotImplementedError(
            "DemonstrationView.resample_to requires alignment-aware track resampling; "
            "this is not implemented yet."
        )


def _slice_track(value: object, start: float, stop: float) -> object:
    if hasattr(value, "slice_time"):
        return value.slice_time(start, stop)  # type: ignore[attr-defined]
    return value
