"""Generic demonstration container and sliced views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from retarget.core.enums import TrackId
from retarget.demo.alignment import TrackAlignment
from retarget.demo.tracks import Track


@dataclass(frozen=True, slots=True)
class Demonstration[K: TrackId]:
    """Multimodal demonstration keyed by typed track identifiers."""

    tracks: Mapping[K, Track]
    alignments: tuple[TrackAlignment[K], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", _freeze_tracks(self.tracks))

    def __getitem__(self, track: K) -> Track:
        return self.tracks[track]

    def get_track(self, track: K) -> Track:
        return self[track]

    def slice_time(self, start: float, stop: float) -> DemonstrationView[K]:
        return DemonstrationView(
            source=self._view_source(),
            tracks={
                track_id: track.slice_time(start, stop)
                for track_id, track in self.tracks.items()
            },
            alignments=self.alignments,
        )

    def _view_source(self) -> Demonstration[K]:
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class DemonstrationView[K: TrackId](Demonstration[K]):
    """Time-sliced view over a demonstration."""

    source: Demonstration[K]

    def _view_source(self) -> Demonstration[K]:
        return self.source

    def resample_to(self, reference: K) -> DemonstrationView[K]:
        raise NotImplementedError(
            "DemonstrationView.resample_to requires alignment-aware track resampling; "
            "this is not implemented yet."
        )


def _freeze_tracks[K: TrackId](tracks: Mapping[K, Track]) -> Mapping[K, Track]:
    copied = dict(tracks)
    for key, value in copied.items():
        if not isinstance(value, Track):
            raise TypeError(
                f"Track {key!r} must be a Track; got {type(value).__name__}"
            )
    return MappingProxyType(copied)
