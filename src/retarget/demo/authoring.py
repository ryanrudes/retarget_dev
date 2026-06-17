"""Typed authoring helpers for demonstration track containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict, cast

from retarget.core.enums import TrackId
from retarget.core.schema import _runtime_id_type
from retarget.demo.demo import Demonstration, _GeneratedTrackIds


class Tracks(TypedDict):
    """Base class for typed demonstration track schema declarations."""


@dataclass(frozen=True, slots=True, kw_only=True)
class TypedDemonstration[TracksT: Tracks](Demonstration[TrackId]):
    """Demonstration built from typed authoring declarations."""

    _typed_tracks: TracksT = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        Demonstration.__post_init__(self)

    @property
    def typed_tracks(self) -> TracksT:
        return cast(TracksT, self._typed_tracks)


def build_demonstration[TracksT: Tracks](
    tracks: TracksT,
) -> TypedDemonstration[TracksT]:
    """Compile typed demo-track authoring into a runtime demonstration."""
    track_items = tuple(tracks.items())
    track_type = _runtime_id_type(
        "Track",
        tuple(name for name, _ in track_items),
        TrackId,
    )
    compiled_tracks = {
        track_type(track_name): track
        for track_name, track in track_items
    }
    return TypedDemonstration(
        tracks=compiled_tracks,
        _generated_ids=_GeneratedTrackIds(tracks=track_type),
        _typed_tracks=cast(TracksT, MappingProxyType(dict(tracks))),
    )
