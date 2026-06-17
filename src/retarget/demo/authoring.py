"""Typed authoring helpers for demonstration track containers."""

from __future__ import annotations

from typing import TypedDict

from retarget.core.enums import TrackId
from retarget.core.schema import _runtime_id_type
from retarget.demo.demo import Demonstration, _GeneratedTrackIds


class Tracks(TypedDict):
    """Base class for typed demonstration track schema declarations."""


def build_demonstration[TracksT: Tracks](
    tracks: TracksT,
) -> Demonstration[TrackId]:
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
    return Demonstration(
        tracks=compiled_tracks,
        _generated_ids=_GeneratedTrackIds(tracks=track_type),
    )
