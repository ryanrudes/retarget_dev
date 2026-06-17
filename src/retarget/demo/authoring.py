"""Typed authoring helpers for demonstration track containers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict, cast

from retarget.core.enums import TrackId
from retarget.core.schema import _runtime_id_type
from retarget.demo.alignment import TrackAlignment
from retarget.demo.demo import Demonstration, DemonstrationView, _GeneratedTrackIds


class Tracks(TypedDict):
    """Base class for typed demonstration track schema declarations."""


def _rebuild_schema_tracks[TracksT: Tracks](
    demo: Demonstration[TrackId],
    template: TracksT,
) -> TracksT:
    return cast(
        TracksT,
        MappingProxyType(
            {name: demo._get_track(name) for name in template},
        ),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TypedDemonstrationView[TracksT: Tracks](DemonstrationView[TrackId]):
    """Sliced or resampled view over a typed demonstration."""

    _schema_tracks: TracksT = field(repr=False, compare=False)

    @property
    def tracks(self) -> TracksT:
        return cast(TracksT, self._schema_tracks)

    def resample_to(self, reference: TrackId | str) -> TypedDemonstrationView[TracksT]:
        view = DemonstrationView.resample_to(self, reference)
        return TypedDemonstrationView(
            source=view.source,
            tracks=view._tracks,
            alignments=view.alignments,
            _generated_ids=view._generated_ids,
            _schema_tracks=_rebuild_schema_tracks(view, self._schema_tracks),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TypedDemonstration[TracksT: Tracks](Demonstration[TrackId]):
    """Demonstration built from typed authoring declarations."""

    _schema_tracks: TracksT = field(repr=False, compare=False)

    @property
    def tracks(self) -> TracksT:
        return cast(TracksT, self._schema_tracks)

    def slice_time(self, start: float, stop: float) -> TypedDemonstrationView[TracksT]:
        view = Demonstration.slice_time(self, start, stop)
        return TypedDemonstrationView(
            source=view.source,
            tracks=view._tracks,
            alignments=view.alignments,
            _generated_ids=view._generated_ids,
            _schema_tracks=_rebuild_schema_tracks(view, self._schema_tracks),
        )

    def resample_with_alignments(
        self,
        reference: TrackId | str,
        alignments: tuple[TrackAlignment[TrackId], ...],
    ) -> TypedDemonstrationView[TracksT]:
        view = Demonstration.resample_with_alignments(self, reference, alignments)
        assert isinstance(view, DemonstrationView)
        return TypedDemonstrationView(
            source=view.source,
            tracks=view._tracks,
            alignments=view.alignments,
            _generated_ids=view._generated_ids,
            _schema_tracks=_rebuild_schema_tracks(view, self._schema_tracks),
        )


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
    schema_tracks = cast(TracksT, MappingProxyType(dict(tracks)))
    return TypedDemonstration(
        tracks=compiled_tracks,
        _generated_ids=_GeneratedTrackIds(tracks=track_type),
        _schema_tracks=schema_tracks,
    )
