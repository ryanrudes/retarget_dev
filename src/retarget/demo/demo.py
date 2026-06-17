"""Typed demonstration container and sliced/resampled views.

A :class:`Demonstration` owns a typed ``Tracks`` mapping (a ``TypedDict`` of
named tracks). ``demo.tracks["mocap"]`` is statically typed from the user's
schema. ``slice_time(...)`` returns a :class:`DemonstrationView`, whose
``tracks`` are sliced/view tracks typed through the common ``Track`` surface.

``DemonstrationView.resample_to(...)`` materializes a new view whose tracks are
sampled onto a shared reference timeline using stored alignments.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypedDict, cast

import numpy as np

from retarget.demo.alignment import TrackAlignment
from retarget.demo.resampling import ResampleMethod
from retarget.demo.tracks import Track


class Tracks(TypedDict):
    """Base class for typed demonstration track-schema declarations."""


class _TrackMapping:
    """Shared read/slice surface for demonstrations and their views.

    Subclasses provide ``_track_map()`` (the visible track mapping) and
    ``_root()`` (the originating root demonstration for sliced views).
    """

    __slots__ = ()

    alignments: tuple[TrackAlignment, ...]

    def _track_map(self) -> Mapping[str, Track]:
        raise NotImplementedError

    def _root(self) -> Demonstration[Any]:
        raise NotImplementedError

    def __getitem__(self, track: str) -> Track:
        return self._track_map()[track]

    def __contains__(self, track: object) -> bool:
        return track in self._track_map()

    def __iter__(self) -> Iterator[str]:
        return iter(self._track_map())

    def track_ids(self) -> tuple[str, ...]:
        """Return the track names in insertion order."""
        return tuple(self._track_map())

    def slice_time(self, start: float, stop: float) -> DemonstrationView:
        return DemonstrationView(
            source=self._root(),
            tracks={
                name: track.slice_time(start, stop)
                for name, track in self._track_map().items()
            },
            alignments=self.alignments,
        )


@dataclass(frozen=True, slots=True)
class Demonstration[TracksT: Tracks = Tracks](_TrackMapping):
    """Multimodal demonstration keyed by authored track names."""

    tracks: TracksT
    alignments: tuple[TrackAlignment, ...] = ()

    def __post_init__(self) -> None:
        frozen = _freeze_tracks(cast(Mapping[str, Track], self.tracks))
        object.__setattr__(self, "tracks", cast(TracksT, frozen))

    def _track_map(self) -> Mapping[str, Track]:
        return cast(Mapping[str, Track], self.tracks)

    def _root(self) -> Demonstration[Any]:
        return self


@dataclass(frozen=True, slots=True)
class DemonstrationView(_TrackMapping):
    """Time-sliced / resampled view over a demonstration."""

    source: Demonstration[Any]
    tracks: Mapping[str, Track]
    alignments: tuple[TrackAlignment, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", MappingProxyType(dict(self.tracks)))

    def _track_map(self) -> Mapping[str, Track]:
        return self.tracks

    def _root(self) -> Demonstration[Any]:
        return self.source

    def resample_to(
        self,
        reference: str,
        *,
        method: ResampleMethod | str = ResampleMethod.NEAREST,
    ) -> DemonstrationView:
        """Return a materialized view resampled onto a reference track timeline.

        ``method`` is the discrete sampling policy forwarded to every
        non-reference track (continuous quantities interpolate regardless).
        """
        if reference not in self.tracks:
            raise KeyError(f"Reference track {reference!r} is not in this view")

        reference_track = self.tracks[reference]
        reference_timestamps = reference_track.timestamps
        alignment_by_source = _alignments_to_reference(
            reference=reference, alignments=self.alignments
        )

        resampled_tracks: dict[str, Track] = {}
        for name, track in self.tracks.items():
            if name == reference:
                resampled_tracks[name] = track
                continue
            try:
                alignment = alignment_by_source[name]
            except KeyError as exc:
                raise ValueError(
                    f"Track {name!r} has no alignment to reference {reference!r}"
                ) from exc
            source_timestamps = cast(
                np.ndarray, alignment.transform.to_source(reference_timestamps)
            )
            try:
                resampled_tracks[name] = track.resample_to(
                    source_timestamps,
                    output_timestamps=reference_timestamps,
                    method=method,
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"Cannot resample track {name!r} onto reference {reference!r}: "
                    f"{type(track).__name__} does not implement resample_to"
                ) from exc

        return DemonstrationView(
            source=self.source,
            tracks=resampled_tracks,
            alignments=self.alignments,
        )


def _alignments_to_reference(
    *,
    reference: str,
    alignments: tuple[TrackAlignment, ...],
) -> Mapping[str, TrackAlignment]:
    alignment_by_source: dict[str, TrackAlignment] = {}
    for alignment in alignments:
        if alignment.reference != reference:
            continue
        if alignment.source in alignment_by_source:
            raise ValueError(
                f"Multiple alignments from {alignment.source!r} "
                f"to reference {reference!r}"
            )
        alignment_by_source[alignment.source] = alignment
    return MappingProxyType(alignment_by_source)


def _freeze_tracks(tracks: Mapping[str, Track]) -> Mapping[str, Track]:
    copied = dict(tracks)
    for key, value in copied.items():
        if not isinstance(value, Track):
            raise TypeError(
                f"Track {key!r} must be a Track; got {type(value).__name__}"
            )
    return MappingProxyType(copied)
