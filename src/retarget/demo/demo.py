"""Generic demonstration containers and sliced/resampled views.

A :class:`Demonstration` owns a mapping of typed track identifiers to concrete
demo tracks. ``slice_time(...)`` returns a :class:`DemonstrationView`, which is
a lightweight container over sliced track views while preserving the root demo
as ``source``.

``DemonstrationView.resample_to(...)`` materializes a new view whose tracks are
sampled onto a shared reference timeline. Pairwise or composed alignments tell
the method how to convert reference timestamps into each source track's native
time basis before delegating interpolation/discrete sampling to each track.
"""

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

    def resample_with_alignments(
        self,
        reference: K,
        alignments: tuple[TrackAlignment[K], ...],
    ) -> DemonstrationView[K]:
        """Compatibility wrapper for pairwise alignments.

        Prefer :func:`retarget.demo.sync.estimate_sync_and_resample_to_reference`
        for the full sync-and-resample workflow.
        """
        import dataclasses
        from retarget.demo.sync import compose_alignments_to_reference

        # Keep the legacy pairwise-alignment path alive for existing callers.
        composed = compose_alignments_to_reference(
            reference=reference,
            alignments=alignments,
        )

        if isinstance(self, DemonstrationView):
            source = dataclasses.replace(self.source, alignments=composed)
            view = dataclasses.replace(self, source=source, alignments=composed)
        else:
            view = DemonstrationView(
                source=self,
                tracks=self.tracks,
                alignments=composed,
            )

        return view.resample_to(reference)

    def _view_source(self) -> Demonstration[K]:
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class DemonstrationView[K: TrackId](Demonstration[K]):
    """Time-sliced view over a demonstration."""

    source: Demonstration[K]

    def _view_source(self) -> Demonstration[K]:
        return self.source

    def resample_to(self, reference: K) -> DemonstrationView[K]:
        """Return a materialized view resampled onto a reference track timeline."""
        if reference not in self.tracks:
            raise KeyError(f"Reference track {reference!r} is not in this view")

        reference_track = self[reference]
        reference_timestamps = reference_track.timestamps
        alignment_by_source = _alignments_to_reference(
            reference=reference,
            alignments=self.alignments,
        )

        resampled_tracks: dict[K, Track] = {}
        for track_id, track in self.tracks.items():
            if track_id == reference:
                resampled_tracks[track_id] = track
                continue

            try:
                alignment = alignment_by_source[track_id]
            except KeyError as exc:
                raise ValueError(
                    f"Track {track_id!r} has no alignment to reference {reference!r}"
                ) from exc

            source_timestamps = alignment.transform.to_source(reference_timestamps)
            try:
                resampled_tracks[track_id] = track.resample_to(
                    source_timestamps,
                    output_timestamps=reference_timestamps,
                )
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"Cannot resample track {track_id!r} onto reference {reference!r}: "
                    f"{type(track).__name__} does not implement resample_to"
                ) from exc

        return DemonstrationView(
            source=self.source,
            tracks=resampled_tracks,
            alignments=self.alignments,
        )


def _alignments_to_reference[K: TrackId](
    *,
    reference: K,
    alignments: tuple[TrackAlignment[K], ...],
) -> Mapping[K, TrackAlignment[K]]:
    alignment_by_source: dict[K, TrackAlignment[K]] = {}
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


def _freeze_tracks[K: TrackId](tracks: Mapping[K, Track]) -> Mapping[K, Track]:
    copied = dict(tracks)
    for key, value in copied.items():
        if not isinstance(value, Track):
            raise TypeError(
                f"Track {key!r} must be a Track; got {type(value).__name__}"
            )
    return MappingProxyType(copied)
