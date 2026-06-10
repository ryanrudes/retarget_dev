"""Example-specific ground-estimation demonstration wrapper and loader."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from mocap_specs import VICON_SCENE
from retarget.core.enums import TrackId
from retarget.demo.alignment import EnergySignal, TrackAlignment, estimate_alignment_from_signals
from retarget.demo.contact import ContactTrack
from retarget.demo.demo import Demonstration, _slice_track
from retarget.demo.loaders import load_mocap_track
from retarget.demo.mocap import MocapTrack, MocapTrackView
from retarget.io import UnbaggedDirectory


class GroundEstimationTrackId(TrackId):
    """Track identifiers for the ground-estimation demonstration."""

    MOCAP = "mocap"
    VIDEO = "video"
    SMPL = "smpl"
    CONTACTS = "contacts"


@dataclass(frozen=True, slots=True)
class GroundEstimationDemo:
    """Ground-estimation demonstration with ergonomic track accessors."""

    mocap: MocapTrack
    tracks: Mapping[GroundEstimationTrackId, object]
    alignments: tuple[TrackAlignment[GroundEstimationTrackId], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", MappingProxyType(dict(self.tracks)))

    def track(self, track: GroundEstimationTrackId) -> object:
        if track is GroundEstimationTrackId.MOCAP:
            return self.mocap
        return self.tracks[track]

    @property
    def video(self) -> object:
        if GroundEstimationTrackId.VIDEO not in self.tracks:
            raise KeyError("No video track is attached to this demonstration")
        return self.tracks[GroundEstimationTrackId.VIDEO]

    @property
    def smpl(self) -> object:
        if GroundEstimationTrackId.SMPL not in self.tracks:
            raise KeyError("No SMPL track is attached to this demonstration")
        return self.tracks[GroundEstimationTrackId.SMPL]

    @property
    def contacts(self) -> ContactTrack:
        value = self.tracks.get(GroundEstimationTrackId.CONTACTS)
        if value is None:
            raise KeyError("No contact track is attached to this demonstration")
        if not isinstance(value, ContactTrack):
            raise TypeError("CONTACTS track is not a ContactTrack")
        return value

    def slice_time(self, start: float, stop: float) -> GroundEstimationDemoView:
        sliced_mocap = self.mocap.slice_time(start, stop)
        sliced_tracks: dict[GroundEstimationTrackId, object] = {
            GroundEstimationTrackId.MOCAP: sliced_mocap,
        }
        for track_id, value in self.tracks.items():
            if track_id is GroundEstimationTrackId.MOCAP:
                continue
            sliced_tracks[track_id] = _slice_track(value, start, stop)
        return GroundEstimationDemoView(
            source=self,
            mocap=sliced_mocap,
            tracks=sliced_tracks,
            alignments=self.alignments,
        )

    def with_track(
        self,
        track: GroundEstimationTrackId,
        value: object,
    ) -> GroundEstimationDemo:
        updated = dict(self.tracks)
        updated[track] = value
        if track is GroundEstimationTrackId.MOCAP:
            if not isinstance(value, MocapTrack):
                raise TypeError("MOCAP track must be a MocapTrack")
            return GroundEstimationDemo(
                mocap=value,
                tracks=updated,
                alignments=self.alignments,
            )
        return GroundEstimationDemo(
            mocap=self.mocap,
            tracks=updated,
            alignments=self.alignments,
        )

    def with_contacts(self, contacts: ContactTrack) -> GroundEstimationDemo:
        mocap_with_contacts = MocapTrack(
            scene_spec=self.mocap.scene_spec,
            state=self.mocap.state,
            timestamps=self.mocap.timestamps,
            marker_frames=self.mocap.marker_frames,
            contacts=contacts,
        )
        updated = dict(self.tracks)
        updated[GroundEstimationTrackId.MOCAP] = mocap_with_contacts
        updated[GroundEstimationTrackId.CONTACTS] = contacts
        return GroundEstimationDemo(
            mocap=mocap_with_contacts,
            tracks=updated,
            alignments=self.alignments,
        )

    def with_alignment(
        self,
        alignment: TrackAlignment[GroundEstimationTrackId],
    ) -> GroundEstimationDemo:
        return GroundEstimationDemo(
            mocap=self.mocap,
            tracks=self.tracks,
            alignments=(*self.alignments, alignment),
        )

    def align(
        self,
        *,
        reference: GroundEstimationTrackId,
        source: GroundEstimationTrackId,
        reference_signal: EnergySignal,
        source_signal: EnergySignal,
        max_lag_seconds: float,
    ) -> GroundEstimationDemo:
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

    def as_generic(self) -> Demonstration[GroundEstimationTrackId]:
        return Demonstration(tracks=self.tracks, alignments=self.alignments)


@dataclass(frozen=True, slots=True)
class GroundEstimationDemoView:
    """Time-sliced ground-estimation demonstration view."""

    source: GroundEstimationDemo
    mocap: MocapTrack | MocapTrackView
    tracks: Mapping[GroundEstimationTrackId, object]
    alignments: tuple[TrackAlignment[GroundEstimationTrackId], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", MappingProxyType(dict(self.tracks)))

    def track(self, track: GroundEstimationTrackId) -> object:
        if track is GroundEstimationTrackId.MOCAP:
            return self.mocap
        return self.tracks[track]

    @property
    def video(self) -> object:
        if GroundEstimationTrackId.VIDEO not in self.tracks:
            raise KeyError("No video track is attached to this demonstration")
        return self.tracks[GroundEstimationTrackId.VIDEO]

    @property
    def smpl(self) -> object:
        if GroundEstimationTrackId.SMPL not in self.tracks:
            raise KeyError("No SMPL track is attached to this demonstration")
        return self.tracks[GroundEstimationTrackId.SMPL]

    def slice_time(self, start: float, stop: float) -> GroundEstimationDemoView:
        sliced_mocap = self.mocap.slice_time(start, stop)  # type: ignore[union-attr]
        sliced_tracks: dict[GroundEstimationTrackId, object] = {
            GroundEstimationTrackId.MOCAP: sliced_mocap,
        }
        for track_id, value in self.tracks.items():
            if track_id is GroundEstimationTrackId.MOCAP:
                continue
            sliced_tracks[track_id] = _slice_track(value, start, stop)
        return GroundEstimationDemoView(
            source=self.source,
            mocap=sliced_mocap,
            tracks=sliced_tracks,
            alignments=self.alignments,
        )

    def resample_to(
        self,
        reference: GroundEstimationTrackId,
    ) -> GroundEstimationDemoView:
        raise NotImplementedError(
            "DemonstrationView.resample_to requires alignment-aware track resampling; "
            "this is not implemented yet."
        )


def load_ground_estimation_demo(
    root: Path | UnbaggedDirectory,
) -> GroundEstimationDemo:
    """
    Load a ground-estimation demonstration from an unbagged export directory.

    The first pass is mocap-only. Video, SMPL, and contact tracks may be added later.
    """
    mocap = load_mocap_track(root, VICON_SCENE)
    # Demonstration time is relative to the first frame so slice_time(0.0, 1.0)
    # means "the first second of the clip", not ROS epoch seconds.
    timestamps = mocap.timestamps - mocap.timestamps[0]
    mocap = MocapTrack(
        scene_spec=mocap.scene_spec,
        state=mocap.state,
        timestamps=timestamps,
        marker_frames=mocap.marker_frames,
    )
    tracks = {GroundEstimationTrackId.MOCAP: mocap}
    return GroundEstimationDemo(mocap=mocap, tracks=tracks)
