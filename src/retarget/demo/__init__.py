"""Demonstration layer for multimodal motion retargeting pipelines."""

from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    TrackAlignment,
    estimate_alignment_from_signals,
)
from retarget.demo.authoring import Tracks, build_demonstration
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.loaders import load_mocap_track
from retarget.demo.mocap import (
    MocapSegmentTrackView,
    MocapSubjectTrackView,
    MocapTrack,
    MocapTrackView,
)
from retarget.demo.tracks import (
    TrackSyncInfo,
    Track,
    TrackView,
)
from retarget.demo.sync import (
    SyncEdge,
    SyncPlan,
    compose_alignments_to_reference,
    estimate_sync,
    estimate_sync_and_resample_to_reference,
    estimate_sync_to_reference,
)

__all__ = [
    "ContactTrack",
    "ContactTrackView",
    "Tracks",
    "build_demonstration",

    # retarget.demo.demo
    "Demonstration",
    "DemonstrationView",


    "EnergySignal",
    "load_mocap_track",


    # retarget.demo.mocap
    "MocapTrack",
    "MocapTrackView",
    "MocapSegmentTrackView",
    "MocapSubjectTrackView",

    # retarget.demo.tracks
    "TrackSyncInfo",
    "Track",
    "TrackView",

    # retarget.demo.alignment
    "TimelineTransform",
    "TrackAlignment",
    "estimate_alignment_from_signals",

    # retarget.demo.sync
    "SyncEdge",
    "SyncPlan",
    "compose_alignments_to_reference",
    "estimate_sync",
    "estimate_sync_and_resample_to_reference",
    "estimate_sync_to_reference",
]
