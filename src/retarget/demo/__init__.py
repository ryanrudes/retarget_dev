"""Demonstration layer for multimodal motion retargeting pipelines."""

from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    TrackAlignment,
    estimate_alignment_from_signals,
)
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.demo import (
    Demonstration,
    DemonstrationView,
    Tracks,
)
from retarget.demo.mocap import MocapTrack
from retarget.demo.tracks import (
    Track,
    TrackSyncInfo,
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
    # Containers
    "Demonstration",
    "DemonstrationView",
    "Tracks",

    # Tracks
    "MocapTrack",
    "ContactTrack",
    "ContactTrackView",
    "Track",
    "TrackView",
    "TrackSyncInfo",

    # Alignment
    "EnergySignal",
    "TimelineTransform",
    "TrackAlignment",
    "estimate_alignment_from_signals",

    # Sync
    "SyncEdge",
    "SyncPlan",
    "compose_alignments_to_reference",
    "estimate_sync",
    "estimate_sync_and_resample_to_reference",
    "estimate_sync_to_reference",
]
