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
from retarget.demo.pose_repair import fill_pose_gaps
from retarget.demo.smpl import SmplTrack, smpl_joint_energy
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
    "SmplTrack",
    "Track",
    "TrackView",
    "TrackSyncInfo",

    # Pose repair
    "fill_pose_gaps",

    # Alignment
    "EnergySignal",
    "TimelineTransform",
    "TrackAlignment",
    "estimate_alignment_from_signals",
    "smpl_joint_energy",

    # Sync
    "SyncEdge",
    "SyncPlan",
    "compose_alignments_to_reference",
    "estimate_sync",
    "estimate_sync_and_resample_to_reference",
    "estimate_sync_to_reference",
]
