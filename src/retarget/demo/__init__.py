"""Demonstration layer for multimodal motion retargeting pipelines."""

from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    TrackAlignment,
    estimate_alignment_from_signals,
)
from retarget.demo.contact import ContactTrack, ContactTrackView
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.loaders import load_mocap_track
from retarget.demo.mocap import (
    MocapSegmentTrackView,
    MocapSubjectTrackView,
    MocapTrack,
    MocapTrackView,
)
from retarget.demo.tracks import TimeRange, TimeTrack

__all__ = [
    "ContactTrack",
    "ContactTrackView",
    "Demonstration",
    "DemonstrationView",
    "EnergySignal",
    "load_mocap_track",
    "MocapSegmentTrackView",
    "MocapSubjectTrackView",
    "MocapTrack",
    "MocapTrackView",
    "TimeRange",
    "TimeTrack",
    "TimelineTransform",
    "TrackAlignment",
    "estimate_alignment_from_signals",
]
