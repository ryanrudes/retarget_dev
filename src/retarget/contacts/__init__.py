"""Automatic contact detection.

Quiet-interval detection, support-surface fitting, clearance/motion feature
scoring, and per-patch contact tracks. The primary entry points are the
config-bound :class:`ContactDetector` and the one-off :func:`detect_contacts` /
:func:`classify_contacts` convenience functions.
"""

from __future__ import annotations

from retarget.core.formats import JumpDetector, robust_jumps, speed_limit

from retarget.contacts.config import ContactDetectionConfig
from retarget.contacts.detect import (
    ContactDetector,
    classify_contacts,
    detect_contacts,
    merge_contact_tracks,
)
from retarget.contacts.plan import ContactPlan, ContactQuery, apply_contact_plan
from retarget.contacts.state import (
    SupportResolver,
    SupportStateTrack,
    highest_score,
    multi_label,
    priority,
)
from retarget.contacts.supports import (
    HeightmapSupportModel,
    InferredGround,
    LocalPercentileHeightmap,
    PlaneSupportModel,
    SupportModel,
    SupportModelType,
    SupportParams,
    TimeIndexedSupport,
    fit_best_support_surface,
    ground_plane,
    infer_ground,
    infer_support_surface,
    plane_support,
)

__all__ = [
    # detection
    "ContactDetector",
    "detect_contacts",
    "classify_contacts",
    "merge_contact_tracks",
    "ContactDetectionConfig",
    # declarative plan
    "ContactPlan",
    "ContactQuery",
    "apply_contact_plan",
    # support state + resolvers
    "SupportStateTrack",
    "SupportResolver",
    "priority",
    "highest_score",
    "multi_label",
    # garbage / jump detectors
    "JumpDetector",
    "speed_limit",
    "robust_jumps",
    # supports
    "SupportModel",
    "SupportModelType",
    "SupportParams",
    "PlaneSupportModel",
    "HeightmapSupportModel",
    "LocalPercentileHeightmap",
    "TimeIndexedSupport",
    "fit_best_support_surface",
    "infer_support_surface",
    "InferredGround",
    "infer_ground",
    "ground_plane",
    "plane_support",
]
