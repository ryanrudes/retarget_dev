"""Declarative contact-detection plans.

A :class:`ContactPlan` is the whole set of detections you want, defined at once:
each :class:`ContactQuery` pairs one patch/scope with one or more *named* supports
(and, optionally, its own config). :func:`apply_contact_plan` runs the lot and
attaches the result to the mocap in a single step -- you never hold or merge a
:class:`~retarget.contacts.state.SupportStateTrack` by hand.

This mirrors the repo's :class:`~retarget.demo.sync.SyncPlan` idiom: a frozen,
self-validating spec executed by a free function.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from retarget.contacts.config import ContactDetectionConfig
from retarget.contacts._scope import Scope
from retarget.contacts.detect import ContactDetector, Support
from retarget.contacts.state import SupportStateTrack
from retarget.core.schema.subject import Subjects
from retarget.core.targets import PatchTarget
from retarget.demo.mocap import MocapTrack

__all__ = ["ContactQuery", "ContactPlan", "apply_contact_plan"]


@dataclass(frozen=True, slots=True)
class ContactQuery:
    """One detection: a patch/scope against one or more named supports.

    ``against`` maps your label (the value ``support_state`` reads back) to the
    thing the patch is tested against -- a :class:`~retarget.contacts.SupportModel`,
    another ``Patch``/``Segment`` whose pose moves, or
    :func:`~retarget.contacts.infer_ground`. ``config`` overrides the plan's default
    for just this query (e.g. a tighter clearance for a hand); ``None`` inherits it.
    """

    scope: Scope
    against: Mapping[str, Support]
    config: ContactDetectionConfig | None = None

    def __post_init__(self) -> None:
        if not self.against:
            raise ValueError("ContactQuery.against must name at least one support")


@dataclass(frozen=True, slots=True)
class ContactPlan:
    """A declarative batch of patch-vs-support detections, defined all at once.

    Run it with :func:`apply_contact_plan` to detect every query, merge into one
    track, and attach it to the mocap. Nothing is mutated.
    """

    queries: tuple[ContactQuery, ...]
    config: ContactDetectionConfig = field(default_factory=ContactDetectionConfig)

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError("ContactPlan must contain at least one query")


def apply_contact_plan[SubjectsT: Subjects](
    mocap: MocapTrack[SubjectsT], plan: ContactPlan
) -> MocapTrack[SubjectsT]:
    """Detect every query in ``plan``, merge, and attach to ``mocap`` -- one step.

    Returns a new ``MocapTrack`` (the original is unchanged) carrying every
    ``(patch, support)`` pair, so each patch's ``support_state(...)`` reads its own
    supports. A transform like ``fill_pose_gaps``; the typed subjects are preserved.
    """
    tracks = [
        ContactDetector(query.config or plan.config).classify(query.scope, against=query.against)
        for query in plan.queries
    ]
    return mocap.with_support_states(_merge_support_state_tracks(tracks))


def _merge_support_state_tracks(tracks: list[SupportStateTrack]) -> SupportStateTrack:
    """Combine per-query support-state tracks onto one shared timeline.

    Disjoint ``(patch, support)`` keys (the normal case) are a plain union; on a
    genuine collision the booleans are OR-combined and scores take the max, and
    per-patch ``invalid`` masks OR together. Mirrors ``merge_contact_tracks``.
    """
    if not tracks:
        raise ValueError("_merge_support_state_tracks requires at least one track")
    timestamps = np.asarray(tracks[0].timestamps, dtype=np.float64)
    for track in tracks[1:]:
        other = np.asarray(track.timestamps, dtype=np.float64)
        if len(other) != len(timestamps) or not np.allclose(other, timestamps):
            raise ValueError("support-state tracks must share timestamps to merge")

    contacts: dict[PatchTarget, dict[str, np.ndarray]] = {}
    scores: dict[PatchTarget, dict[str, np.ndarray]] = {}
    invalid: dict[PatchTarget, np.ndarray] = {}
    for track in tracks:
        for target, per_support in track.contacts.items():
            dest = contacts.setdefault(target, {})
            for name, array in per_support.items():
                arr = np.asarray(array, dtype=np.bool_)
                dest[name] = arr.copy() if name not in dest else (dest[name] | arr)
        for target, per_support_scores in track.scores.items():
            dest_scores = scores.setdefault(target, {})
            for name, values in per_support_scores.items():
                arr_f = np.asarray(values, dtype=np.float64)
                dest_scores[name] = (
                    arr_f.copy() if name not in dest_scores else np.maximum(dest_scores[name], arr_f)
                )
        for target, mask in track.invalid.items():
            m = np.asarray(mask, dtype=np.bool_)
            invalid[target] = m.copy() if target not in invalid else (invalid[target] | m)

    return SupportStateTrack(
        contacts=contacts,
        scores=scores,
        timestamps=timestamps,
        invalid=invalid,
        nominal_hz_override=tracks[0].nominal_hz_override,
    )
