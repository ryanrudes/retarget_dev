"""The contact-detection driver.

:class:`ContactDetector` binds a :class:`ContactDetectionConfig` and detects
contacts for the patches in a *scope* (a ``Patch``/``Segment``/``Subject``/
``MocapTrack`` or an iterable of patches) against a *support* (an inferred floor
when ``against`` is ``None``; a provided ``SupportModel``; or a moving support
derived from another patch/segment). Each patch is detected independently and
keyed by its :class:`~retarget.core.targets.PatchTarget`.

``detect_contacts`` is a thin one-off convenience that constructs a detector and
delegates, so a single implementation lives in the class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from retarget.contacts._features import angular_speed_from_frames, compute_features
from retarget.contacts._quiet import QuietResult, detect_quiet, local_polynomial_derivative, score_hysteresis_mask
from retarget.contacts._scoring import score_features
from retarget.contacts.config import ContactDetectionConfig, ResolvedConfig
from retarget.contacts.state import SupportStateTrack
from retarget.contacts.supports import (
    InferredGround,
    SupportModel,
    TimeIndexedSupport,
    _up_vector,
    infer_ground,
    infer_support_surface,
)
from retarget.core.schema.patch import Patch
from retarget.core.schema.segment import Segment
from retarget.core.schema.subject import Subject
from retarget.core.targets import PatchTarget
from retarget.core.types import FloatArray, FloatArray1D, Points3, TimeBool, TimeMat3, TimeVec3
from retarget.demo.contact import ContactTrack
from retarget.demo.mocap import MocapTrack
from retarget.utils.intervals import clean_mask_by_time, intervals_from_mask, mask_from_intervals

Support = SupportModel | TimeIndexedSupport | Patch[Any] | Segment[Any, Any] | InferredGround
Scope = Patch[Any] | Segment[Any, Any] | Subject[Any] | MocapTrack[Any] | Iterable[Patch[Any]]


@dataclass(frozen=True, slots=True)
class _PatchData:
    """Per-patch trajectory + quiet detection, gathered once and reused per support."""

    target: PatchTarget
    points: FloatArray
    sample_points: FloatArray  # (T, K, 3): footprint samples for region-aware clearance
    velocities: FloatArray
    frames: TimeMat3
    quiet: QuietResult
    coverage: FloatArray1D
    timestamps: FloatArray


def _normalize_scope(scope: Scope) -> list[Patch[Any]]:
    """Resolve a scope to the geometry-bearing patches to detect on."""
    if isinstance(scope, Patch):
        if not scope.has_geometry():
            raise ValueError("the given patch has no calibrated geometry to detect on")
        return [scope]
    if isinstance(scope, Segment):
        patches: list[Patch[Any]] = list(cast(Mapping[str, Patch[Any]], scope.patches).values())
    elif isinstance(scope, Subject):
        patches = [
            patch
            for segment in cast(Mapping[str, Segment[Any, Any]], scope.segments).values()
            for patch in cast(Mapping[str, Patch[Any]], segment.patches).values()
        ]
    elif isinstance(scope, MocapTrack):
        patches = [
            patch
            for subject in cast(Mapping[str, Subject[Any]], scope.subjects).values()
            for segment in cast(Mapping[str, Segment[Any, Any]], subject.segments).values()
            for patch in cast(Mapping[str, Patch[Any]], segment.patches).values()
        ]
    else:
        patches = list(scope)
        for patch in patches:
            if not isinstance(patch, Patch):
                raise TypeError(f"scope iterable must contain Patch objects; got {type(patch).__name__}")
    geometry = [patch for patch in patches if patch.has_geometry()]
    if not geometry:
        raise ValueError("scope contains no geometry-bearing patches to detect on")
    return geometry


def _patch_timestamps(patch: Patch[Any]) -> FloatArray:
    # The bound runtime carries the (possibly sliced) mocap timeline shared by the scope.
    return cast(FloatArray, np.asarray(patch._runtime().timestamps, dtype=np.float64))


def _gather_patch_data(patch: Patch[Any], resolved: ResolvedConfig) -> _PatchData:
    points = cast(FloatArray, np.asarray(patch.points(), dtype=np.float64))
    timestamps = _patch_timestamps(patch)
    velocities = local_polynomial_derivative(
        points, timestamps, resolved.quiet.derivative_window_time, resolved.quiet.derivative_poly_degree
    )
    frames = patch.frames()
    quiet = detect_quiet(points, timestamps, resolved.quiet)
    coverage = cast(FloatArray1D, np.asarray(patch.pose_coverage(), dtype=np.float64))
    # Region-aware contact samples the rectangle's corners + center; for a planar
    # support the closest approach is exactly at a corner. Motion features stay on
    # the center point; only clearance uses the footprint.
    if resolved.region_contact and patch.region is not None:
        corners = np.asarray(patch.boundary_points(), dtype=np.float64)  # (T, 4, 3)
        sample_points = cast(FloatArray, np.concatenate([corners, points[:, None, :]], axis=1))
    else:
        sample_points = points[:, None, :]
    return _PatchData(
        target=patch.target,
        points=points,
        sample_points=sample_points,
        velocities=velocities,
        frames=frames,
        quiet=quiet,
        coverage=coverage,
        timestamps=timestamps,
    )


def _moving_support_from(against: Patch[Any] | Segment[Any, Any], up_axis: int) -> TimeIndexedSupport:
    """Build a per-frame moving support from a tracked patch (deck surface) or segment pose."""
    if isinstance(against, Patch):
        origins = cast(TimeVec3, np.asarray(against.points(), dtype=np.float64))
        normals = cast(TimeVec3, np.asarray(against.normals(), dtype=np.float64))
        name = against.target.patch
    else:
        runtime_rotations = np.asarray(against.rotations(), dtype=np.float64)
        origins = cast(TimeVec3, np.asarray(against.translations(), dtype=np.float64))
        up = _up_vector(up_axis)
        normals = cast(TimeVec3, np.einsum("tij,j->ti", runtime_rotations, up))
        name = against.segment_target().segment
    return TimeIndexedSupport(origins=origins, normals=normals, name=name)


def _resolve_support(
    against: Support,
    data: list[_PatchData],
    resolved: ResolvedConfig,
) -> SupportModel | TimeIndexedSupport:
    """Resolve ``against`` to a single support shared by every patch in the scope."""
    if isinstance(against, InferredGround):
        cloud = cast(Points3, np.vstack([d.points for d in data]))
        quiet_mask = np.concatenate([np.asarray(d.quiet.mask) for d in data])
        return infer_support_surface(cloud, quiet_mask, resolved.support)
    if isinstance(against, (Patch, Segment)):
        return _moving_support_from(against, resolved.up_axis)
    if isinstance(against, TimeIndexedSupport):
        return against
    if isinstance(against, SupportModel):
        return against
    raise TypeError(f"unsupported `against` value of type {type(against).__name__}")


def _evaluate_support(
    support: SupportModel | TimeIndexedSupport,
    sample_points: FloatArray,
    center_points: FloatArray,
    target: PatchTarget,
) -> tuple[FloatArray, Points3]:
    """Return ``(closest-approach clearance (T,), normals (T, 3))`` of the footprint.

    Clearance is the minimum over the ``K`` footprint samples in ``sample_points``
    ``(T, K, 3)`` (the rectangle corners + center; ``K=1`` for point-based), so a
    patch reads as touching when its nearest part does. The support normal is read at
    ``center_points`` ``(T, 3)``.
    """
    n_time, n_samples, _ = sample_points.shape
    if isinstance(support, TimeIndexedSupport):
        if len(support) != n_time:
            raise ValueError(
                f"moving support length {len(support)} does not match patch "
                f"{target.patch!r} length {n_time} (different timelines?)"
            )
        diff = sample_points - support.origins[:, None, :]  # (T, K, 3)
        per_sample = np.einsum("tkd,td->tk", diff, support.normals)  # (T, K)
        return cast(FloatArray, per_sample.min(axis=1)), support.normals
    flat = cast(Points3, sample_points.reshape(n_time * n_samples, 3))
    per_sample = np.asarray(support.clearance(flat), dtype=np.float64).reshape(n_time, n_samples)
    normals = np.asarray(support.normals_at(cast(Points3, center_points)), dtype=np.float64)
    return cast(FloatArray, per_sample.min(axis=1)), cast(Points3, normals)


def _detect_patch(
    data: _PatchData,
    support: SupportModel | TimeIndexedSupport,
    resolved: ResolvedConfig,
) -> tuple[TimeBool, FloatArray]:
    """Detect one patch against one support; return ``(contact_mask, score)``."""
    clearance, normals = _evaluate_support(support, data.sample_points, data.points, data.target)
    angular = angular_speed_from_frames(data.frames, data.timestamps)
    features = compute_features(
        clearance=clearance,
        normals=normals,
        velocities=cast(TimeVec3, data.velocities),
        quiet=data.quiet,
        angular_speed=angular,
        contact_offset_max=resolved.contact_offset_max,
    )
    score = score_features(features, resolved)
    mask = score_hysteresis_mask(score, resolved.score_on, resolved.score_off)
    mask = clean_mask_by_time(data.timestamps, mask, resolved.max_gap_time, resolved.min_blip_time)
    mask = mask_from_intervals(
        data.timestamps, intervals_from_mask(data.timestamps, mask, min_duration=resolved.min_contact_time)
    )
    return mask, score


def _unknown_mask(data: _PatchData, timestamps: FloatArray, resolved: ResolvedConfig) -> TimeBool | None:
    """Per-patch untrustworthy-frame mask: sustained low coverage and/or jumps.

    Low-coverage runs are temporally cleaned (sustained dropouts only); jumps from
    the optional detector are kept as-is (brief but real garbage). Returns
    ``None`` when nothing is flagged, so reliably-tracked patches behave as before.
    """
    low_coverage = cast(TimeBool, data.coverage < resolved.min_coverage)
    invalid = np.asarray(
        clean_mask_by_time(
            timestamps, low_coverage, max_gap_time=resolved.max_gap_time, min_blip_time=resolved.unknown_min_duration
        )
    )
    if resolved.jumps is not None:
        invalid = invalid | np.asarray(resolved.jumps(data.points, timestamps), dtype=np.bool_)
    if not bool(np.any(invalid)):
        return None
    return cast(TimeBool, invalid)


@dataclass(frozen=True, slots=True)
class ContactDetector:
    """A reusable, config-bound contact detector.

    Construct once, then call :meth:`detect` (one support -> boolean
    :class:`ContactTrack`) across many scopes/supports. ``against`` is a
    per-call argument since it varies (an inferred floor vs a moving deck).
    """

    config: ContactDetectionConfig = ContactDetectionConfig()

    def detect(self, scope: Scope, *, against: Support = infer_ground()) -> ContactTrack:
        """Detect every geometry-bearing patch in ``scope`` against one support.

        ``against`` defaults to :func:`~retarget.contacts.infer_ground` (fit a floor
        from the motion); pass a ``ground_plane(h)``, another patch/segment, or your
        own :class:`SupportModel` instead.
        """
        resolved = self.config.resolve()
        data = [_gather_patch_data(patch, resolved) for patch in _normalize_scope(scope)]
        timestamps = data[0].timestamps
        support = _resolve_support(against, data, resolved)
        contacts: dict[PatchTarget, np.ndarray] = {}
        confidences: dict[PatchTarget, np.ndarray] = {}
        for item in data:
            mask, score = _detect_patch(item, support, resolved)
            contacts[item.target] = mask
            confidences[item.target] = score
        return ContactTrack(contacts=contacts, timestamps=timestamps, confidences=confidences)

    def classify(
        self,
        scope: Scope,
        *,
        against: Mapping[str, Support],
    ) -> SupportStateTrack:
        """Detect every patch in ``scope`` against many *named* supports.

        Returns a :class:`SupportStateTrack` carrying the per-support booleans
        (several may be True at once) plus scores; the categorical label is a
        read-time view via ``support_state(...)``. Use
        :func:`~retarget.contacts.infer_ground` for a fitted floor, e.g.
        ``against={"ground": infer_ground(), "board": deck_patch}``.

        Untrustworthy frames (low marker coverage sustained for
        ``unknown_min_duration``, plus optional ``config.jumps`` garbage samples)
        are flagged so ``support_state(unknown="...")`` can surface them. Labels
        are chosen by you at read time, not here.
        """
        if not against:
            raise ValueError("classify requires at least one named support in `against`")
        resolved = self.config.resolve()
        data = [_gather_patch_data(patch, resolved) for patch in _normalize_scope(scope)]
        timestamps = data[0].timestamps
        supports = {name: _resolve_support(spec, data, resolved) for name, spec in against.items()}
        contacts: dict[PatchTarget, dict[str, np.ndarray]] = {}
        scores: dict[PatchTarget, dict[str, np.ndarray]] = {}
        invalid: dict[PatchTarget, np.ndarray] = {}
        for item in data:
            per_contacts: dict[str, np.ndarray] = {}
            per_scores: dict[str, np.ndarray] = {}
            for name, support in supports.items():
                mask, score = _detect_patch(item, support, resolved)
                per_contacts[name] = mask
                per_scores[name] = score
            contacts[item.target] = per_contacts
            scores[item.target] = per_scores
            unknown_mask = _unknown_mask(item, timestamps, resolved)
            if unknown_mask is not None:
                invalid[item.target] = unknown_mask
        return SupportStateTrack(
            contacts=contacts,
            scores=scores,
            timestamps=timestamps,
            invalid=invalid,
        )


def detect_contacts(
    scope: Scope,
    *,
    against: Support = infer_ground(),
    config: ContactDetectionConfig = ContactDetectionConfig(),
) -> ContactTrack:
    """One-off convenience: ``ContactDetector(config).detect(scope, against=against)``."""
    return ContactDetector(config).detect(scope, against=against)


def classify_contacts(
    scope: Scope,
    *,
    against: Mapping[str, Support],
    config: ContactDetectionConfig = ContactDetectionConfig(),
) -> SupportStateTrack:
    """One-off convenience: ``ContactDetector(config).classify(scope, against=against)``."""
    return ContactDetector(config).classify(scope, against=against)


def merge_contact_tracks(*tracks: ContactTrack) -> ContactTrack:
    """OR-combine boolean contact tracks (max confidence) onto a shared timeline.

    Useful to fuse per-support detections (e.g. ground + skateboard) into a
    single "supported vs airborne" track.
    """
    if not tracks:
        raise ValueError("merge_contact_tracks requires at least one track")
    timestamps = np.asarray(tracks[0].timestamps, dtype=np.float64)
    for track in tracks[1:]:
        other = np.asarray(track.timestamps, dtype=np.float64)
        if len(other) != len(timestamps) or not np.allclose(other, timestamps):
            raise ValueError("contact tracks must share timestamps to merge")
    contacts: dict[PatchTarget, np.ndarray] = {}
    confidences: dict[PatchTarget, np.ndarray] = {}
    for track in tracks:
        for target, array in track.contacts.items():
            contacts[target] = array.copy() if target not in contacts else (contacts[target] | array)
        for target, array in track.confidences.items():
            confidences[target] = array.copy() if target not in confidences else np.maximum(confidences[target], array)
    return ContactTrack(contacts=contacts, timestamps=timestamps, confidences=confidences)
