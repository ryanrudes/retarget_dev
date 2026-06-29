"""The contact-detection driver.

:class:`ContactDetector` binds a :class:`ContactDetectionConfig` and detects
contacts for the patches in a *scope* (a ``Patch``/``Segment``/``Subject``/
``MocapTrack`` or an iterable of patches) against a *support* (an inferred floor
when ``against`` is ``None``; a provided ``SupportModel``; or a moving support
derived from another patch/segment). Each patch is detected independently and
keyed by its :class:`~retarget.core.targets.PatchTarget`.

Against a *moving* support, motion features are measured relative to the support
(see :func:`_support_relative_motion`), so a patch riding along with it -- a shoe
resting on a gliding skateboard -- reads as at rest rather than "in the air".

``detect_contacts`` is a thin one-off convenience that constructs a detector and
delegates, so a single implementation lives in the class.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from retarget.contacts._features import Features, angular_speed_from_frames, compute_features
from retarget.contacts._quiet import QuietResult, detect_quiet, local_polynomial_derivative, score_hysteresis_mask
from retarget.contacts._scope import Scope, normalize_scope
from retarget.contacts._scoring import channel_residuals, confidence_features
from retarget.contacts.config import ContactDetectionConfig, ResolvedConfig
from retarget.contacts.noise import ChannelNoise
from retarget.contacts.state import SupportStateTrack
from retarget.contacts.supports import (
    InferredGround,
    SupportModel,
    TimeIndexedSupport,
    _up_vector,
    infer_ground,
    infer_support_surface,
)
from retarget.core.geometry import face_signal, sampling_at
from retarget.core.schema.patch import Patch
from retarget.core.schema.segment import Segment
from retarget.core.targets import PatchTarget
from retarget.core.types import FloatArray, FloatArray1D, Points3, TimeBool, TimeMat3, TimeVec3
from retarget.demo.contact import ContactTrack
from retarget.fungeom.signals import point_bundle_signal
from retarget.utils.intervals import clean_mask_by_time, intervals_from_mask, mask_from_intervals

if TYPE_CHECKING:
    from fungeom import Face

Support = SupportModel | TimeIndexedSupport | Patch | Segment[Any, Any] | InferredGround


@dataclass(frozen=True, slots=True)
class _FaceSupport:
    """A moving patch support that carries its fungeom ``Face`` for *bounded* clearance.

    Unlike :class:`~retarget.contacts.supports.TimeIndexedSupport` (an infinite plane per
    frame), the support's footprint edge is respected: a query point off the edge reads as a
    gap, not contact. ``origins``/``normals``/``frames`` mirror ``TimeIndexedSupport`` so
    support-relative motion/spin are computed identically; only clearance differs.
    """

    face: Face
    timestamps: FloatArray
    translations: TimeVec3
    rotations: TimeMat3
    origins: TimeVec3
    normals: TimeVec3
    frames: TimeMat3
    name: str


# A support resolved to its per-detection form: a static model, a moving plane, or a moving
# Face (bounded). Motion treats the two moving forms alike; only clearance distinguishes them.
_ResolvedSupport = SupportModel | TimeIndexedSupport | _FaceSupport


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


def _patch_timestamps(patch: Patch) -> FloatArray:
    # The bound runtime carries the (possibly sliced) mocap timeline shared by the scope.
    return cast(FloatArray, np.asarray(patch._runtime().timestamps, dtype=np.float64))


def _gather_patch_data(patch: Patch, resolved: ResolvedConfig) -> _PatchData:
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
    if resolved.region_contact and patch.has_region():
        corners = np.asarray(patch.boundary_points(), dtype=np.float64)  # (T, K, 3)
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


def _moving_support_from(against: Patch | Segment[Any, Any], up_axis: int) -> _ResolvedSupport:
    """Build a per-frame moving support from a tracked patch (deck surface) or segment pose.

    A :class:`~retarget.core.schema.patch.Patch` support carries its fungeom ``Face`` as a
    :class:`_FaceSupport` so clearance respects the footprint edge; a ``Segment`` support has no
    Face and becomes a moving plane (:class:`~retarget.contacts.supports.TimeIndexedSupport`). In
    both cases ``frames`` is the backing body's full ``(T, 3, 3)`` orientation, so motion can be
    measured relative to the support's rotation, not just its translation (see
    :func:`_support_relative_angular_speed`).
    """
    if isinstance(against, Patch):
        runtime = against._runtime()
        return _FaceSupport(
            face=against.face(),
            timestamps=cast(FloatArray, np.asarray(runtime.timestamps, dtype=np.float64)),
            translations=cast(TimeVec3, np.asarray(runtime.translations, dtype=np.float64)),
            rotations=cast(TimeMat3, np.asarray(runtime.rotations, dtype=np.float64)),
            origins=cast(TimeVec3, np.asarray(against.points(), dtype=np.float64)),
            normals=cast(TimeVec3, np.asarray(against.normals(), dtype=np.float64)),
            frames=cast(TimeMat3, np.asarray(against.frames(), dtype=np.float64)),
            name=against.target.patch,
        )
    runtime_rotations = cast(TimeMat3, np.asarray(against.rotations(), dtype=np.float64))
    origins = cast(TimeVec3, np.asarray(against.translations(), dtype=np.float64))
    up = _up_vector(up_axis)
    normals = cast(TimeVec3, np.einsum("tij,j->ti", runtime_rotations, up))
    return TimeIndexedSupport(
        origins=origins, normals=normals, name=against.segment_target().segment, frames=runtime_rotations
    )


def _resolve_support(
    against: Support,
    data: list[_PatchData],
    resolved: ResolvedConfig,
) -> _ResolvedSupport:
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


def _face_support_clearance(support: _FaceSupport, sample_points: FloatArray, target: PatchTarget) -> FloatArray:
    """Signed, *bounded* closest-approach clearance of a footprint to a moving Face support.

    Per footprint sample: the signed perpendicular distance to the support plane plus the lateral
    overhang beyond the footprint, ``perp + sqrt(max(0, bounded^2 - perp^2))``. Inside the
    footprint ``bounded == |perp|`` so this is the plain perpendicular clearance (matching the
    infinite-plane support); off the edge the overhang is added as a positive gap, so a sample off
    the support never reads as contact. fungeom's ``clearance`` is unsigned, hence the re-signing
    with the signed perpendicular distance. Occluded samples (``present`` is False -> NaN) cannot
    be the closest approach and are dropped; a fully occluded frame is NaN.
    """
    n_time, n_samples, _ = sample_points.shape
    if len(support.timestamps) != n_time:
        raise ValueError(
            f"moving support length {len(support.timestamps)} does not match patch "
            f"{target.patch!r} length {n_time} (different timelines?)"
        )
    face = face_signal(
        support.face, support.timestamps, cast(FloatArray, support.translations), cast(FloatArray, support.rotations)
    )
    sampling = sampling_at(support.timestamps)
    bundle = point_bundle_signal(support.timestamps, sample_points, tuple(str(i) for i in range(n_samples)))
    perp_values, present = face.plane().signed_distance(bundle).resolve_over(sampling)
    bounded_values, _ = face.clearance(bundle).resolve_over(sampling)
    perp = np.asarray(perp_values, dtype=np.float64)  # signed perpendicular distance (T, K)
    bounded = np.asarray(bounded_values, dtype=np.float64)  # unsigned bounded distance (T, K)
    signed = perp + np.sqrt(np.maximum(0.0, bounded * bounded - perp * perp))
    signed = np.where(np.asarray(present, dtype=bool), signed, np.inf)  # occluded can't be closest
    clearance = signed.min(axis=1)
    return cast(FloatArray, np.where(np.isinf(clearance), np.nan, clearance))


def _evaluate_support(
    support: _ResolvedSupport,
    sample_points: FloatArray,
    center_points: FloatArray,
    target: PatchTarget,
) -> tuple[FloatArray, Points3]:
    """Return ``(closest-approach clearance (T,), normals (T, 3))`` of the footprint.

    Clearance is the minimum over the ``K`` footprint samples in ``sample_points`` ``(T, K, 3)``
    (the rectangle corners + center; ``K=1`` for point-based), so a patch reads as touching when
    its nearest part does. A :class:`_FaceSupport` measures *bounded* clearance (the support's
    footprint edge is respected); other supports measure the infinite-plane / model clearance. The
    support normal is read at ``center_points`` ``(T, 3)``.
    """
    n_time, n_samples, _ = sample_points.shape
    if isinstance(support, _FaceSupport):
        return _face_support_clearance(support, sample_points, target), support.normals
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


def _support_relative_motion(
    data: _PatchData,
    support: _ResolvedSupport,
    resolved: ResolvedConfig,
) -> tuple[FloatArray, QuietResult]:
    """Motion features (velocity + quiet) measured relative to the support.

    A patch resting on a *moving* support (e.g. a rolling skateboard deck) rides
    with it: during the glide its world-frame velocity is large even though it is
    motionless *relative to the deck*, with zero clearance throughout. Scoring
    contact off world motion then drives ``tangential_speed`` and the quiet
    activity/spread up and collapses the score, so the glide wrongly reads as
    "air". For a moving :class:`~retarget.contacts.supports.TimeIndexedSupport`
    we therefore measure motion against the support: the patch trajectory minus
    the support origin, so a co-moving patch reads as at rest and the support's
    translation is subtracted out.

    Static supports do not move, so their support-relative motion equals the
    world motion already gathered -- speed and windowed range are invariant to a
    constant origin shift -- and we reuse it unchanged (zero behavior change for
    inferred/ground/heightmap supports). The rotational channel is handled the same
    way by :func:`_support_relative_angular_speed`.
    """
    if not isinstance(support, (TimeIndexedSupport, _FaceSupport)):
        return data.velocities, data.quiet
    relative_points = data.points - np.asarray(support.origins, dtype=np.float64)
    velocities = local_polynomial_derivative(
        relative_points,
        data.timestamps,
        resolved.quiet.derivative_window_time,
        resolved.quiet.derivative_poly_degree,
    )
    quiet = detect_quiet(relative_points, data.timestamps, resolved.quiet)
    return velocities, quiet


def _support_relative_angular_speed(
    data: _PatchData,
    support: _ResolvedSupport,
    resolved: ResolvedConfig,
) -> FloatArray:
    """Angular speed of the patch measured relative to the support's orientation.

    The contact-relevant quantity is how fast the patch spins *against its support*,
    not in the world. A patch rigidly fixed to a rolling/tilting deck inherits the
    deck's world-frame spin (a balance board lifted and rotated in space), which would
    drive ``angular_speed`` up and collapse the score even though the contact never
    breaks. We therefore express the patch orientation in the support frame,
    ``R_rel = R_supportᵀ R_patch``, and differentiate that -- a co-rotating patch reads
    as ~at rest.

    This is the rotational twin of :func:`_support_relative_motion`. A support with no
    orientation (``frames is None``: a bare plane, or any static ``SupportModel``) is
    the ``ω_support = 0`` case: ``R_rel`` differs from ``R_patch`` only by a constant
    left factor, so its angular speed equals the world-frame angular speed exactly --
    same single formula, no separate path, zero behavior change for planar supports.
    """
    window = resolved.quiet.derivative_window_time
    degree = resolved.quiet.derivative_poly_degree
    if not isinstance(support, (TimeIndexedSupport, _FaceSupport)) or support.frames is None:
        return angular_speed_from_frames(data.frames, data.timestamps, window, degree)
    support_frames = np.asarray(support.frames, dtype=np.float64)
    relative_frames = cast(TimeMat3, np.einsum("tji,tjk->tik", support_frames, data.frames))
    return angular_speed_from_frames(relative_frames, data.timestamps, window, degree)


def _patch_features(
    data: _PatchData,
    support: _ResolvedSupport,
    resolved: ResolvedConfig,
) -> tuple[Features, ChannelNoise]:
    """Compute one patch's per-frame feature channels against one support, plus its noise.

    The single place support-relative clearance/motion/spin become the scorer's
    :class:`Features`; shared by :func:`_detect_patch` and the
    :func:`classify_feature_channels` diagnostic so they can never drift apart.
    """
    noise = resolved.noise.for_patch(data.target)
    clearance, normals = _evaluate_support(support, data.sample_points, data.points, data.target)
    velocities, quiet = _support_relative_motion(data, support, resolved)
    angular = _support_relative_angular_speed(data, support, resolved)
    features = compute_features(
        clearance=clearance,
        normals=normals,
        velocities=cast(TimeVec3, velocities),
        quiet=quiet,
        angular_speed=angular,
        max_resting_bias=resolved.max_resting_bias,
    )
    return features, noise


def _detect_patch(
    data: _PatchData,
    support: _ResolvedSupport,
    resolved: ResolvedConfig,
) -> tuple[TimeBool, FloatArray]:
    """Detect one patch against one support; return ``(contact_mask, confidence)``."""
    features, noise = _patch_features(data, support, resolved)
    confidence = confidence_features(features, noise, resolved)
    mask = score_hysteresis_mask(confidence, resolved.enter_confidence, resolved.exit_confidence)
    mask = clean_mask_by_time(data.timestamps, mask, resolved.max_gap_time, resolved.min_blip_time)
    mask = mask_from_intervals(
        data.timestamps, intervals_from_mask(data.timestamps, mask, min_duration=resolved.min_contact_time)
    )
    return mask, confidence


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
        data = [_gather_patch_data(patch, resolved) for patch in normalize_scope(scope)]
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
        data = [_gather_patch_data(patch, resolved) for patch in normalize_scope(scope)]
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


# The detection signals, in the order they read best top-to-bottom: where the patch sits
# (clearance), how it moves against the surface (the speeds), how quiet it is, then spin.
CONTACT_CHANNELS: tuple[str, ...] = (
    "clearance",
    "normal_speed",
    "tangential_speed",
    "quiet_activity",
    "quiet_spread",
    "angular_speed",
)


@dataclass(frozen=True, slots=True)
class SupportFeatureChannels:
    """The per-frame signals behind one patch-vs-one-support contact decision.

    ``raw`` is each channel's physical value (metres, m/s, rad/s, …). ``residual`` is its
    *trust-weighted standardized residual* in σ units -- exactly what the χ² scorer squares
    and sums, so a channel reading ≫1 σ is what drove the confidence down at that frame.
    ``confidence`` is the fused contact probability in ``[0, 1]``. Built by
    :func:`classify_feature_channels` for explainability and visualization; detection itself
    does not need it. Channels are keyed by :data:`CONTACT_CHANNELS`.
    """

    timestamps: FloatArray
    raw: Mapping[str, FloatArray]
    residual: Mapping[str, FloatArray]
    confidence: FloatArray


def _patch_support_channels(
    data: _PatchData, support: _ResolvedSupport, resolved: ResolvedConfig
) -> SupportFeatureChannels:
    """Build the explainability channels for one patch against one support."""
    features, noise = _patch_features(data, support, resolved)
    trust = resolved.channel_trust
    residuals = channel_residuals(features, noise, resolved)
    raw: dict[str, FloatArray] = {
        "clearance": features.clearance,
        "normal_speed": features.normal_speed,
        "tangential_speed": features.tangential_speed,
        "quiet_activity": features.quiet_activity,
        "quiet_spread": features.quiet_spread,
        "angular_speed": features.angular_speed,
    }
    # Fold the channel's trust weight into the residual so the σ value shown is its actual
    # contribution to the penalty (sqrt(trust) · z, since the scorer uses trust · z²).
    weighted: dict[str, FloatArray] = {
        name: cast(FloatArray, np.sqrt(trust.get(name, 1.0)) * residuals[name]) for name in CONTACT_CHANNELS
    }
    return SupportFeatureChannels(
        timestamps=data.timestamps,
        raw=MappingProxyType(raw),
        residual=MappingProxyType(weighted),
        confidence=confidence_features(features, noise, resolved),
    )


def classify_feature_channels(
    scope: Scope,
    *,
    against: Mapping[str, Support],
    config: ContactDetectionConfig = ContactDetectionConfig(),
) -> dict[PatchTarget, dict[str, SupportFeatureChannels]]:
    """Per-patch, per-named-support detection signals -- the *why* behind ``classify``.

    Returns ``{patch_target: {support_name: SupportFeatureChannels}}`` using the identical
    feature/scoring pipeline as :meth:`ContactDetector.classify`, so the channels explain
    exactly the same decision. Use it to visualize or debug which signal moved the
    confidence (e.g. a foot riding a rolling board: clearance ~0 but world-frame spin high).
    """
    if not against:
        raise ValueError("classify_feature_channels requires at least one named support in `against`")
    resolved = config.resolve()
    data = [_gather_patch_data(patch, resolved) for patch in normalize_scope(scope)]
    supports = {name: _resolve_support(spec, data, resolved) for name, spec in against.items()}
    return {
        item.target: {name: _patch_support_channels(item, support, resolved) for name, support in supports.items()}
        for item in data
    }
