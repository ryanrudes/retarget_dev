"""Fuse per-channel residuals into a calibrated contact confidence in ``[0, 1]``.

Each feature is divided by its measured noise σ to form a standardized residual; under
"resting contact" the residuals are ~N(0, 1). The summed squared residuals are a
squared Mahalanobis distance, distributed χ² with one degree of freedom per channel, so
the **contact confidence** is the χ² survival probability ``P(χ²_k ≥ Σ residual²)`` -- the
chance that residuals this small would arise from sensor noise if the patch were truly in
contact. It is ~0.5 for a resting frame and collapses to 0 as any channel departs by many
σ (a real gap or real motion). Clearance is one-sided: a gap is penalized fully, a
penetration (squish / plane-fit error) far more gently.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.stats import chi2

from retarget.contacts._features import Features
from retarget.contacts.config import ConfidenceCombiner, ResolvedConfig
from retarget.contacts.noise import ChannelNoise
from retarget.core.types import FloatArray


def _residual(value: FloatArray, sigma: float) -> FloatArray:
    safe = sigma if sigma > 1e-12 else 1.0
    return np.asarray(value, dtype=np.float64) / safe


def channel_residuals(features: Features, noise: ChannelNoise, config: ResolvedConfig) -> dict[str, FloatArray]:
    """Per-channel standardized residual ``z`` ``(T,)`` -- value over its noise σ.

    These are the raw (pre-trust) standardized residuals the scorer squares and sums. The
    clearance channel is one-sided: gaps (``z > 0``) keep full strength while penetration
    (``z < 0``) is softened by ``penetration_softening``, so a squish/plane-fit error is
    forgiven far more gently than a real gap. The single source of truth for both
    :func:`confidence_features` and the diagnostics in
    :func:`~retarget.contacts.detect.classify_feature_channels`.
    """
    kappa = max(config.penetration_softening, 1e-6)
    z_clear = _residual(features.clearance, noise.clearance_sigma)
    z_clear = np.where(z_clear < 0.0, z_clear / kappa, z_clear)
    return {
        "clearance": z_clear,
        "normal_speed": _residual(features.normal_speed, noise.normal_speed_sigma),
        "tangential_speed": _residual(features.tangential_speed, noise.tangential_speed_sigma),
        "quiet_activity": _residual(features.quiet_activity, noise.quiet_activity_sigma),
        "quiet_spread": _residual(features.quiet_spread, noise.quiet_spread_sigma),
        "angular_speed": _residual(features.angular_speed, noise.angular_speed_sigma),
    }


def confidence_features(features: Features, noise: ChannelNoise, config: ResolvedConfig) -> FloatArray:
    """Map per-frame features to a calibrated contact confidence in ``[0, 1]`` ``(T,)``."""
    trust = config.channel_trust
    residuals = channel_residuals(features, noise, config)
    squared = {name: trust.get(name, 1.0) * np.square(r) for name, r in residuals.items()}
    k = len(squared)

    if config.confidence_combiner is ConfidenceCombiner.FISHER:
        stat = np.zeros_like(residuals["clearance"], dtype=np.float64)
        for r2 in squared.values():
            p = np.clip(chi2.sf(r2, df=1), 1e-300, 1.0)
            stat = stat - 2.0 * np.log(p)
        confidence = chi2.sf(stat, df=2 * k)
    else:
        penalty = np.sum(list(squared.values()), axis=0)
        confidence = chi2.sf(penalty, df=k)

    return cast(FloatArray, np.clip(confidence, 0.0, 1.0))
