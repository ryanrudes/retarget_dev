"""Fuse support-relative features into a per-frame contact score in ``[0, 1]``.

The score is ``exp(-0.5 * penalty)`` where the penalty is a weighted sum of
squared feature/scale ratios -- small clearances and quiet, slow motion give a
low penalty and a high contact score. Ported from the reference pipeline.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from retarget.contacts._features import Features
from retarget.contacts.config import ResolvedConfig
from retarget.core.types import FloatArray


def _squared_ratio(value: FloatArray, scale: float) -> FloatArray:
    safe = scale if scale > 1e-12 else 1.0
    return (value / safe) ** 2


def score_features(features: Features, config: ResolvedConfig) -> FloatArray:
    """Map per-frame features to contact scores in ``[0, 1]`` with shape ``(T,)``."""
    weights = config.weights
    penalty = np.zeros_like(features.abs_clearance, dtype=np.float64)
    penalty += weights.get("clearance", 0.0) * _squared_ratio(features.abs_clearance, config.clearance_scale)
    penalty += weights.get("normal_speed", 0.0) * _squared_ratio(features.normal_speed, config.normal_speed_scale)
    penalty += weights.get("tangential_speed", 0.0) * _squared_ratio(
        features.tangential_speed, config.tangential_speed_scale
    )
    penalty += weights.get("quiet_activity", 0.0) * _squared_ratio(
        features.quiet_activity, features.quiet_activity_scale
    )
    penalty += weights.get("quiet_spread", 0.0) * _squared_ratio(
        features.quiet_spread, features.quiet_spread_scale
    )
    penalty += weights.get("angular_speed", 0.0) * _squared_ratio(features.angular_speed, config.angular_speed_scale)
    return cast(FloatArray, np.clip(np.exp(-0.5 * penalty), 0.0, 1.0))
