"""Configuration for contact detection.

:class:`ContactDetectionConfig` exposes ~6 intuitive knobs with good defaults;
the full reference parameter set (feature weights, quiet smoothing windows,
RANSAC settings, …) lives in the private :class:`_AdvancedContactConfig` and is
rarely touched. :meth:`ContactDetectionConfig.resolve` collapses both into the
flat :class:`ResolvedConfig` the driver consumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.formats import JumpDetector
from retarget.contacts._quiet import QuietParams
from retarget.contacts.supports import SupportParams


def _default_feature_weights() -> Mapping[str, float]:
    return MappingProxyType(
        {
            "clearance": 1.50,
            "normal_speed": 0.75,
            "tangential_speed": 1.25,
            "angular_speed": 0.25,
            "quiet_activity": 0.75,
            "quiet_spread": 0.50,
        }
    )


@dataclass(frozen=True, slots=True)
class _AdvancedContactConfig:
    """Full reference parameter set behind the slim public knobs."""

    feature_weights: Mapping[str, float] = field(default_factory=_default_feature_weights)
    angular_speed_scale: float = 1.00
    contact_offset_max: float | None = None
    min_blip_time: float = 0.15
    off_ratio: float = 0.6
    unknown_min_duration: float = 0.20
    quiet: QuietParams = field(default_factory=QuietParams)
    support: SupportParams = field(default_factory=SupportParams)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Flat, fully-resolved detection parameters consumed by the driver."""

    clearance_scale: float
    normal_speed_scale: float
    tangential_speed_scale: float
    angular_speed_scale: float
    contact_offset_max: float
    weights: Mapping[str, float]
    score_on: float
    score_off: float
    min_contact_time: float
    max_gap_time: float
    min_blip_time: float
    min_coverage: float
    jumps: JumpDetector | None
    unknown_min_duration: float
    up_axis: int
    region_contact: bool
    quiet: QuietParams
    support: SupportParams


@dataclass(frozen=True, slots=True)
class ContactDetectionConfig:
    """Tunable knobs for automatic contact detection.

    Attributes:
        contact_clearance: How close (m) along the support normal counts as touching.
        quiet_speed: Speed scale (m/s) for the normal/tangential motion penalties.
        min_contact_time: Shortest kept contact interval (s).
        max_gap_time: Brief separations up to this long (s) are bridged.
        sensitivity: Hysteresis on-threshold in ``(0, 1)``; off-threshold is a fixed
            fraction of it. Higher = stricter (fewer, more confident contacts).
        min_coverage: Trust threshold -- frames whose marker coverage (fraction of
            the segment's markers observed) falls below this are flagged
            untrustworthy, so ``support_state(unknown="...")`` can surface them.
        jumps: Optional garbage-sample detector (``speed_limit`` / ``robust_jumps``
            / your own); flagged teleports also count as untrustworthy. ``None``
            = coverage only.
        up_axis: Semantic up direction used for the inferred floor and plane normals.
        axis_convention: Maps semantic axes to concrete world axes.
        region_contact: Use the whole patch footprint (the rectangle's closest
            approach to the support) rather than only the center point, so a patch
            counts as touching when any part of it does. ``False`` = point-based.
        advanced: Full reference parameter set; rarely overridden.
    """

    contact_clearance: float = 0.03
    quiet_speed: float = 0.08
    min_contact_time: float = 0.18
    max_gap_time: float = 0.10
    sensitivity: float = 0.60
    min_coverage: float = 0.50
    jumps: JumpDetector | None = None
    up_axis: SemanticAxis = SemanticAxis.UP
    axis_convention: AxisConvention = Z_UP_AXES
    region_contact: bool = True
    advanced: _AdvancedContactConfig = field(default_factory=_AdvancedContactConfig)

    def __post_init__(self) -> None:
        for name in ("contact_clearance", "quiet_speed", "min_contact_time", "max_gap_time"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.sensitivity < 1.0:
            raise ValueError("sensitivity must be in (0, 1)")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")

    @property
    def up_axis_index(self) -> int:
        """Concrete world-axis index (0/1/2) for the configured semantic up axis."""
        return int(self.axis_convention.signed_axis(self.up_axis).axis)

    def resolve(self) -> ResolvedConfig:
        """Collapse public knobs and the advanced bundle into a flat config."""
        advanced = self.advanced
        up_index = self.up_axis_index
        offset_max = (
            advanced.contact_offset_max
            if advanced.contact_offset_max is not None
            else 1.5 * self.contact_clearance
        )
        return ResolvedConfig(
            clearance_scale=self.contact_clearance,
            normal_speed_scale=self.quiet_speed,
            tangential_speed_scale=self.quiet_speed,
            angular_speed_scale=advanced.angular_speed_scale,
            contact_offset_max=offset_max,
            weights=advanced.feature_weights,
            score_on=self.sensitivity,
            score_off=self.sensitivity * advanced.off_ratio,
            min_contact_time=self.min_contact_time,
            max_gap_time=self.max_gap_time,
            min_blip_time=advanced.min_blip_time,
            min_coverage=self.min_coverage,
            jumps=self.jumps,
            unknown_min_duration=advanced.unknown_min_duration,
            up_axis=up_index,
            region_contact=self.region_contact,
            quiet=advanced.quiet,
            support=replace(advanced.support, up_axis=up_index),
        )
