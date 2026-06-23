"""Configuration for contact detection.

Detection scores each signal by how many measured-noise standard deviations it sits
from "resting contact" (a Mahalanobis distance), then turns that into a calibrated
**contact confidence** in ``[0, 1]``. So the public knobs are *confidence levels*, and
the per-channel noise scales come from a :class:`~retarget.contacts.noise.NoiseCalibration`
(measured from a static recording) rather than hand-tuned distances.

:class:`ContactDetectionConfig` exposes the slim public knobs; the full reference set
lives in the private :class:`_AdvancedContactConfig`. :meth:`ContactDetectionConfig.resolve`
collapses both into the flat :class:`ResolvedConfig` the driver consumes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType

from retarget.core.axes import AxisConvention, SemanticAxis, Z_UP_AXES
from retarget.core.formats import JumpDetector
from retarget.contacts._quiet import QuietParams
from retarget.contacts.noise import DEFAULT_NOISE, NoiseCalibration
from retarget.contacts.supports import SupportParams

CONTACT_CHANNELS = (
    "clearance",
    "normal_speed",
    "tangential_speed",
    "quiet_activity",
    "quiet_spread",
    "angular_speed",
)


class ConfidenceCombiner(StrEnum):
    """How per-channel residuals combine into one contact confidence."""

    CHI2 = "chi2"
    """Treat the summed squared residuals as chi-squared with one dof per channel."""
    FISHER = "fisher"
    """Combine per-channel tail probabilities via Fisher's method."""


def _default_channel_trust() -> Mapping[str, float]:
    return MappingProxyType({channel: 1.0 for channel in CONTACT_CHANNELS})


@dataclass(frozen=True, slots=True)
class _AdvancedContactConfig:
    """Full reference parameter set behind the slim public knobs."""

    channel_trust: Mapping[str, float] = field(default_factory=_default_channel_trust)
    """Per-channel variance inflation: 1.0 trusts the measured σ; <1 treats the channel
    as noisier (less influential). Replaces the old hand-tuned feature weights."""
    confidence_combiner: ConfidenceCombiner = ConfidenceCombiner.CHI2
    penetration_softening: float = 4.0
    """Penetration (clearance below the resting plane) is penalized this many times more
    gently than an equal gap -- squish / plane-fit error is benign, a gap is not."""
    max_resting_bias: float = 0.01
    """The resting-clearance bias (a constant geometry offset between the modeled patch
    and support planes) is clipped to ±this absolute distance (m). Big enough to absorb a
    real resting offset, small enough never to erase a real surface gap. This is a
    *systematic* offset scale, deliberately decoupled from the (random) noise σ."""
    confidence_release_ratio: float = 0.85
    """Exit confidence = contact_confidence · this, unless release_confidence is set."""
    min_blip_time: float = 0.15
    unknown_min_duration: float = 0.20
    quiet: QuietParams = field(default_factory=QuietParams)
    support: SupportParams = field(default_factory=SupportParams)


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """Flat, fully-resolved detection parameters consumed by the driver."""

    noise: NoiseCalibration
    channel_trust: Mapping[str, float]
    confidence_combiner: ConfidenceCombiner
    penetration_softening: float
    max_resting_bias: float
    enter_confidence: float
    exit_confidence: float
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
        contact_confidence: Confidence in ``(0, 1)`` required to *enter* contact. It is
            the probability that residuals this small would arise from sensor noise if the
            patch were genuinely resting in contact; higher = stricter.
        release_confidence: Confidence to *exit* contact (hysteresis). ``None`` derives it
            from ``contact_confidence`` via ``advanced.confidence_release_ratio``. Must be
            ``<= contact_confidence``.
        noise: Measured per-channel noise scales (see
            :func:`~retarget.contacts.noise.estimate_noise`). ``None`` uses
            :data:`~retarget.contacts.noise.DEFAULT_NOISE` (typical optical mocap).
        min_contact_time: Shortest kept contact interval (s).
        max_gap_time: Brief separations up to this long (s) are bridged.
        min_coverage: Frames whose marker coverage falls below this are flagged
            untrustworthy, so ``support_state(unknown="...")`` can surface them.
        jumps: Optional garbage-sample detector; flagged teleports count as untrustworthy.
        up_axis: Semantic up direction used for inferred floors and plane normals.
        axis_convention: Maps semantic axes to concrete world axes.
        region_contact: Use the whole patch footprint (closest approach) rather than the
            center point only.
        advanced: Full reference parameter set; rarely overridden.
    """

    contact_confidence: float = 0.95
    release_confidence: float | None = None
    noise: NoiseCalibration | None = None
    min_contact_time: float = 0.18
    max_gap_time: float = 0.10
    min_coverage: float = 0.50
    jumps: JumpDetector | None = None
    up_axis: SemanticAxis = SemanticAxis.UP
    axis_convention: AxisConvention = Z_UP_AXES
    region_contact: bool = True
    advanced: _AdvancedContactConfig = field(default_factory=_AdvancedContactConfig)

    def __post_init__(self) -> None:
        for name in ("min_contact_time", "max_gap_time"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.contact_confidence < 1.0:
            raise ValueError("contact_confidence must be in (0, 1)")
        if self.release_confidence is not None and not 0.0 < self.release_confidence <= self.contact_confidence:
            raise ValueError("release_confidence must be in (0, contact_confidence]")
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
        exit_confidence = (
            self.release_confidence
            if self.release_confidence is not None
            else self.contact_confidence * advanced.confidence_release_ratio
        )
        return ResolvedConfig(
            noise=self.noise if self.noise is not None else NoiseCalibration(default=DEFAULT_NOISE),
            channel_trust=advanced.channel_trust,
            confidence_combiner=advanced.confidence_combiner,
            penetration_softening=advanced.penetration_softening,
            max_resting_bias=advanced.max_resting_bias,
            enter_confidence=self.contact_confidence,
            exit_confidence=exit_confidence,
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
