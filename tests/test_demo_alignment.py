from __future__ import annotations

import numpy as np
import pytest

from demo_vocab import GroundEstimationTrackId
from retarget.core.enums import TrackId
from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    TrackAlignment,
    estimate_alignment_from_signals,
)
from retarget.demo.demo import Demonstration
from conftest import make_mocap_track


class DemoTrackId(TrackId):
    SOURCE = "source"
    REFERENCE = "reference"


# --- EnergySignal ---


def test_energy_signal_accepts_scalar_1d_values() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0, 0.1, 0.2]),
        values=np.array([1.0, 2.0, 3.0]),
        name="contact_energy",
    )

    assert signal.name == "contact_energy"
    assert signal.timestamps.dtype == np.float64
    assert signal.values.dtype == np.float64


def test_energy_signal_allows_missing_name() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0]),
        values=np.array([1.0]),
    )

    assert signal.name is None


def test_energy_signal_coerces_name_to_string() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0]),
        values=np.array([1.0]),
        name=123,
    )

    assert signal.name == "123"


@pytest.mark.parametrize(
    ("timestamps", "values", "match"),
    [
        (
            np.array([[0.0, 0.1]]),
            np.array([1.0, 2.0]),
            "timestamps must be a 1D array",
        ),
        (
            np.array([0.0, 0.1]),
            np.array([[1.0, 2.0]]),
            "values must be a 1D scalar signal",
        ),
        (
            np.array([0.0, 0.1]),
            np.array([1.0]),
            "matching shape",
        ),
    ],
)
def test_energy_signal_rejects_invalid_shapes(
    timestamps: np.ndarray,
    values: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        EnergySignal(timestamps=timestamps, values=values)


@pytest.mark.parametrize(
    ("timestamps", "values", "match"),
    [
        (
            np.array([0.0, 0.0]),
            np.array([1.0, 2.0]),
            "strictly increasing",
        ),
        (
            np.array([0.0, np.inf]),
            np.array([1.0, 2.0]),
            "timestamps must be finite",
        ),
        (
            np.array([0.0, 0.1]),
            np.array([1.0, np.nan]),
            "values must be finite",
        ),
    ],
)
def test_energy_signal_rejects_invalid_values(
    timestamps: np.ndarray,
    values: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        EnergySignal(timestamps=timestamps, values=values)


def test_energy_signal_repr_with_name() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0, 0.1, 0.2]),
        values=np.array([1.0, 2.0, 3.0]),
        name="contact_energy",
    )

    text = repr(signal)

    assert "EnergySignal(" in text
    assert "name='contact_energy'" in text
    assert "samples=3" in text
    assert "start=np.float64(0.0)" not in text
    assert "start=0.0" in text
    assert "stop=0.2" in text
    assert "mean=" in text
    assert "std=" in text
    assert text.endswith(")")


def test_energy_signal_repr_empty() -> None:
    signal = EnergySignal(
        timestamps=np.array([], dtype=np.float64),
        values=np.array([], dtype=np.float64),
    )

    text = repr(signal)

    assert "samples=0" in text
    assert "start=None" in text
    assert "stop=None" in text
    assert "mean=" not in text
    assert text.endswith(")")


def test_energy_signal_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="EnergySignal timestamps must be strictly increasing"):
        EnergySignal(
            timestamps=np.array([0.0, 0.1, 0.1]),
            values=np.array([0.0, 1.0, 2.0]),
        )


# --- TimelineTransform ---


def test_timeline_transform_identity() -> None:
    transform = TimelineTransform.identity()
    times = np.array([0.0, 1.0, 2.0])

    np.testing.assert_allclose(transform.to_reference(times), times)
    np.testing.assert_allclose(transform.to_source(times), times)


def test_timeline_transform_to_reference_and_to_source() -> None:
    transform = TimelineTransform(scale=2.0, offset=-0.5)

    source_times = np.array([0.0, 1.0, 2.0])
    reference_times = np.array([-0.5, 1.5, 3.5])

    np.testing.assert_allclose(transform.to_reference(source_times), reference_times)
    np.testing.assert_allclose(transform.to_source(reference_times), source_times)


def test_timeline_transform_accepts_scalar_time() -> None:
    transform = TimelineTransform(scale=2.0, offset=1.0)

    assert transform.to_reference(3.0) == 7.0
    assert transform.to_source(7.0) == 3.0


def test_timeline_transform_inverse() -> None:
    transform = TimelineTransform(scale=2.0, offset=-0.5)
    inverse = transform.inverse()

    times = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        inverse.to_reference(transform.to_reference(times)),
        times,
    )


def test_timeline_transform_then_composes_in_order() -> None:
    first = TimelineTransform(scale=2.0, offset=1.0)
    second = TimelineTransform(scale=3.0, offset=-4.0)

    composed = first.then(second)

    times = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        composed.to_reference(times),
        second.to_reference(first.to_reference(times)),
    )


@pytest.mark.parametrize(
    ("scale", "offset", "match"),
    [
        (np.inf, 0.0, "scale must be finite"),
        (1.0, np.nan, "offset must be finite"),
    ],
)
def test_timeline_transform_rejects_non_finite_values(
    scale: float,
    offset: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        TimelineTransform(scale=scale, offset=offset)


def test_timeline_transform_zero_scale_to_source_raises() -> None:
    transform = TimelineTransform(scale=0.0, offset=1.0)

    with pytest.raises(ZeroDivisionError, match="zero-scale"):
        transform.to_source(1.0)


def test_timeline_transform_zero_scale_inverse_raises() -> None:
    transform = TimelineTransform(scale=0.0, offset=1.0)

    with pytest.raises(ZeroDivisionError, match="zero-scale"):
        transform.inverse()


def test_timeline_transform_round_trip() -> None:
    transform = TimelineTransform(scale=1.0, offset=0.5)
    source_times = np.array([0.0, 1.0, 2.0])
    reference_times = transform.to_reference(source_times)
    recovered = transform.to_source(reference_times)
    np.testing.assert_allclose(recovered, source_times)


# --- TrackAlignment ---


def test_track_alignment_stores_fields() -> None:
    alignment = TrackAlignment(
        source=DemoTrackId.SOURCE,
        reference=DemoTrackId.REFERENCE,
        transform=TimelineTransform(scale=1.0, offset=-0.2),
        score=0.8,
    )

    assert alignment.source is DemoTrackId.SOURCE
    assert alignment.reference is DemoTrackId.REFERENCE
    assert alignment.transform.offset == -0.2
    assert alignment.score == 0.8


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_track_alignment_rejects_non_finite_score(score: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        TrackAlignment(
            source=DemoTrackId.SOURCE,
            reference=DemoTrackId.REFERENCE,
            transform=TimelineTransform.identity(),
            score=score,
        )


def test_track_alignment_allows_missing_score() -> None:
    alignment = TrackAlignment(
        source=DemoTrackId.SOURCE,
        reference=DemoTrackId.REFERENCE,
        transform=TimelineTransform.identity(),
    )

    assert alignment.score is None


# --- estimate_alignment_from_signals ---


def test_estimate_alignment_returns_identity_for_too_short_signals() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0]),
        values=np.array([1.0]),
    )

    transform, score = estimate_alignment_from_signals(
        reference=signal,
        source=signal,
        max_lag_seconds=1.0,
    )

    assert transform == TimelineTransform.identity()
    assert score == 0.0


def test_estimate_alignment_returns_identity_for_no_overlap() -> None:
    reference = EnergySignal(
        timestamps=np.array([0.0, 0.1, 0.2]),
        values=np.array([0.0, 1.0, 0.0]),
    )
    source = EnergySignal(
        timestamps=np.array([1.0, 1.1, 1.2]),
        values=np.array([0.0, 1.0, 0.0]),
    )

    transform, score = estimate_alignment_from_signals(
        reference=reference,
        source=source,
        max_lag_seconds=1.0,
    )

    assert transform == TimelineTransform.identity()
    assert score == 0.0


def test_estimate_alignment_rejects_negative_max_lag() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0, 0.1]),
        values=np.array([0.0, 1.0]),
    )

    with pytest.raises(ValueError, match="max_lag_seconds must be non-negative"):
        estimate_alignment_from_signals(
            reference=signal,
            source=signal,
            max_lag_seconds=-1.0,
        )


def test_estimate_alignment_rejects_constant_signals() -> None:
    signal = EnergySignal(
        timestamps=np.array([0.0, 0.1, 0.2]),
        values=np.ones(3),
    )

    with pytest.raises(ValueError, match="constant energy signal"):
        estimate_alignment_from_signals(
            reference=signal,
            source=signal,
            max_lag_seconds=0.2,
        )


def _gaussian(times: np.ndarray, *, center: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / 0.05) ** 2)


def test_estimate_alignment_finds_source_to_reference_offset() -> None:
    timestamps = np.arange(0.0, 2.0, 0.01)
    reference_values = np.exp(-0.5 * ((timestamps - 0.8) / 0.03) ** 2)
    source_values = np.exp(-0.5 * ((timestamps - 1.0) / 0.03) ** 2)

    reference = EnergySignal(
        timestamps=timestamps,
        values=reference_values,
        name="reference_pulse",
    )
    source = EnergySignal(
        timestamps=timestamps,
        values=source_values,
        name="source_pulse",
    )

    transform, score = estimate_alignment_from_signals(
        reference=reference,
        source=source,
        max_lag_seconds=0.5,
    )

    assert transform.scale == 1.0
    assert transform.offset == pytest.approx(-0.2, abs=0.02)
    assert score > 0.8


def test_source_delayed_relative_to_reference() -> None:
    times = np.linspace(0.0, 2.0, 401)
    reference = EnergySignal(times, _gaussian(times, center=1.0))
    source = EnergySignal(times, _gaussian(times, center=1.3))
    transform, score = estimate_alignment_from_signals(
        reference=reference,
        source=source,
        max_lag_seconds=0.5,
    )
    assert score > 0.5
    assert transform.offset == pytest.approx(-0.3, abs=0.05)


def test_source_advanced_relative_to_reference() -> None:
    times = np.linspace(0.0, 2.0, 401)
    reference = EnergySignal(times, _gaussian(times, center=1.0))
    source = EnergySignal(times, _gaussian(times, center=0.7))
    transform, score = estimate_alignment_from_signals(
        reference=reference,
        source=source,
        max_lag_seconds=0.5,
    )
    assert score > 0.5
    assert transform.offset == pytest.approx(0.3, abs=0.05)


def test_constant_reference_signal_raises() -> None:
    times = np.linspace(0.0, 2.0, 101)
    reference = EnergySignal(times, np.ones_like(times))
    source = EnergySignal(times, _gaussian(times, center=1.0))
    with pytest.raises(ValueError, match="constant energy signal"):
        estimate_alignment_from_signals(
            reference=reference,
            source=source,
            max_lag_seconds=0.5,
        )


def test_constant_source_signal_raises() -> None:
    times = np.linspace(0.0, 2.0, 101)
    reference = EnergySignal(times, _gaussian(times, center=1.0))
    source = EnergySignal(times, np.ones_like(times))
    with pytest.raises(ValueError, match="constant energy signal"):
        estimate_alignment_from_signals(
            reference=reference,
            source=source,
            max_lag_seconds=0.5,
        )


# --- Demonstration integration ---


def test_demonstration_accepts_alignments_in_constructor() -> None:
    mocap = make_mocap_track()
    times = mocap.timestamps
    reference_signal = EnergySignal(
        timestamps=times,
        values=np.array([1.0, 0.0, 0.0]),
    )
    source_signal = EnergySignal(
        timestamps=times,
        values=np.array([0.0, 1.0, 0.0]),
    )
    transform, score = estimate_alignment_from_signals(
        reference=reference_signal,
        source=source_signal,
        max_lag_seconds=0.5,
    )
    alignment = TrackAlignment(
        source=GroundEstimationTrackId.MOCAP,
        reference=GroundEstimationTrackId.MOCAP,
        transform=transform,
        score=score,
    )
    demo = Demonstration(
        tracks={GroundEstimationTrackId.MOCAP: mocap},
        alignments=(alignment,),
    )
    assert len(demo.alignments) == 1
    stored = demo.alignments[0]
    assert stored.source is GroundEstimationTrackId.MOCAP
    assert stored.reference is GroundEstimationTrackId.MOCAP
    assert isinstance(stored.transform, TimelineTransform)
