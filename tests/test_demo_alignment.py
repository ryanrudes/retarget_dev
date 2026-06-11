from __future__ import annotations

import numpy as np
import pytest

from demo_vocab import GroundEstimationTrackId
from retarget.demo.alignment import (
    EnergySignal,
    TimelineTransform,
    estimate_alignment_from_signals,
)
from retarget.demo.demo import Demonstration
from conftest import make_mocap_track


def test_timeline_transform_round_trip() -> None:
    transform = TimelineTransform(scale=1.0, offset=0.5)
    source_times = np.array([0.0, 1.0, 2.0])
    reference_times = transform.to_reference(source_times)
    recovered = transform.to_source(reference_times)
    np.testing.assert_allclose(recovered, source_times)


def _gaussian(times: np.ndarray, *, center: float) -> np.ndarray:
    return np.exp(-0.5 * ((times - center) / 0.05) ** 2)


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


def test_energy_signal_rejects_duplicate_timestamps() -> None:
    with pytest.raises(ValueError, match="EnergySignal timestamps must be strictly increasing"):
        EnergySignal(
            timestamps=np.array([0.0, 0.1, 0.1]),
            values=np.array([0.0, 1.0, 2.0]),
        )


def test_demo_align_stores_alignment_metadata() -> None:
    mocap = make_mocap_track()
    demo = Demonstration(tracks={GroundEstimationTrackId.MOCAP: mocap})
    times = mocap.timestamps
    reference_signal = EnergySignal(
        timestamps=times,
        values=np.array([1.0, 0.0, 0.0]),
    )
    source_signal = EnergySignal(
        timestamps=times,
        values=np.array([0.0, 1.0, 0.0]),
    )
    aligned = demo.align(
        reference=GroundEstimationTrackId.MOCAP,
        source=GroundEstimationTrackId.MOCAP,
        reference_signal=reference_signal,
        source_signal=source_signal,
        max_lag_seconds=0.5,
    )
    assert len(aligned.alignments) == 1
    alignment = aligned.alignments[0]
    assert alignment.source is GroundEstimationTrackId.MOCAP
    assert alignment.reference is GroundEstimationTrackId.MOCAP
    assert isinstance(alignment.transform, TimelineTransform)
