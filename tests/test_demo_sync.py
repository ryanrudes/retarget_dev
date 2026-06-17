from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pytest

from retarget.core.types import FloatArray
from retarget.demo import Tracks, build_demonstration
from retarget.demo.alignment import EnergySignal, TimelineTransform, TrackAlignment
from retarget.demo.demo import Demonstration
from retarget.demo.sync import (
    SyncEdge,
    SyncPlan,
    compose_alignments_to_reference,
    estimate_sync,
    estimate_sync_and_resample_to_reference,
    estimate_sync_to_reference,
)
from retarget.demo.tracks import Track, TrackView

REFERENCE = "reference"
MOCAP = "mocap"
CONTACT = "contact"
AUX = "aux"


class DummyTrack(Track):
    def __init__(
        self,
        timestamps: FloatArray,
        values: FloatArray,
        nominal_hz_override: float | None = None,
    ) -> None:
        self._timestamps = timestamps
        self._values = values
        self._nominal_hz_override = nominal_hz_override

    @property
    def timestamps(self) -> FloatArray:
        return self._timestamps

    @property
    def values(self) -> FloatArray:
        return self._values

    @property
    def nominal_hz_override(self) -> float | None:
        return self._nominal_hz_override

    def resample_to(
        self,
        timestamps: FloatArray,
        *,
        output_timestamps: FloatArray | None = None,
    ) -> "DummyTrack":
        sample_timestamps = np.asarray(timestamps, dtype=np.float64)
        result_timestamps = (
            sample_timestamps
            if output_timestamps is None
            else np.asarray(output_timestamps, dtype=np.float64)
        )
        return DummyTrack(
            timestamps=result_timestamps,
            values=np.interp(sample_timestamps, self.timestamps, self.values),
            nominal_hz_override=self.nominal_hz_override,
        )

    @classmethod
    def _view_type(cls) -> type["DummyTrackView"]:
        return DummyTrackView


@dataclass(frozen=True, slots=True)
class DummyTrackView(TrackView[DummyTrack]):
    @property
    def timestamps(self) -> FloatArray:
        return self.source.timestamps[list(self.indices)]

    @property
    def values(self) -> FloatArray:
        return self.source.values[list(self.indices)]

    def resample_to(
        self,
        timestamps: FloatArray,
        *,
        output_timestamps: FloatArray | None = None,
    ) -> DummyTrack:
        sample_timestamps = np.asarray(timestamps, dtype=np.float64)
        result_timestamps = (
            sample_timestamps
            if output_timestamps is None
            else np.asarray(output_timestamps, dtype=np.float64)
        )
        return DummyTrack(
            timestamps=result_timestamps,
            values=np.interp(sample_timestamps, self.timestamps, self.values),
            nominal_hz_override=self.nominal_hz_override,
        )


class TypedSyncTracks(Tracks):
    reference: DummyTrack
    mocap: DummyTrack


def _signal(track: Track) -> EnergySignal:
    assert isinstance(track, DummyTrack | DummyTrackView)
    return EnergySignal(name="dummy-energy", timestamps=track.timestamps, values=track.values)


def _dummy_track(shift: float = 0.0) -> DummyTrack:
    timestamps = np.linspace(0.0, 2.0, 101)
    values = np.exp(-((timestamps - 1.0 - shift) ** 2) / 0.01)
    return DummyTrack(timestamps=timestamps, values=values)


def _edge(source: str, reference: str, max_lag: float = 0.25) -> SyncEdge:
    return SyncEdge(
        source=source,
        reference=reference,
        source_signal=_signal,
        reference_signal=_signal,
        max_lag_seconds=max_lag,
    )


def test_sync_plan_accepts_connected_non_star_graph() -> None:
    plan = SyncPlan(reference=REFERENCE, edges=(_edge(MOCAP, REFERENCE), _edge(CONTACT, MOCAP)))
    assert plan.track_ids == frozenset({REFERENCE, MOCAP, CONTACT})
    assert plan.edge_ids == frozenset({(MOCAP, REFERENCE), (CONTACT, MOCAP)})


def test_sync_edge_rejects_self_edge() -> None:
    with pytest.raises(ValueError, match="source and reference must be different"):
        _edge(MOCAP, MOCAP)


def test_sync_edge_rejects_negative_max_lag() -> None:
    with pytest.raises(ValueError, match="max_lag_seconds must be non-negative"):
        _edge(MOCAP, REFERENCE, max_lag=-0.1)


def test_sync_plan_rejects_empty_edges() -> None:
    with pytest.raises(ValueError, match="at least one edge"):
        SyncPlan(reference=REFERENCE, edges=())


def test_sync_plan_rejects_duplicate_directed_edges() -> None:
    edge = _edge(MOCAP, REFERENCE)
    with pytest.raises(ValueError, match="duplicate directed edges"):
        SyncPlan(reference=REFERENCE, edges=(edge, edge))


def test_sync_plan_rejects_duplicate_undirected_edges() -> None:
    with pytest.raises(ValueError, match="duplicate undirected edges"):
        SyncPlan(reference=REFERENCE, edges=(_edge(MOCAP, REFERENCE), _edge(REFERENCE, MOCAP)))


def test_sync_plan_rejects_disconnected_graph() -> None:
    with pytest.raises(ValueError, match="graph must be connected"):
        SyncPlan(reference=REFERENCE, edges=(_edge(MOCAP, REFERENCE), _edge(CONTACT, AUX)))


def test_estimate_sync_rejects_missing_tracks() -> None:
    plan = SyncPlan(reference=REFERENCE, edges=(_edge(MOCAP, REFERENCE),))
    demo = Demonstration(tracks={REFERENCE: _dummy_track()})
    with pytest.raises(KeyError, match="missing tracks"):
        estimate_sync(demo, plan)


def test_estimate_sync_returns_pairwise_alignments() -> None:
    plan = SyncPlan(
        reference=REFERENCE,
        edges=(_edge(MOCAP, REFERENCE, 0.5), _edge(CONTACT, MOCAP, 0.5)),
    )
    demo = Demonstration(
        tracks={
            REFERENCE: _dummy_track(0.0),
            MOCAP: _dummy_track(0.1),
            CONTACT: _dummy_track(0.2),
        }
    )
    alignments = estimate_sync(demo, plan)
    assert [(a.source, a.reference) for a in alignments] == [(MOCAP, REFERENCE), (CONTACT, MOCAP)]
    assert all(a.score is not None and isfinite(a.score) for a in alignments)


def test_compose_alignments_to_reference_composes_chain() -> None:
    alignments = compose_alignments_to_reference(
        reference=REFERENCE,
        alignments=(
            TrackAlignment(source=MOCAP, reference=REFERENCE, transform=TimelineTransform(scale=3.0, offset=1.0)),
            TrackAlignment(source=CONTACT, reference=MOCAP, transform=TimelineTransform(scale=1.0, offset=2.0)),
        ),
    )
    by_source = {a.source: a for a in alignments}
    assert by_source[MOCAP].transform == TimelineTransform(scale=3.0, offset=1.0)
    assert by_source[CONTACT].transform == TimelineTransform(scale=3.0, offset=7.0)


def test_compose_alignments_to_reference_inverts_reverse_edge() -> None:
    alignments = compose_alignments_to_reference(
        reference=REFERENCE,
        alignments=(
            TrackAlignment(source=REFERENCE, reference=MOCAP, transform=TimelineTransform(scale=2.0, offset=4.0)),
            TrackAlignment(source=CONTACT, reference=MOCAP, transform=TimelineTransform(scale=1.0, offset=3.0)),
        ),
    )
    by_source = {a.source: a for a in alignments}
    assert by_source[MOCAP].transform == TimelineTransform(scale=0.5, offset=-2.0)
    assert by_source[CONTACT].transform == TimelineTransform(scale=0.5, offset=-0.5)


def test_compose_alignments_to_reference_rejects_disconnected_graph() -> None:
    with pytest.raises(ValueError, match="Alignment graph must be connected"):
        compose_alignments_to_reference(
            reference=REFERENCE,
            alignments=(
                TrackAlignment(source=MOCAP, reference=REFERENCE, transform=TimelineTransform.identity()),
                TrackAlignment(source=CONTACT, reference=AUX, transform=TimelineTransform.identity()),
            ),
        )


def test_estimate_sync_to_reference_returns_root_alignments() -> None:
    plan = SyncPlan(
        reference=REFERENCE,
        edges=(_edge(MOCAP, REFERENCE, 0.5), _edge(CONTACT, MOCAP, 0.5)),
    )
    demo = Demonstration(
        tracks={REFERENCE: _dummy_track(0.0), MOCAP: _dummy_track(0.1), CONTACT: _dummy_track(0.2)}
    )
    alignments = estimate_sync_to_reference(demo, plan)
    assert {a.source for a in alignments} == {MOCAP, CONTACT}
    assert all(a.reference == REFERENCE for a in alignments)


def test_estimate_sync_and_resample_to_reference_returns_reference_time_view() -> None:
    plan = SyncPlan(reference=REFERENCE, edges=(_edge(MOCAP, REFERENCE, 0.5),))
    demo = Demonstration(tracks={REFERENCE: _dummy_track(0.0), MOCAP: _dummy_track(0.1)})
    resampled = estimate_sync_and_resample_to_reference(demo, plan, start=0.75, stop=1.25)
    reference_timestamps = resampled[REFERENCE].timestamps
    np.testing.assert_array_equal(resampled[MOCAP].timestamps, reference_timestamps)
    assert len(reference_timestamps) > 0
    assert len(resampled.alignments) == 1
    assert resampled.alignments[0].source == MOCAP


def test_estimate_sync_and_resample_to_reference_on_typed_demo() -> None:
    demo = build_demonstration(
        TypedSyncTracks(reference=_dummy_track(0.0), mocap=_dummy_track(0.1))
    )
    plan = SyncPlan(reference=REFERENCE, edges=(_edge(MOCAP, REFERENCE, 0.5),))
    resampled = estimate_sync_and_resample_to_reference(demo, plan, start=0.75, stop=1.25)
    np.testing.assert_array_equal(
        resampled["mocap"].timestamps, resampled["reference"].timestamps
    )
