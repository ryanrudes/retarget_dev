"""
Optional ergonomic facade over ``Demonstration[GroundEstimationTrackId]``.

This module is **not** the required pattern for using demonstrations. The primary
example in ``demo_specs.load_ground_estimation_demo`` returns a generic
``Demonstration`` directly; use ``demo.track(GroundEstimationTrackId.MOCAP)`` and
narrow the result to ``MocapTrack`` or ``MocapTrackView`` when you need
mocap-specific APIs.

Import this facade only if you prefer property accessors such as ``demo.mocap``.
"""

from __future__ import annotations

from dataclasses import dataclass

from retarget.demo.alignment import EnergySignal, TrackAlignment
from retarget.demo.contact import ContactTrack
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.mocap import MocapTrack, MocapTrackView

from demo_specs import GroundEstimationTrackId


@dataclass(frozen=True, slots=True)
class GroundEstimationDemo:
    """Optional sugar: ergonomic track accessors over a generic demonstration."""

    inner: Demonstration[GroundEstimationTrackId]

    def track(self, track: GroundEstimationTrackId) -> object:
        return self.inner.track(track)

    @property
    def mocap(self) -> MocapTrack:
        value = self.inner.track(GroundEstimationTrackId.MOCAP)
        if not isinstance(value, MocapTrack):
            raise TypeError("MOCAP track is not a MocapTrack")
        return value

    @property
    def video(self) -> object:
        try:
            return self.inner.track(GroundEstimationTrackId.VIDEO)
        except KeyError as exc:
            raise KeyError("No video track is attached to this demonstration") from exc

    @property
    def smpl(self) -> object:
        try:
            return self.inner.track(GroundEstimationTrackId.SMPL)
        except KeyError as exc:
            raise KeyError("No SMPL track is attached to this demonstration") from exc

    @property
    def contacts(self) -> ContactTrack:
        try:
            value = self.inner.track(GroundEstimationTrackId.CONTACTS)
        except KeyError as exc:
            raise KeyError("No contact track is attached to this demonstration") from exc
        if not isinstance(value, ContactTrack):
            raise TypeError("CONTACTS track is not a ContactTrack")
        return value

    def slice_time(self, start: float, stop: float) -> GroundEstimationDemoView:
        return GroundEstimationDemoView(
            inner=self.inner.slice_time(start, stop)
        )

    def with_track(
        self,
        track: GroundEstimationTrackId,
        value: object,
    ) -> GroundEstimationDemo:
        return GroundEstimationDemo(
            inner=self.inner.with_track(track, value)
        )

    def with_contacts(self, contacts: ContactTrack) -> GroundEstimationDemo:
        mocap = self.mocap
        mocap_with_contacts = MocapTrack(
            scene_spec=mocap.scene_spec,
            state=mocap.state,
            timestamps=mocap.timestamps,
            marker_frames=mocap.marker_frames,
            contacts=contacts,
        )
        inner = (
            self.inner
            .with_track(GroundEstimationTrackId.MOCAP, mocap_with_contacts)
            .with_track(GroundEstimationTrackId.CONTACTS, contacts)
        )
        return GroundEstimationDemo(inner=inner)

    def with_alignment(
        self,
        alignment: TrackAlignment[GroundEstimationTrackId],
    ) -> GroundEstimationDemo:
        return GroundEstimationDemo(
            inner=self.inner.with_alignment(alignment)
        )

    def align(
        self,
        *,
        reference: GroundEstimationTrackId,
        source: GroundEstimationTrackId,
        reference_signal: EnergySignal,
        source_signal: EnergySignal,
        max_lag_seconds: float,
    ) -> GroundEstimationDemo:
        return GroundEstimationDemo(
            inner=self.inner.align(
                reference=reference,
                source=source,
                reference_signal=reference_signal,
                source_signal=source_signal,
                max_lag_seconds=max_lag_seconds,
            )
        )

    @classmethod
    def wrap(cls, demo: Demonstration[GroundEstimationTrackId]) -> GroundEstimationDemo:
        return cls(inner=demo)


@dataclass(frozen=True, slots=True)
class GroundEstimationDemoView:
    """Time-sliced view with optional ergonomic track accessors."""

    inner: DemonstrationView[GroundEstimationTrackId]

    def track(self, track: GroundEstimationTrackId) -> object:
        return self.inner.track(track)

    @property
    def mocap(self) -> MocapTrack | MocapTrackView:
        value = self.inner.track(GroundEstimationTrackId.MOCAP)
        if not isinstance(value, MocapTrack | MocapTrackView):
            raise TypeError("MOCAP track is not a mocap track or view")
        return value

    @property
    def video(self) -> object:
        try:
            return self.inner.track(GroundEstimationTrackId.VIDEO)
        except KeyError as exc:
            raise KeyError("No video track is attached to this demonstration") from exc

    @property
    def smpl(self) -> object:
        try:
            return self.inner.track(GroundEstimationTrackId.SMPL)
        except KeyError as exc:
            raise KeyError("No SMPL track is attached to this demonstration") from exc

    @property
    def contacts(self) -> ContactTrack:
        try:
            value = self.inner.track(GroundEstimationTrackId.CONTACTS)
        except KeyError as exc:
            raise KeyError("No contact track is attached to this demonstration") from exc
        if not isinstance(value, ContactTrack):
            raise TypeError("CONTACTS track is not a ContactTrack")
        return value

    def slice_time(self, start: float, stop: float) -> GroundEstimationDemoView:
        return GroundEstimationDemoView(
            inner=self.inner.slice_time(start, stop)
        )

    def resample_to(
        self,
        reference: GroundEstimationTrackId,
    ) -> GroundEstimationDemoView:
        return GroundEstimationDemoView(
            inner=self.inner.resample_to(reference)
        )
