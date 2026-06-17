from __future__ import annotations

import numpy as np
import pytest

from retarget.core.targets import PatchTarget
from retarget.demo.alignment import TimelineTransform, TrackAlignment
from retarget.demo.contact import ContactTrack
from retarget.demo.demo import Demonstration, DemonstrationView
from retarget.demo.resampling import ResampleMethod
from retarget.demo.tracks import Track


from conftest import make_string_patch_target as _target, make_string_contact_track

def _contact_track(
    *,
    target: PatchTarget,
    timestamps: list[float],
    contacts: list[bool],
    confidences: list[float],
) -> ContactTrack:
    track, _ = make_string_contact_track(
        timestamps=timestamps,
        target_name=target.patch,
        contacts=contacts,
        confidences=confidences,
    )
    return track


class ReferenceOnlyTrack(Track):
    """Track with timestamps but no resampling implementation."""

    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = np.array(timestamps, dtype=np.float64)

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps


def test_demonstration_view_resample_to_reference_track() -> None:
    reference_target = _target("reference")
    source_target = _target("source")
    reference_track = _contact_track(
        target=reference_target,
        timestamps=[10.0, 11.0, 12.0],
        contacts=[False, True, False],
        confidences=[0.1, 0.2, 0.3],
    )
    source_track = _contact_track(
        target=source_target,
        timestamps=[0.0, 1.0, 2.0],
        contacts=[True, False, True],
        confidences=[0.4, 0.5, 0.6],
    )
    demo = Demonstration(
        tracks={"reference": reference_track, "source": source_track},
        alignments=(
            TrackAlignment(
                source="source",
                reference="reference",
                transform=TimelineTransform(scale=1.0, offset=10.0),
            ),
        ),
    )
    view = demo.slice_time(0.0, 13.0)

    resampled = view.resample_to("reference")

    assert isinstance(resampled, DemonstrationView)
    np.testing.assert_array_equal(
        resampled["reference"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["source"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["source"].state(source_target),
        np.array([True, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled["source"].confidence(source_target),
        np.array([0.4, 0.5, 0.6], dtype=np.float64),
    )
    assert resampled.source is demo
    assert resampled.alignments == demo.alignments


def test_demonstration_view_resample_to_mocap_reference_and_contact_source() -> None:
    reference_target = _target("reference")
    source_target = _target("source")
    reference_track = _contact_track(
        target=reference_target,
        timestamps=[10.0, 11.0, 12.0],
        contacts=[False, True, False],
        confidences=[0.1, 0.2, 0.3],
    )
    source_track = _contact_track(
        target=source_target,
        timestamps=[0.0, 1.0, 2.0],
        contacts=[True, False, True],
        confidences=[0.4, 0.5, 0.6],
    )
    demo = Demonstration(
        tracks={"mocap": reference_track, "contact": source_track},
        alignments=(
            TrackAlignment(
                source="contact",
                reference="mocap",
                transform=TimelineTransform(scale=1.0, offset=10.0),
            ),
        ),
    )
    view = demo.slice_time(0.0, 13.0)

    resampled = view.resample_to("mocap")

    np.testing.assert_array_equal(
        resampled["mocap"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["contact"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["contact"].state(source_target),
        np.array([True, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled["contact"].confidence(source_target),
        np.array([0.4, 0.5, 0.6], dtype=np.float64),
    )


def test_demonstration_view_resample_to_forwards_discrete_method() -> None:
    reference_target = _target("reference")
    source_target = _target("source")
    # Reference timestamps fall between source samples so NEAREST vs PREVIOUS differ.
    reference_track = _contact_track(
        target=reference_target,
        timestamps=[10.4, 11.6],
        contacts=[True, True],
        confidences=[0.5, 0.5],
    )
    source_track = _contact_track(
        target=source_target,
        timestamps=[0.0, 1.0, 2.0],
        contacts=[True, False, True],
        confidences=[0.1, 0.2, 0.3],
    )
    demo = Demonstration(
        tracks={"reference": reference_track, "source": source_track},
        alignments=(
            TrackAlignment(
                source="source",
                reference="reference",
                transform=TimelineTransform(scale=1.0, offset=10.0),
            ),
        ),
    )
    view = demo.slice_time(0.0, 13.0)

    nearest = view.resample_to("reference", method=ResampleMethod.NEAREST)
    previous = view.resample_to("reference", method=ResampleMethod.PREVIOUS)

    np.testing.assert_array_equal(
        nearest["source"].state(source_target),
        np.array([True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        previous["source"].state(source_target),
        np.array([True, False], dtype=np.bool_),
    )


def test_demonstration_view_resample_to_does_not_resample_reference_track() -> None:
    source_target = _target("source")
    reference_track = ReferenceOnlyTrack([10.0, 11.0, 12.0])
    source_track = _contact_track(
        target=source_target,
        timestamps=[0.0, 1.0, 2.0],
        contacts=[True, False, True],
        confidences=[0.4, 0.5, 0.6],
    )
    demo = Demonstration(
        tracks={"reference": reference_track, "source": source_track},
        alignments=(
            TrackAlignment(
                source="source",
                reference="reference",
                transform=TimelineTransform(scale=1.0, offset=10.0),
            ),
        ),
    )
    view = DemonstrationView(
        source=demo,
        tracks=dict(demo.tracks),
        alignments=demo.alignments,
    )

    resampled = view.resample_to("reference")

    assert resampled["reference"] is view["reference"]
    np.testing.assert_array_equal(
        resampled["source"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["source"].state(source_target),
        np.array([True, False, True], dtype=np.bool_),
    )


def test_demonstration_view_resample_to_requires_non_reference_resampling() -> None:
    reference_target = _target("reference")
    reference_track = _contact_track(
        target=reference_target,
        timestamps=[10.0, 11.0, 12.0],
        contacts=[False, True, False],
        confidences=[0.1, 0.2, 0.3],
    )
    source_track = ReferenceOnlyTrack([0.0, 1.0, 2.0])
    demo = Demonstration(
        tracks={"reference": reference_track, "source": source_track},
        alignments=(
            TrackAlignment(
                source="source",
                reference="reference",
                transform=TimelineTransform(scale=1.0, offset=10.0),
            ),
        ),
    )
    view = DemonstrationView(
        source=demo,
        tracks=dict(demo.tracks),
        alignments=demo.alignments,
    )

    with pytest.raises(NotImplementedError, match="does not implement resample_to"):
        view.resample_to("reference")


def test_demonstration_view_resample_to_missing_reference_raises_key_error() -> None:
    target = _target("track")
    demo = Demonstration(
        tracks={
            "track": _contact_track(
                target=target,
                timestamps=[0.0, 1.0],
                contacts=[False, True],
                confidences=[0.1, 0.2],
            )
        }
    )
    view = demo.slice_time(0.0, 2.0)

    with pytest.raises(KeyError, match="Reference track"):
        view.resample_to("missing")


def test_demonstration_view_resample_to_missing_alignment_raises_value_error() -> None:
    reference_target = _target("reference")
    source_target = _target("source")
    demo = Demonstration(
        tracks={
            "reference": _contact_track(
                target=reference_target,
                timestamps=[0.0, 1.0],
                contacts=[False, True],
                confidences=[0.1, 0.2],
            ),
            "source": _contact_track(
                target=source_target,
                timestamps=[0.0, 1.0],
                contacts=[True, False],
                confidences=[0.3, 0.4],
            ),
        }
    )
    view = demo.slice_time(0.0, 2.0)

    with pytest.raises(ValueError, match="no alignment"):
        view.resample_to("reference")


def test_demonstration_view_resample_to_duplicate_alignment_raises_value_error() -> None:
    reference_target = _target("reference")
    source_target = _target("source")
    demo = Demonstration(
        tracks={
            "reference": _contact_track(
                target=reference_target,
                timestamps=[0.0, 1.0],
                contacts=[False, True],
                confidences=[0.1, 0.2],
            ),
            "source": _contact_track(
                target=source_target,
                timestamps=[0.0, 1.0],
                contacts=[True, False],
                confidences=[0.3, 0.4],
            ),
        },
        alignments=(
            TrackAlignment(
                source="source",
                reference="reference",
                transform=TimelineTransform.identity(),
            ),
            TrackAlignment(
                source="source",
                reference="reference",
                transform=TimelineTransform(scale=1.0, offset=1.0),
            ),
        ),
    )
    view = demo.slice_time(0.0, 2.0)

    with pytest.raises(ValueError, match="Multiple alignments"):
        view.resample_to("reference")