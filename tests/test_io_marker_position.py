from __future__ import annotations

import numpy as np

from retarget.io import MarkerObservation, ViconMarkersFrame, marker_position


def _frame(*observations: MarkerObservation) -> ViconMarkersFrame:
    return ViconMarkersFrame(stamp_seconds=0.0, markers=observations)


def test_marker_position_returns_observed_position() -> None:
    expected = np.array([1.0, 2.0, 3.0])
    frame = _frame(
        MarkerObservation(
            marker_name="heel",
            subject_name="Left_Shoe_Improved",
            segment_name="Left_Shoe_Improved",
            position_world=expected,
            occluded=False,
        )
    )
    observed = marker_position(
        frame,
        subject_name="Left_Shoe_Improved",
        segment_name="Left_Shoe_Improved",
        marker_name="heel",
    )
    assert observed is not None
    np.testing.assert_allclose(observed, expected)


def test_marker_position_returns_none_when_missing() -> None:
    frame = _frame()
    assert (
        marker_position(
            frame,
            subject_name="s",
            segment_name="g",
            marker_name="heel",
        )
        is None
    )


def test_marker_position_skips_occluded_observation() -> None:
    frame = _frame(
        MarkerObservation(
            marker_name="heel",
            subject_name="s",
            segment_name="g",
            position_world=np.array([9.0, 9.0, 9.0]),
            occluded=True,
        )
    )
    assert (
        marker_position(frame, subject_name="s", segment_name="g", marker_name="heel")
        is None
    )
