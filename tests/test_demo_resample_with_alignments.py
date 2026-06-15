from __future__ import annotations

import numpy as np

from retarget.demo.alignment import TimelineTransform, TrackAlignment
from retarget.demo.demo import Demonstration, DemonstrationView
from conftest import make_string_patch_target as _target, make_string_contact_track


def test_resample_with_alignments_on_demonstration() -> None:
    mocap_target = _target("mocap")
    contact_target = _target("contact")

    reference_track, _ = make_string_contact_track(
        target_name="reference",
        timestamps=[10.0, 11.0, 12.0],
        contacts=[False, True, False],
        confidences=[0.1, 0.2, 0.3],
    )
    mocap_track, _ = make_string_contact_track(
        target_name="mocap",
        timestamps=[0.0, 1.0, 2.0],
        contacts=[True, False, True],
        confidences=[0.4, 0.5, 0.6],
    )
    contact_track, _ = make_string_contact_track(
        target_name="contact",
        timestamps=[0.0, 1.0, 2.0],
        contacts=[False, True, False],
        confidences=[0.7, 0.8, 0.9],
    )

    # We have pairwise alignments:
    # contact -> mocap (shift by +1.0)
    # mocap -> reference (shift by +10.0)
    pairwise_alignments = (
        TrackAlignment(
            source="contact",
            reference="mocap",
            transform=TimelineTransform(scale=1.0, offset=1.0),
        ),
        TrackAlignment(
            source="mocap",
            reference="reference",
            transform=TimelineTransform(scale=1.0, offset=10.0),
        ),
    )

    demo = Demonstration(
        tracks={
            "reference": reference_track,
            "mocap": mocap_track,
            "contact": contact_track,
        }
    )

    # Call resample_with_alignments on the Demonstration
    resampled = demo.resample_with_alignments(
        reference="reference",
        alignments=pairwise_alignments,
    )

    assert isinstance(resampled, DemonstrationView)
    assert resampled.source is demo
    
    # Composed alignments should have been stored:
    # mocap reference timeline has offset 10.0 relative to mocap
    # contact reference timeline has offset 11.0 relative to reference
    composed_alignments = {a.source: a for a in resampled.alignments}
    assert set(composed_alignments) == {"mocap", "contact"}
    assert composed_alignments["mocap"].transform == TimelineTransform(scale=1.0, offset=10.0)
    assert composed_alignments["contact"].transform == TimelineTransform(scale=1.0, offset=11.0)

    # Verify the resampled track values
    np.testing.assert_array_equal(
        resampled["reference"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["mocap"].timestamps,
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        resampled["mocap"].state(mocap_target),
        np.array([True, False, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        resampled["contact"].state(contact_target),
        # contact timeline: [0, 1, 2] -> reference: [11, 12, 13]
        # sampled at reference: [10, 11, 12] -> maps to contact time: [-1, 0, 1]
        # nearest contact states at [-1, 0, 1] -> [False, False, True]
        np.array([False, False, True], dtype=np.bool_),
    )


def test_resample_with_alignments_on_demonstration_view() -> None:
    mocap_target = _target("mocap")

    reference_track, _ = make_string_contact_track(
        target_name="reference",
        timestamps=[10.0, 11.0, 12.0],
        contacts=[False, True, False],
        confidences=[0.1, 0.2, 0.3],
    )
    mocap_track, _ = make_string_contact_track(
        target_name="mocap",
        timestamps=[0.0, 1.0, 2.0],
        contacts=[True, False, True],
        confidences=[0.4, 0.5, 0.6],
    )

    pairwise_alignments = (
        TrackAlignment(
            source="mocap",
            reference="reference",
            transform=TimelineTransform(scale=1.0, offset=10.0),
        ),
    )

    demo = Demonstration(
        tracks={
            "reference": reference_track,
            "mocap": mocap_track,
        }
    )

    # Slice the demo first to get a DemonstrationView
    view = demo.slice_time(0.0, 13.0)
    assert isinstance(view, DemonstrationView)

    resampled = view.resample_with_alignments(
        reference="reference",
        alignments=pairwise_alignments,
    )

    assert isinstance(resampled, DemonstrationView)
    assert resampled.source.alignments == resampled.alignments
    np.testing.assert_array_equal(
        resampled["mocap"].state(mocap_target),
        np.array([True, False, True], dtype=np.bool_),
    )
