"""Utility functions for retarget."""

from retarget.utils.geometry import (
    point_in_polygon,
    fit_patch_frame,
)

from retarget.utils.intervals import (
    IntervalList,
    intervals_from_mask,
    mask_from_intervals,
    fill_short_false_runs,
    remove_short_true_runs,
    clean_mask_by_time,
)

from retarget.utils.sampler import (
    estimate_nominal_hz,
    validate_nominal_hz,
)

__all__ = [
    # retarget.utils.geometry
    "point_in_polygon",
    "fit_patch_frame",

    # retarget.utils.intervals
    "IntervalList",
    "intervals_from_mask",
    "mask_from_intervals",
    "fill_short_false_runs",
    "remove_short_true_runs",
    "clean_mask_by_time",

    # retarget.utils.sampler
    "estimate_nominal_hz",
    "validate_nominal_hz",
]