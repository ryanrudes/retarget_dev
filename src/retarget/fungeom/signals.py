"""Compiled demonstration arrays → fungeom over-time signals.

Two layers:

* **array-level** (`point_bundle_signal`, `transform_bundle_signal`) translate
  retarget's native time-major arrays straight into fungeom resolvers. They own
  the two awkward bits the handoff spec calls out: deriving an occlusion
  ``present`` mask from NaN, and wrapping each pose into a grid of ``Transform``
  resolvers (SE(3) has no ergonomic raw form).
* **domain-level** (`marker_cloud_signal`, `pose_signal`) pull those arrays off a
  bound :class:`~retarget.demo.MocapTrack` using the authored string names as
  keys, and call the array-level builders.

Absence is carried by ``present`` (never a silent drop): an occluded marker is
``Unresolvable`` at that instant, not a NaN. Geometry stays in fungeom — this
module only moves names and arrays across the seam.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fungeom import (
    Boundary,
    CoordinateFrame,
    Frame,
    Interpolation,
    Point3BundleSignal,
    Transform,
    TransformBundleSignal,
    WORLD_FRAME,
)
from fungeom.values import RigidTransform

from retarget.core.schema.base import _schema_fields, _schema_get

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from retarget.demo import MocapTrack

__all__ = [
    "point_bundle_signal",
    "transform_bundle_signal",
    "marker_cloud_signal",
    "pose_signal",
]


# --------------------------------------------------------------------------- #
# array-level builders (pure; reusable without a full track)
# --------------------------------------------------------------------------- #
def point_bundle_signal(
    times: ArrayLike,
    frames: ArrayLike,
    keys: Sequence[str],
    *,
    present: ArrayLike | None = None,
    frame: CoordinateFrame | Frame = WORLD_FRAME,
    via: Interpolation = Interpolation.linear,
    outside: Boundary = Boundary.undefined,
    max_gap: float | None = None,
) -> Point3BundleSignal:
    """Build a :class:`Point3BundleSignal` (marker cloud) from time-major arrays.

    ``frames`` is ``(T, N, 3)``. A marker is *absent* at a frame when its row is
    non-finite (the retarget occlusion convention); that absence becomes the
    fungeom ``present`` mask, and the offending coordinates are neutralised so
    construction never chokes on a NaN. An explicit ``present`` mask is ANDed
    with finiteness — you can mark a finite sample absent, but never claim a
    non-finite sample present.
    """
    times_arr = np.asarray(times, dtype=np.float64)
    frames_arr = np.asarray(frames, dtype=np.float64)
    if frames_arr.ndim != 3 or frames_arr.shape[2] != 3:
        raise ValueError(f"frames must have shape (T, N, 3), got {frames_arr.shape}")
    if frames_arr.shape[0] != times_arr.shape[0]:
        raise ValueError(f"times has {times_arr.shape[0]} samples but frames has {frames_arr.shape[0]}")
    if len(keys) != frames_arr.shape[1]:
        raise ValueError(f"got {len(keys)} keys for {frames_arr.shape[1]} markers")

    finite = np.asarray(np.isfinite(frames_arr).all(axis=-1), dtype=bool)  # (T, N)
    mask = _present_mask(finite, present)
    frames_arr = np.where(finite[..., None], frames_arr, 0.0)

    return Point3BundleSignal.from_frames(
        times_arr,
        frames_arr,
        keys=list(keys),
        frame=frame,
        via=via,
        outside=outside,
        max_gap=max_gap,
        present=mask,
    )


def transform_bundle_signal(
    times: ArrayLike,
    translations: ArrayLike,
    rotations: ArrayLike,
    keys: Sequence[str],
    *,
    present: ArrayLike | None = None,
    via: Interpolation = Interpolation.linear,
    outside: Boundary = Boundary.undefined,
    max_gap: float | None = None,
) -> TransformBundleSignal:
    """Build a :class:`TransformBundleSignal` (joint/segment poses) from arrays.

    ``translations`` is ``(T, N, 3)`` and ``rotations`` is ``(T, N, 3, 3)``. Each
    pose is wrapped as ``Transform.known(RigidTransform.from_matrix(...))`` into
    the ``(T, N)`` grid that ``TransformBundleSignal.from_frames`` requires — the
    spec's first gotcha. Non-finite poses are treated as absent (``present``
    mask + an identity filler in the grid), matching the marker-cloud convention.
    """
    times_arr = np.asarray(times, dtype=np.float64)
    trans = np.asarray(translations, dtype=np.float64)
    rots = np.asarray(rotations, dtype=np.float64)
    if trans.ndim != 3 or trans.shape[2] != 3:
        raise ValueError(f"translations must have shape (T, N, 3), got {trans.shape}")
    if rots.ndim != 4 or rots.shape[2:] != (3, 3):
        raise ValueError(f"rotations must have shape (T, N, 3, 3), got {rots.shape}")
    if trans.shape[:2] != rots.shape[:2]:
        raise ValueError("translations and rotations must agree on (T, N)")
    num_times, num_entities = trans.shape[0], trans.shape[1]
    if times_arr.shape[0] != num_times:
        raise ValueError(f"times has {times_arr.shape[0]} samples but poses have {num_times}")
    if len(keys) != num_entities:
        raise ValueError(f"got {len(keys)} keys for {num_entities} segments")

    finite = np.asarray(np.isfinite(trans).all(axis=-1), dtype=bool) & np.asarray(
        np.isfinite(rots).all(axis=(-1, -2)), dtype=bool
    )  # (T, N)
    mask = _present_mask(finite, present)

    grid: list[list[Transform]] = []
    for t in range(num_times):
        row: list[Transform] = []
        for n in range(num_entities):
            if mask[t, n]:
                row.append(Transform.known(RigidTransform.from_matrix(_homogeneous(rots[t, n], trans[t, n]))))
            else:
                row.append(Transform.identity())
        grid.append(row)

    return TransformBundleSignal.from_frames(
        times_arr,
        grid,
        keys=list(keys),
        via=via,
        outside=outside,
        max_gap=max_gap,
        present=mask,
    )


def _homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Assemble a ``(4, 4)`` homogeneous matrix from a ``(3, 3)`` R and ``(3,)`` t."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _present_mask(finite: np.ndarray, present: ArrayLike | None) -> np.ndarray:
    """Combine derived finiteness with an optional explicit ``(T, N)`` present mask.

    An explicit mask may mark a finite sample absent, but never claim a
    non-finite sample present (finiteness always wins). Its shape must match
    ``finite`` exactly — a ``(T,)`` per-frame mask would otherwise silently
    mis-broadcast and corrupt the very absence this module exists to carry.
    """
    if present is None:
        return finite
    present_arr = np.asarray(present, dtype=bool)
    if present_arr.shape != finite.shape:
        raise ValueError(f"present must have shape {finite.shape}, got {present_arr.shape}")
    return np.asarray(present_arr & finite, dtype=bool)


# --------------------------------------------------------------------------- #
# domain-level builders (off a bound MocapTrack)
# --------------------------------------------------------------------------- #
def marker_cloud_signal(
    track: MocapTrack[Any],
    subject: str,
    segment: str,
    *,
    markers: Sequence[str] | None = None,
    modeled: bool = False,
    frame: CoordinateFrame | Frame = WORLD_FRAME,
    via: Interpolation = Interpolation.linear,
    outside: Boundary = Boundary.undefined,
    max_gap: float | None = None,
) -> Point3BundleSignal:
    """A segment's marker set as a :class:`Point3BundleSignal`, keyed by name.

    ``modeled=False`` (default) uses the measured Vicon cloud — occluded frames
    are NaN and become the ``present`` mask. ``modeled=True`` rigidly transports
    the rest positions through the segment pose (never occluded). Observed mode
    requires raw marker frames; it is unavailable on a resampled track.
    """
    segment_obj = _segment(track, subject, segment)
    names = tuple(markers) if markers is not None else _schema_fields(segment_obj.markers)
    frames = np.asarray(segment_obj.marker_positions(*names, modeled=modeled), dtype=np.float64)
    return point_bundle_signal(
        track.timestamps,
        frames,
        names,
        frame=frame,
        via=via,
        outside=outside,
        max_gap=max_gap,
    )


def pose_signal(
    track: MocapTrack[Any],
    subject: str,
    *,
    segments: Sequence[str] | None = None,
    present: ArrayLike | None = None,
    via: Interpolation = Interpolation.linear,
    outside: Boundary = Boundary.undefined,
    max_gap: float | None = None,
) -> TransformBundleSignal:
    """A subject's segment poses as a :class:`TransformBundleSignal`, keyed by segment.

    Each rigid segment contributes one pose ("joint") per frame. Keys are the
    authored segment names, which are unique within a subject.
    """
    subject_obj = _subject(track, subject)
    names = tuple(segments) if segments is not None else _schema_fields(subject_obj.segments)
    num_times = int(np.asarray(track.timestamps).shape[0])
    num_entities = len(names)
    translations = np.empty((num_times, num_entities, 3), dtype=np.float64)
    rotations = np.empty((num_times, num_entities, 3, 3), dtype=np.float64)
    for index, name in enumerate(names):
        seg = _schema_get(subject_obj.segments, name)
        translations[:, index, :] = np.asarray(seg.translations(), dtype=np.float64)
        rotations[:, index, :, :] = np.asarray(seg.rotations(), dtype=np.float64)
    return transform_bundle_signal(
        track.timestamps,
        translations,
        rotations,
        names,
        present=present,
        via=via,
        outside=outside,
        max_gap=max_gap,
    )


# --------------------------------------------------------------------------- #
# internal lookup helpers: resolve authored string names against a schema's
# dataclass fields (the public builders keep their string params at this seam)
# --------------------------------------------------------------------------- #
def _subject(track: MocapTrack[Any], subject: str) -> Any:
    return _schema_get(track.subjects, subject)


def _segment(track: MocapTrack[Any], subject: str, segment: str) -> Any:
    return _schema_get(_schema_get(track.subjects, subject).segments, segment)
