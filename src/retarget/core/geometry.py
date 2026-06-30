"""The bridge that turns a bound patch ``Face`` + segment pose into a *moving* fungeom ``FaceSignal``.

A patch is authored as data -- a fungeom :class:`~fungeom.Face` over the segment's marker symbols --
and the binding resolves it to a segment-local ``Face`` at bind time (``Face.bind(env)``). At query
time :func:`face_signal` fixes that ``Face`` in the segment frame and transports it by the segment pose
as a fungeom :class:`~fungeom.FaceSignal`; the patch query methods materialize
``frame()``/``plane()``/``boundary()`` over the track timestamps with :func:`sampling_at`. Geometry
lives in fungeom -- retarget only assembles the pose carrier and reads the signals back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fungeom import FaceSignal, Sampling, TransformSignal

if TYPE_CHECKING:
    from fungeom import Face

    from retarget.core.types import FloatArray

__all__ = [
    "face_signal",
    "segment_pose_signal",
    "sampling_at",
]


def segment_pose_signal(
    timestamps: FloatArray,
    translations: FloatArray,
    rotations: FloatArray,
) -> TransformSignal:
    """The segment pose over time as a fungeom :class:`~fungeom.TransformSignal`.

    Assembles the runtime's ``(T, 3, 3)`` rotations and ``(T, 3)`` translations into a dense
    ``(T, 4, 4)`` stack and wraps it with the vectorized ``TransformSignal.from_matrices``
    carrier, so resolving over the track timestamps stays O(T) in numpy.
    """
    times = np.asarray(timestamps, dtype=np.float64)
    rotation = np.asarray(rotations, dtype=np.float64)
    translation = np.asarray(translations, dtype=np.float64)
    matrices = np.zeros((times.shape[0], 4, 4), dtype=np.float64)
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = translation
    matrices[:, 3, 3] = 1.0
    return TransformSignal.from_matrices(times, matrices)


def face_signal(
    face: Face,
    timestamps: FloatArray,
    translations: FloatArray,
    rotations: FloatArray,
) -> FaceSignal:
    """The patch as a *moving* fungeom :class:`~fungeom.FaceSignal`.

    The bind-time segment-local ``face`` fixed in the segment frame and transported by the
    segment pose over time. ``face_signal(...).frame()/plane()/boundary()/clearance(...)`` are
    the patch's world geometry as signals, materialized via :func:`sampling_at`.
    """
    return FaceSignal.of(face, segment_pose_signal(timestamps, translations, rotations))


def sampling_at(timestamps: FloatArray) -> Sampling:
    """A fungeom :class:`~fungeom.Sampling` at the track timestamps (for ``resolve_over``)."""
    return Sampling.at_times(np.asarray(timestamps, dtype=np.float64))
