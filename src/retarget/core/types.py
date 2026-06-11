from typing import Literal, Any

import numpy.typing as npt
import numpy as np


FloatArray = npt.NDArray[np.float64]
"""A numpy array of floating-point numbers."""

type Sign = Literal[-1, 1]
"""The sign of a scalar value."""

type Vec2 = np.ndarray[tuple[Literal[2]], np.dtype[np.floating[Any]]]
"""A 2-dimensional floating-point vector."""

type Vec3 = np.ndarray[tuple[Literal[3]], np.dtype[np.floating[Any]]]
"""A 3-dimensional floating-point vector."""

type Vec4 = np.ndarray[tuple[Literal[4]], np.dtype[np.floating[Any]]]
"""A 4-dimensional floating-point vector."""

type Vec6 = np.ndarray[tuple[Literal[6]], np.dtype[np.floating[Any]]]
"""A 6-dimensional floating-point vector."""

type Mat3 = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.floating[Any]]]
"""A 3x3 floating-point matrix."""

type Points2 = np.ndarray[
    tuple[int, Literal[2]],
    np.dtype[np.floating[Any]],
]
"""A set of 2D points."""

type Points3 = np.ndarray[
    tuple[int, Literal[3]],
    np.dtype[np.floating[Any]],
]
"""A set of 3D points."""

type TimeVec3 = np.ndarray[
    tuple[int, Literal[3]],
    np.dtype[np.floating[Any]],
]
"""Time-major vector signal with shape ``(T, 3)``."""

type TimeEntityVec3 = np.ndarray[
    tuple[int, int, Literal[3]],
    np.dtype[np.floating[Any]],
]
"""Time-major multi-entity vector signal with shape ``(T, N, 3)``."""

type TimeMat3 = np.ndarray[
    tuple[int, Literal[3], Literal[3]],
    np.dtype[np.floating[Any]],
]
"""Time-major rotation matrix signal with shape ``(T, 3, 3)``."""

type TimeQuat = np.ndarray[
    tuple[int, Literal[4]],
    np.dtype[np.floating[Any]],
]
"""Time-major quaternion signal with shape ``(T, 4)``."""

type TimeBool = np.ndarray[
    tuple[int],
    np.dtype[np.bool_],
]
"""Time-major boolean signal with shape ``(T,)``."""

type TimeEntityBool = np.ndarray[
    tuple[int, int],
    np.dtype[np.bool_],
]
"""Time-major multi-entity boolean signal with shape ``(T, N)``."""