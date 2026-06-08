from typing import Literal, Any

import numpy as np


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