from __future__ import annotations

import numpy as np

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
"""A floating-point array of arbitrary shape (stored as ``float64``)."""

type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.intp]]
"""An integer array of arbitrary shape (stored as ``intp``)."""
