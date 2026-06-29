"""Typed parameter dataclasses for SMPL-family variants."""

from smpl.params.smpl import SmplParams
from smpl.params.smplx import (
    SMPLX_NUM_BODY_JOINTS,
    SMPLX_NUM_FACE_JOINTS,
    SMPLX_NUM_HAND_JOINTS,
    SMPLX_NUM_JOINTS,
    SmplxParams,
)

__all__ = [
    "SMPLX_NUM_BODY_JOINTS",
    "SMPLX_NUM_FACE_JOINTS",
    "SMPLX_NUM_HAND_JOINTS",
    "SMPLX_NUM_JOINTS",
    "SmplParams",
    "SmplxParams",
]
