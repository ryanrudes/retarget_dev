"""Per-variant SMPL-family model classes."""

from smpl.models.smpl import SmplModel
from smpl.models.smplx import SmplxModel, synthetic_smplx_model

__all__ = [
    "SmplModel",
    "SmplxModel",
    "synthetic_smplx_model",
]
