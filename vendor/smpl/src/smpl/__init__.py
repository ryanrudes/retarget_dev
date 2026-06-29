"""A standalone, typed numpy library for SMPL-family body models and forward kinematics."""

from smpl.io.load import BodyModelData, load_npz, synthetic_model
from smpl.models.smpl import SmplModel
from smpl.models.smplx import SmplxModel, synthetic_smplx_model
from smpl.params.smpl import SmplParams
from smpl.params.smplx import SMPLX_NUM_JOINTS, SmplxParams
from smpl.registry import (
    AnyModel,
    ModelFactory,
    Variant,
    get_model_class,
    load,
    register_model,
    registered_variants,
    unregister_model,
)

__all__ = [
    "SMPLX_NUM_JOINTS",
    "AnyModel",
    "BodyModelData",
    "ModelFactory",
    "SmplModel",
    "SmplParams",
    "SmplxModel",
    "SmplxParams",
    "Variant",
    "get_model_class",
    "load",
    "load_npz",
    "register_model",
    "registered_variants",
    "synthetic_model",
    "synthetic_smplx_model",
    "unregister_model",
]
