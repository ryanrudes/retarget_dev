from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smpl import (
    BodyModelData,
    SmplModel,
    SmplxModel,
    Variant,
    get_model_class,
    load,
    register_model,
    registered_variants,
    synthetic_model,
    synthetic_smplx_model,
    unregister_model,
)


def test_builtin_variants_resolve_to_their_classes() -> None:
    assert get_model_class("smpl") is SmplModel
    assert get_model_class("smplx") is SmplxModel
    assert {"smpl", "smplx"} <= registered_variants()


def test_variant_enum_is_the_closed_builtin_family() -> None:
    # The built-ins are a closed StrEnum (like urdf's RobotId); members ARE their strings, so they
    # resolve and compare exactly like the plain variant strings, and the enum + string interoperate.
    assert set(Variant) == {Variant.SMPL, Variant.SMPLX}
    assert Variant.SMPL == "smpl"
    assert Variant.SMPLX == "smplx"
    assert get_model_class(Variant.SMPL) is SmplModel
    assert get_model_class(Variant.SMPLX) is SmplxModel
    assert {Variant.SMPL, Variant.SMPLX} <= registered_variants()


def test_get_model_class_unknown_variant_raises() -> None:
    with pytest.raises(KeyError, match="nope"):
        get_model_class("nope")


def _save(path: Path, body: BodyModelData) -> None:
    np.savez(
        path,
        v_template=body.v_template,
        shapedirs=body.shapedirs,
        J_regressor=body.j_regressor,
        parents=body.parents.astype(np.int64),
        joint_names=np.array(body.joint_names),
    )


def test_load_roundtrips_smpl_and_smplx(tmp_path: Path) -> None:
    smpl_path = tmp_path / "smpl.npz"
    _save(smpl_path, synthetic_model(num_joints=3, num_betas=2))
    smpl_model = load("smpl", smpl_path)
    assert isinstance(smpl_model, SmplModel)
    assert smpl_model.body.num_joints == 3

    smplx_path = tmp_path / "smplx.npz"
    _save(smplx_path, synthetic_smplx_model(num_betas=2))
    smplx_model = load(Variant.SMPLX, smplx_path)  # the enum path narrows the return type
    assert isinstance(smplx_model, SmplxModel)
    assert smplx_model.body.num_joints == 55


def test_register_and_unregister_runtime_variant() -> None:
    register_model("smpl_alias", SmplModel)
    try:
        assert get_model_class("smpl_alias") is SmplModel
        assert "smpl_alias" in registered_variants()

        with pytest.raises(ValueError, match="already registered"):
            register_model("smpl_alias", SmplxModel)

        register_model("smpl_alias", SmplxModel, replace=True)
        assert get_model_class("smpl_alias") is SmplxModel
    finally:
        removed = unregister_model("smpl_alias")
        assert removed is SmplxModel

    assert "smpl_alias" not in registered_variants()


def test_builtin_variants_are_protected() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_model("smpl", SmplxModel)
    with pytest.raises(ValueError, match="built-in"):
        unregister_model("smpl")
    with pytest.raises(ValueError, match="empty"):
        register_model("", SmplModel)
