from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smpl import BodyModelData, load_npz, synthetic_model


def test_synthetic_model_is_valid() -> None:
    model = synthetic_model(num_joints=4, num_betas=3)
    assert model.num_joints == 4
    assert model.num_betas == 3
    assert model.num_vertices == 6
    assert model.parents[0] == -1
    np.testing.assert_array_equal(model.parents[1:], np.arange(3))
    # Root regresses to the origin so a root rotation rotates the tree rigidly about it.
    rest = model.j_regressor @ model.v_template
    np.testing.assert_allclose(rest[0], np.zeros(3))
    assert len(set(model.joint_names)) == 4


def test_load_npz_roundtrip_with_kintree_table(tmp_path: Path) -> None:
    src = synthetic_model(num_joints=3, num_betas=2)
    parents_row = np.where(src.parents < 0, 2**32 - 1, src.parents).astype(np.int64)
    kintree_table = np.stack([parents_row, np.arange(3, dtype=np.int64)])

    path = tmp_path / "model.npz"
    np.savez(
        path,
        v_template=src.v_template,
        shapedirs=src.shapedirs,
        J_regressor=src.j_regressor,
        kintree_table=kintree_table,
        joint_names=np.array(src.joint_names),
    )

    loaded = load_npz(path)
    assert isinstance(loaded, BodyModelData)
    np.testing.assert_array_equal(loaded.parents, src.parents)
    np.testing.assert_allclose(loaded.v_template, src.v_template)
    assert loaded.joint_names == src.joint_names


def test_load_npz_defaults_joint_names_when_absent(tmp_path: Path) -> None:
    src = synthetic_model(num_joints=3, num_betas=2)
    path = tmp_path / "model_no_names.npz"
    np.savez(
        path,
        v_template=src.v_template,
        shapedirs=src.shapedirs,
        j_regressor=src.j_regressor,
        parents=src.parents.astype(np.int64),
    )

    loaded = load_npz(path)
    assert loaded.joint_names == ("joint_0", "joint_1", "joint_2")


def test_body_model_rejects_inconsistent_arrays() -> None:
    with pytest.raises(ValueError, match="j_regressor"):
        BodyModelData(
            v_template=np.zeros((5, 3)),
            shapedirs=np.zeros((5, 3, 2)),
            j_regressor=np.zeros((3, 4)),  # V mismatch
            parents=np.array([-1, 0, 1]),
            joint_names=("a", "b", "c"),
        )

    with pytest.raises(ValueError, match="unique"):
        BodyModelData(
            v_template=np.zeros((3, 3)),
            shapedirs=np.zeros((3, 3, 2)),
            j_regressor=np.eye(3),
            parents=np.array([-1, 0, 1]),
            joint_names=("a", "a", "c"),
        )
