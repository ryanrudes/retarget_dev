from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from smpl.core.types import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class BodyModelData:
    """The structured arrays that define a SMPL-family body model.

    SMPL, SMPL-X, SMPL+H, MANO and friends share one math: a template mesh deformed by shape
    blendshapes, a joint regressor that reads rest-pose joints off the (shaped) mesh, and a
    kinematic tree. This holds exactly those joint-level arrays; pose blendshapes, skinning
    weights and mesh faces (needed for vertices, not joints) are deliberately out of scope.
    """

    v_template: FloatArray
    """The rest-pose template mesh vertices of shape ``(V, 3)``."""

    shapedirs: FloatArray
    """The shape blendshape basis of shape ``(V, 3, B)``; ``shapedirs @ betas`` displaces the template."""

    j_regressor: FloatArray
    """The joint regressor of shape ``(J, V)``; ``j_regressor @ v_shaped`` yields rest joints ``(J, 3)``."""

    parents: IntArray
    """The parent index per joint of shape ``(J,)``; ``parents[0] == -1`` marks the root."""

    joint_names: tuple[str, ...]
    """The name of each joint, ordered to match ``parents`` and the rows of ``j_regressor``."""

    def __post_init__(self) -> None:
        """Validate that the arrays are mutually consistent and describe a valid tree."""
        object.__setattr__(self, "v_template", np.asarray(self.v_template, dtype=np.float64))
        object.__setattr__(self, "shapedirs", np.asarray(self.shapedirs, dtype=np.float64))
        object.__setattr__(self, "j_regressor", np.asarray(self.j_regressor, dtype=np.float64))
        object.__setattr__(self, "parents", np.asarray(self.parents, dtype=np.intp))
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))

        if self.v_template.ndim != 2 or self.v_template.shape[1] != 3:
            raise ValueError(f"v_template must have shape (V, 3); got {self.v_template.shape}.")
        num_vertices = self.v_template.shape[0]

        if self.shapedirs.ndim != 3 or self.shapedirs.shape[:2] != (num_vertices, 3):
            raise ValueError(
                f"shapedirs must have shape ({num_vertices}, 3, B); got {self.shapedirs.shape}."
            )

        if self.j_regressor.ndim != 2 or self.j_regressor.shape[1] != num_vertices:
            raise ValueError(
                f"j_regressor must have shape (J, {num_vertices}); got {self.j_regressor.shape}."
            )
        num_joints = self.j_regressor.shape[0]

        if self.parents.shape != (num_joints,):
            raise ValueError(f"parents must have shape ({num_joints},); got {self.parents.shape}.")
        if num_joints == 0 or self.parents[0] != -1:
            raise ValueError("parents[0] must be -1 for the root joint.")
        for i in range(1, num_joints):
            if not 0 <= int(self.parents[i]) < i:
                raise ValueError(
                    f"parents[{i}] must reference an earlier joint in [0, {i}); got {int(self.parents[i])}."
                )

        if len(self.joint_names) != num_joints:
            raise ValueError(
                f"got {len(self.joint_names)} joint_names for {num_joints} joints."
            )
        if len(set(self.joint_names)) != num_joints:
            raise ValueError("joint_names must be unique.")

    @property
    def num_vertices(self) -> int:
        """The number of template vertices ``V``."""
        return self.v_template.shape[0]

    @property
    def num_joints(self) -> int:
        """The number of joints ``J``."""
        return self.j_regressor.shape[0]

    @property
    def num_betas(self) -> int:
        """The number of shape coefficients ``B``."""
        return self.shapedirs.shape[2]


def _pick(arrays: dict[str, Any], *names: str) -> Any:
    """Return the first present entry among ``names`` (case-insensitive), or raise."""
    lowered = {key.lower(): key for key in arrays}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return arrays[key]
    raise KeyError(f"none of the expected keys {names!r} were found in {tuple(arrays)!r}.")


def _as_dense(value: Any) -> FloatArray:
    """Densify a possibly-sparse array (e.g. a saved scipy ``J_regressor``) to ``float64``."""
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def load_npz(path: str | PathLike[str]) -> BodyModelData:
    """Load a SMPL-family ``.npz`` file into a :class:`BodyModelData`.

    The loader is tolerant of the key-naming variations seen across SMPL exports. The expected
    (case-insensitive) keys are:

    - ``v_template`` — template vertices ``(V, 3)``.
    - ``shapedirs`` — shape basis ``(V, 3, B)``.
    - ``J_regressor`` (or ``j_regressor``) — joint regressor ``(J, V)``; densified if sparse.
    - ``kintree_table`` (row 0 is the parent of each joint) or ``parents`` — the kinematic tree.
      The root's parent (any value ``< 0`` or ``>= J``) is normalised to ``-1``.
    - ``joint_names`` (optional) — otherwise joints are named ``joint_0 … joint_{J-1}``.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        The loaded, validated body model.
    """
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    v_template = np.asarray(_pick(arrays, "v_template"), dtype=np.float64)
    shapedirs = np.asarray(_pick(arrays, "shapedirs"), dtype=np.float64)
    j_regressor = _as_dense(_pick(arrays, "J_regressor", "j_regressor"))
    num_joints = j_regressor.shape[0]

    try:
        parents_raw = np.asarray(_pick(arrays, "parents"))
    except KeyError:
        kintree = np.asarray(_pick(arrays, "kintree_table", "kintree"))
        parents_raw = kintree[0]
    parents = np.asarray(parents_raw, dtype=np.int64).reshape(-1)
    parents = np.where((parents < 0) | (parents >= num_joints), -1, parents).astype(np.intp)

    try:
        raw_names = _pick(arrays, "joint_names", "joints_name")
        joint_names = tuple(str(name) for name in np.asarray(raw_names).reshape(-1).tolist())
    except KeyError:
        joint_names = tuple(f"joint_{i}" for i in range(num_joints))

    return BodyModelData(
        v_template=v_template,
        shapedirs=shapedirs,
        j_regressor=j_regressor,
        parents=parents,
        joint_names=joint_names,
    )


def synthetic_model(num_joints: int = 3, num_betas: int = 2) -> BodyModelData:
    """Build a tiny, valid SMPL-family model for tests (no licensed model files needed).

    The joints form a simple parent chain ``0 <- 1 <- 2 <- …`` rooted at the origin. The joint
    regressor is a one-hot selector over the first ``num_joints`` template vertices, which are
    laid out along the +x axis at integer offsets so the rest joints are distinct and the root
    sits at the origin. Shape blendshapes are small, deterministic, and non-degenerate.

    Args:
        num_joints: The number of joints ``J`` (``>= 1``).
        num_betas: The number of shape coefficients ``B`` (``>= 1``).

    Returns:
        A validated :class:`BodyModelData`.
    """
    if num_joints < 1:
        raise ValueError(f"num_joints must be >= 1; got {num_joints}.")
    if num_betas < 1:
        raise ValueError(f"num_betas must be >= 1; got {num_betas}.")

    # A couple of extra "free" vertices beyond the regressed joints to look like a real mesh.
    num_vertices = num_joints + 2

    v_template = np.zeros((num_vertices, 3), dtype=np.float64)
    v_template[:num_joints, 0] = np.arange(num_joints, dtype=np.float64)  # joints along +x, root at 0
    v_template[num_joints:, 1] = np.arange(1, num_vertices - num_joints + 1, dtype=np.float64)

    j_regressor = np.zeros((num_joints, num_vertices), dtype=np.float64)
    j_regressor[np.arange(num_joints), np.arange(num_joints)] = 1.0

    # Small, deterministic, distinct shape directions; tiny so they stay clearly a perturbation.
    shapedirs = np.zeros((num_vertices, 3, num_betas), dtype=np.float64)
    ramp = np.linspace(0.01, 0.05, num=num_vertices * 3 * num_betas, dtype=np.float64)
    shapedirs[...] = ramp.reshape(num_vertices, 3, num_betas)

    parents = np.empty((num_joints,), dtype=np.intp)
    parents[0] = -1
    parents[1:] = np.arange(num_joints - 1, dtype=np.intp)

    base_names = ("pelvis", "spine", "head", "left_hand", "right_hand")
    joint_names = tuple(
        base_names[i] if i < len(base_names) else f"joint_{i}" for i in range(num_joints)
    )

    return BodyModelData(
        v_template=v_template,
        shapedirs=shapedirs,
        j_regressor=j_regressor,
        parents=parents,
        joint_names=joint_names,
    )
