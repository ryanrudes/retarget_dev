from __future__ import annotations

from dataclasses import dataclass

from smpl.core.types import FloatArray
from smpl.io.load import BodyModelData, synthetic_model
from smpl.models import base
from smpl.params.smplx import SMPLX_NUM_JOINTS, SmplxParams


@dataclass(frozen=True, slots=True)
class SmplxModel:
    """A SMPL-X body model: shape + (body/face/hand) pose params in, world joint transforms out.

    SMPL-X is the SMPL superset with articulated hands and a face, so it has more joints (``55``)
    but the *math is identical*: this model reuses the very same core LBS forward kernel as
    :class:`~smpl.models.smpl.SmplModel`, differing only in that :class:`SmplxParams` packs the
    extra articulations and assembles them via :meth:`SmplxParams.full_pose`. Like ``SmplModel`` it
    structurally satisfies retarget's ``BodyModel`` protocol without importing retarget. Forward
    kinematics is joint-level only; pose blendshapes, skinning and expression (which affect only
    *vertices*) are not involved.
    """

    body: BodyModelData
    """The structured arrays defining the body model (an SMPL-X-shaped tree of ``55`` joints)."""

    @property
    def joint_names(self) -> tuple[str, ...]:
        """The name of each joint, ordered to match the FK outputs' ``J`` axis."""
        return self.body.joint_names

    def v_shaped(self, betas: FloatArray) -> FloatArray:
        """Return the shaped template ``v_template + shapedirs @ betas`` of shape ``(V, 3)``."""
        return base.shaped_template(self.body, betas)

    def rest_joints(self, betas: FloatArray) -> FloatArray:
        """Return the rest-pose joints ``j_regressor @ v_shaped(betas)`` of shape ``(J, 3)``."""
        return base.rest_joints(self.body, betas)

    def forward_transforms(self, params: SmplxParams) -> FloatArray:
        """Run forward kinematics, returning world bone transforms of shape ``(T, J, 4, 4)``.

        The SMPL-X articulations are assembled into the canonical ``(T, 55, 3)`` pose, then posed
        through the shared kernel; the root translation is added into every joint transform, so
        ``forward_transforms(params)[..., :3, 3]`` are the world joint positions.
        """
        return base.forward_transforms(self.body, params.betas, params.full_pose(), params.transl)

    def forward_joints(self, params: SmplxParams) -> FloatArray:
        """Return the world joint positions of shape ``(T, J, 3)``."""
        return self.forward_transforms(params)[:, :, :3, 3]


def synthetic_smplx_model(num_betas: int = 10) -> BodyModelData:
    """Build a tiny, valid SMPL-X-shaped body model for tests (no licensed model files needed).

    This reuses :func:`~smpl.io.load.synthetic_model` with the full SMPL-X joint count, yielding a
    rooted chain of ``55`` joints (root regressed to the origin) that :class:`SmplxModel` and a
    canonical :class:`SmplxParams` (e.g. ``SmplxParams.zeros(T)``) can be exercised against. The
    tree is a simple chain rather than the real SMPL-X branching skeleton, which is all the
    joint-level FK invariants need.

    Args:
        num_betas: The number of shape coefficients ``B`` (``>= 1``).

    Returns:
        A validated :class:`~smpl.io.load.BodyModelData` with :data:`SMPLX_NUM_JOINTS` joints.
    """
    return synthetic_model(num_joints=SMPLX_NUM_JOINTS, num_betas=num_betas)
