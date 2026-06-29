from __future__ import annotations

from dataclasses import dataclass

from smpl.core.types import FloatArray
from smpl.io.load import BodyModelData
from smpl.models import base
from smpl.params.smpl import SmplParams


@dataclass(frozen=True, slots=True)
class SmplModel:
    """A SMPL body model: shape + pose params in, world joint transforms out.

    This structurally satisfies retarget's ``BodyModel`` protocol (a ``joint_names`` property plus
    ``forward_joints(params) -> (T, J, 3)``) without importing retarget. Forward kinematics is
    joint-level only: rest joints are read off the shaped template by the regressor, then the
    kinematic tree is posed; pose blendshapes and skinning (which only affect *vertices*) are not
    involved.
    """

    body: BodyModelData
    """The structured arrays defining the body model."""

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

    def forward_transforms(self, params: SmplParams) -> FloatArray:
        """Run forward kinematics, returning world bone transforms of shape ``(T, J, 4, 4)``.

        The root translation is added into the translation block of every joint transform, so
        ``forward_transforms(params)[..., :3, 3]`` are the world joint positions.
        """
        return base.forward_transforms(self.body, params.betas, params.pose, params.transl)

    def forward_joints(self, params: SmplParams) -> FloatArray:
        """Return the world joint positions of shape ``(T, J, 3)``."""
        return self.forward_transforms(params)[:, :, :3, 3]
