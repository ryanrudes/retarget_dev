from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from scipy.spatial.transform import RigidTransform as ScipyRigidTransform
from scipy.spatial.transform import Rotation

from retarget.core.types import Vec3, Mat3


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """
    Transform from a child frame into a parent frame.

        p_parent = R_parent_child @ p_child + t_parent_child

    This class is a domain wrapper around scipy.spatial.transform.RigidTransform.
    """

    scipy: ScipyRigidTransform

    @staticmethod
    def identity() -> RigidTransform:
        return RigidTransform(
            ScipyRigidTransform.identity()
        )

    @staticmethod
    def from_rotation_translation(
        rotation: Mat3,
        translation: Vec3,
    ) -> RigidTransform:
        return RigidTransform(
            ScipyRigidTransform.from_components(
                translation=translation,
                rotation=Rotation.from_matrix(rotation),
            )
        )

    @staticmethod
    def from_translation(translation: Vec3) -> RigidTransform:
        return RigidTransform.from_rotation_translation(
            rotation=Rotation.identity().as_matrix(),
            translation=translation,
        )

    @property
    def rotation(self) -> Mat3:
        return cast(Mat3, self.scipy.rotation.as_matrix())

    @property
    def translation(self) -> Vec3:
        return cast(Vec3, self.scipy.translation)

    def apply(self, point: Vec3) -> Vec3:
        return cast(Vec3, self.scipy.apply(point))

    def inverse(self) -> RigidTransform:
        return RigidTransform(self.scipy.inv())

    def compose(self, other: RigidTransform) -> RigidTransform:
        """
        If self is T_A_B and other is T_B_C, return T_A_C.
        """
        return RigidTransform(self.scipy * other.scipy)