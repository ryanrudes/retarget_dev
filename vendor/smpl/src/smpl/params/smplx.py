from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smpl.core.types import FloatArray

SMPLX_NUM_BODY_JOINTS = 21
"""The number of articulated body joints in SMPL-X (everything below the global orientation, above the hands/face)."""

SMPLX_NUM_HAND_JOINTS = 15
"""The number of articulated joints per hand in SMPL-X."""

SMPLX_NUM_FACE_JOINTS = 3
"""The number of articulated face joints in SMPL-X (jaw + two eyes)."""

SMPLX_NUM_JOINTS = (
    1 + SMPLX_NUM_BODY_JOINTS + SMPLX_NUM_FACE_JOINTS + 2 * SMPLX_NUM_HAND_JOINTS
)
"""The total SMPL-X joint count ``Jx`` (``55``): global orient + body + jaw/eyes + both hands."""


@dataclass(frozen=True, slots=True)
class SmplxParams:
    """Per-sequence SMPL-X parameters, time-major over ``T`` frames.

    SMPL-X is the SMPL superset that adds articulated hands and a face (jaw + eyes) plus facial
    ``expression`` coefficients. The articulations are carried as separate, typed fields rather
    than one opaque pose block; :meth:`full_pose` assembles them into the single ``(T, Jx, 3)``
    axis-angle array the core forward-kinematics kernel consumes, in the canonical SMPL-X joint
    order ``[global_orient, body, jaw, left_eye, right_eye, left_hand, right_hand]``.

    Only the array ranks, the per-frame arrays' shared frame count ``T``, and the fixed SMPL-X
    joint counts are validated here; the shape (``B``) and expression (``E``) widths are free.
    """

    betas: FloatArray
    """The shape coefficients of shape ``(B,)``, constant across the sequence."""

    global_orient: FloatArray
    """The root (pelvis) axis-angle orientation of shape ``(T, 3)``; joint ``0`` of the assembled pose."""

    body_pose: FloatArray
    """The body joint axis-angle rotations of shape ``(T, 21, 3)`` (everything below the root)."""

    jaw_pose: FloatArray
    """The jaw axis-angle rotation of shape ``(T, 3)``."""

    leye_pose: FloatArray
    """The left-eye axis-angle rotation of shape ``(T, 3)``."""

    reye_pose: FloatArray
    """The right-eye axis-angle rotation of shape ``(T, 3)``."""

    left_hand_pose: FloatArray
    """The left-hand joint axis-angle rotations of shape ``(T, 15, 3)``."""

    right_hand_pose: FloatArray
    """The right-hand joint axis-angle rotations of shape ``(T, 15, 3)``."""

    expression: FloatArray
    """The facial expression coefficients of shape ``(E,)``, constant across the sequence."""

    transl: FloatArray
    """The root translation of shape ``(T, 3)``, in world space."""

    def __post_init__(self) -> None:
        """Coerce arrays to ``float64`` and validate their ranks, joint counts and frame counts."""
        for name in (
            "betas",
            "global_orient",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "left_hand_pose",
            "right_hand_pose",
            "expression",
            "transl",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float64))

        if self.betas.ndim != 1:
            raise ValueError(f"betas must have shape (B,); got {self.betas.shape}.")
        if self.expression.ndim != 1:
            raise ValueError(f"expression must have shape (E,); got {self.expression.shape}.")

        self._check_axis_angle("global_orient", self.global_orient)
        self._check_axis_angle("jaw_pose", self.jaw_pose)
        self._check_axis_angle("leye_pose", self.leye_pose)
        self._check_axis_angle("reye_pose", self.reye_pose)
        self._check_axis_angle("transl", self.transl)

        self._check_joint_pose("body_pose", self.body_pose, SMPLX_NUM_BODY_JOINTS)
        self._check_joint_pose("left_hand_pose", self.left_hand_pose, SMPLX_NUM_HAND_JOINTS)
        self._check_joint_pose("right_hand_pose", self.right_hand_pose, SMPLX_NUM_HAND_JOINTS)

        frames = self.global_orient.shape[0]
        for name in (
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "left_hand_pose",
            "right_hand_pose",
            "transl",
        ):
            other = getattr(self, name).shape[0]
            if other != frames:
                raise ValueError(
                    f"all per-frame arrays must share frame count T; global_orient has {frames} "
                    f"but {name} has {other}."
                )

    @staticmethod
    def _check_axis_angle(name: str, value: FloatArray) -> None:
        """Validate that a per-frame axis-angle vector field has shape ``(T, 3)``."""
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(f"{name} must have shape (T, 3); got {value.shape}.")

    @staticmethod
    def _check_joint_pose(name: str, value: FloatArray, num_joints: int) -> None:
        """Validate that a per-frame multi-joint axis-angle field has shape ``(T, num_joints, 3)``."""
        if value.ndim != 3 or value.shape[1:] != (num_joints, 3):
            raise ValueError(f"{name} must have shape (T, {num_joints}, 3); got {value.shape}.")

    def full_pose(self) -> FloatArray:
        """Assemble the canonical SMPL-X axis-angle pose of shape ``(T, 55, 3)``.

        The joints are concatenated in SMPL-X order: global orientation, the 21 body joints, the
        jaw, the left and right eyes, then the 15 left-hand and 15 right-hand joints. This is the
        single per-joint pose array the core forward-kinematics kernel consumes.
        """
        return np.concatenate(
            [
                self.global_orient[:, None, :],
                self.body_pose,
                self.jaw_pose[:, None, :],
                self.leye_pose[:, None, :],
                self.reye_pose[:, None, :],
                self.left_hand_pose,
                self.right_hand_pose,
            ],
            axis=1,
        )

    @classmethod
    def zeros(cls, num_frames: int, *, num_betas: int = 10, num_expression: int = 10) -> SmplxParams:
        """Build a zero (rest) SMPL-X parameter set with the canonical joint layout.

        Args:
            num_frames: The number of frames ``T`` (``>= 0``).
            num_betas: The number of shape coefficients ``B``.
            num_expression: The number of expression coefficients ``E``.

        Returns:
            A :class:`SmplxParams` whose every articulation, translation and coefficient is zero.
        """
        return cls(
            betas=np.zeros(num_betas),
            global_orient=np.zeros((num_frames, 3)),
            body_pose=np.zeros((num_frames, SMPLX_NUM_BODY_JOINTS, 3)),
            jaw_pose=np.zeros((num_frames, 3)),
            leye_pose=np.zeros((num_frames, 3)),
            reye_pose=np.zeros((num_frames, 3)),
            left_hand_pose=np.zeros((num_frames, SMPLX_NUM_HAND_JOINTS, 3)),
            right_hand_pose=np.zeros((num_frames, SMPLX_NUM_HAND_JOINTS, 3)),
            expression=np.zeros(num_expression),
            transl=np.zeros((num_frames, 3)),
        )

    @property
    def num_frames(self) -> int:
        """The number of frames ``T``."""
        return self.global_orient.shape[0]

    @property
    def num_joints(self) -> int:
        """The total assembled joint count ``Jx`` (always :data:`SMPLX_NUM_JOINTS`)."""
        return SMPLX_NUM_JOINTS

    @property
    def num_betas(self) -> int:
        """The number of shape coefficients ``B``."""
        return self.betas.shape[0]

    @property
    def num_expression(self) -> int:
        """The number of expression coefficients ``E``."""
        return self.expression.shape[0]
