from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smpl.core.types import FloatArray


@dataclass(frozen=True, slots=True)
class SmplParams:
    """Per-sequence SMPL parameters, time-major over ``T`` frames.

    Shapes are validated structurally at construction. The shape coefficient count ``B`` and the
    joint count ``J`` are properties of the model, not the params, so only the array ranks and the
    shared frame count ``T`` are checked here.
    """

    betas: FloatArray
    """The shape coefficients of shape ``(B,)``, constant across the sequence."""

    pose: FloatArray
    """Axis-angle joint rotations of shape ``(T, J, 3)``; index ``0`` is the global orientation."""

    transl: FloatArray
    """The root translation of shape ``(T, 3)``, in world space."""

    def __post_init__(self) -> None:
        """Coerce arrays to ``float64`` and validate their ranks and frame counts."""
        object.__setattr__(self, "betas", np.asarray(self.betas, dtype=np.float64))
        object.__setattr__(self, "pose", np.asarray(self.pose, dtype=np.float64))
        object.__setattr__(self, "transl", np.asarray(self.transl, dtype=np.float64))

        if self.betas.ndim != 1:
            raise ValueError(f"betas must have shape (B,); got {self.betas.shape}.")
        if self.pose.ndim != 3 or self.pose.shape[2] != 3:
            raise ValueError(f"pose must have shape (T, J, 3); got {self.pose.shape}.")
        if self.transl.ndim != 2 or self.transl.shape[1] != 3:
            raise ValueError(f"transl must have shape (T, 3); got {self.transl.shape}.")
        if self.transl.shape[0] != self.pose.shape[0]:
            raise ValueError(
                f"pose and transl disagree on frame count T: {self.pose.shape[0]} vs {self.transl.shape[0]}."
            )

    @property
    def num_frames(self) -> int:
        """The number of frames ``T``."""
        return self.pose.shape[0]

    @property
    def num_joints(self) -> int:
        """The number of joints ``J`` implied by ``pose``."""
        return self.pose.shape[1]

    @property
    def num_betas(self) -> int:
        """The number of shape coefficients ``B``."""
        return self.betas.shape[0]
