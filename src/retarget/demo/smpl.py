"""SMPL body-joint tracks (e.g. recovered from video) for the demonstration container.

A :class:`SmplTrack` holds world-frame SMPL joint positions over time -- the output of an
upstream video->SMPL estimator (GVHMR / WHAM / 4D-Humans), which retarget *consumes* the way it
consumes pre-unbagged mocap (the heavy estimation stays upstream). It is an ordinary
:class:`~retarget.demo.tracks.Track`, so it slots into a
:class:`~retarget.demo.demo.Demonstration` and synchronizes against mocap through the existing
energy/timeline machinery: :func:`smpl_joint_energy` is the first concrete scalar
:data:`~retarget.demo.alignment.SignalExtractor`, and it feeds ``estimate_sync`` unchanged.

This is the *track + temporal-sync* layer only. A later step can map these joints into the typed
scene schema (joints as markers, feet as ``patches``) to reuse contact detection, and add the
spatial registration (SMPL world -> Vicon world) that temporal sync does not do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import numpy as np

from retarget.core.formats import finite_difference_velocity, speed_from_velocity
from retarget.core.types import TimeVec3
from retarget.demo.alignment import EnergySignal, SignalExtractor
from retarget.demo.tracks import Track, indices_for_time_range


@runtime_checkable
class BodyModel[ParamsT](Protocol):
    """A SMPL-family body model: forward kinematics from params to world-frame joints.

    This is the contract retarget depends on; an external ``smpl`` library (SMPL / SMPL-X / SMPL+H
    / MANO / FLAME / …, the URDF-style vendored counterpart) implements it. ``forward_joints``
    returns world-frame joints ``(T, J, 3)`` for ``T`` frames of ``params`` (the model's own
    variant-specific parameter object — pose / shape / translation); ``joint_names`` names the ``J``
    joints in column order. An implementation may use torch internally but must return a numpy
    array, so retarget stays torch-free and the track is model-agnostic downstream.
    """

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def forward_joints(self, params: ParamsT) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SmplTrack(Track):
    """World-frame SMPL joint trajectories over time.

    ``joints`` is ``(T, J, 3)`` world-frame joint positions and ``joint_names`` names the ``J``
    joints (unique, in column order). ``timestamps`` is ``(T,)`` strictly increasing. retarget
    consumes pre-extracted SMPL output; the video->SMPL estimation is upstream.
    """

    joints: np.ndarray
    joint_names: tuple[str, ...]
    timestamps: np.ndarray
    nominal_hz_override: float | None = None

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints, dtype=np.float64)
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        names = tuple(str(name) for name in self.joint_names)
        if joints.ndim != 3 or joints.shape[2] != 3:
            raise ValueError(f"joints must have shape (T, J, 3), got {joints.shape}")
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be a 1D array")
        if joints.shape[0] != timestamps.shape[0]:
            raise ValueError(f"joints has {joints.shape[0]} frames but timestamps has {timestamps.shape[0]}")
        if len(names) != joints.shape[1]:
            raise ValueError(f"got {len(names)} joint_names for {joints.shape[1]} joints")
        if len(set(names)) != len(names):
            raise ValueError("joint_names must be unique")
        if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "joint_names", names)

    @classmethod
    def from_body[ParamsT](
        cls,
        model: BodyModel[ParamsT],
        params: ParamsT,
        timestamps: np.ndarray,
        *,
        nominal_hz_override: float | None = None,
    ) -> SmplTrack:
        """Build a track by running a body model's forward kinematics.

        ``model`` is any SMPL-family :class:`BodyModel` (e.g. from the vendored ``smpl`` library);
        ``params`` is that model's per-frame parameter object (variant-specific pose / shape /
        translation). The track holds the resulting world-frame joints, so contact + sync
        downstream are model- and variant-agnostic. This is the ergonomic entry point: pass a model
        and params instead of hand-computing joints.
        """
        joints = np.asarray(model.forward_joints(params), dtype=np.float64)
        return cls(
            joints=joints,
            joint_names=model.joint_names,
            timestamps=np.asarray(timestamps, dtype=np.float64),
            nominal_hz_override=nominal_hz_override,
        )

    def joint_index(self, name: str) -> int:
        """Column index of the named joint."""
        try:
            return self.joint_names.index(name)
        except ValueError:
            available = ", ".join(self.joint_names) or "(none)"
            raise KeyError(f"track has no joint {name!r} (available: {available})") from None

    def joint_positions(self, name: str) -> TimeVec3:
        """World-frame positions of one joint, shape ``(T, 3)``."""
        return cast(TimeVec3, self.joints[:, self.joint_index(name), :])

    def joint_velocities(self, name: str) -> TimeVec3:
        """World-frame finite-difference velocity of one joint, shape ``(T, 3)``."""
        return cast(TimeVec3, finite_difference_velocity(self.joint_positions(name), self.timestamps))

    def slice_time(self, start: float, stop: float) -> SmplTrack:
        return self._select(indices_for_time_range(self.timestamps, start, stop))

    def _select(self, indices: tuple[int, ...]) -> SmplTrack:
        idx = list(indices)
        return SmplTrack(
            joints=self.joints[idx],
            joint_names=self.joint_names,
            timestamps=self.timestamps[idx],
            nominal_hz_override=self.nominal_hz_override,
        )


def smpl_joint_energy(joint: str, *, axis: int | None = None) -> SignalExtractor:
    """A sync :data:`~retarget.demo.alignment.SignalExtractor` from a SMPL joint's motion.

    Returns the joint's 3-D speed (``axis=None``) or the magnitude of its velocity along one world
    axis (e.g. ``axis=2`` for the vertical bob of a footstep). Pair the *same* physical quantity on
    the reference track -- e.g. the corresponding mocap marker's vertical speed -- so the
    cross-correlation aligns like with like. The extractor is track-typed: it expects a
    :class:`SmplTrack` and raises on anything else.
    """

    def extract(track: Track) -> EnergySignal:
        if not isinstance(track, SmplTrack):
            raise TypeError(f"smpl_joint_energy expects a SmplTrack, got {type(track).__name__}")
        velocity = np.asarray(track.joint_velocities(joint), dtype=np.float64)
        values = speed_from_velocity(velocity) if axis is None else np.abs(velocity[:, axis])
        label = "speed" if axis is None else f"axis{axis}"
        return EnergySignal(timestamps=track.timestamps, values=values, name=f"smpl:{joint}:{label}")

    return extract
