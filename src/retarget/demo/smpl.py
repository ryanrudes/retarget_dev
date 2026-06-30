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
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
from fungeom import Face, Point3, Point3Bundle, Region2

from retarget.core import Patch, Subjects
from retarget.core.formats import finite_difference_velocity, speed_from_velocity
from retarget.core.schema.base import _schema_items
from retarget.core.keys import SegmentKey
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.transform import RigidTransform
from retarget.core.types import TimeVec3
from retarget.demo.alignment import EnergySignal, SignalExtractor
from retarget.demo.mocap import MocapTrack
from retarget.demo.tracks import Track, indices_for_time_range

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from retarget.core import Segment, Subject


@runtime_checkable
class BodyModel[ParamsT](Protocol):
    """A SMPL-family body model: forward kinematics from params to world-frame joints + bones.

    This is the contract retarget depends on; an external ``smpl`` library (SMPL / SMPL-X / SMPL+H
    / MANO / FLAME / …, the URDF-style vendored counterpart) implements it. For ``T`` frames of
    ``params`` (the model's own variant-specific parameter object — pose / shape / translation):
    ``forward_joints`` returns world-frame joint positions ``(T, J, 3)``; ``forward_transforms``
    returns full world-frame bone transforms ``(T, J, 4, 4)`` (the joint position is its
    ``[:3, 3]``) — these drive the typed scene's segment poses in :func:`smpl_mocap_track`.
    ``joint_names`` names the ``J`` joints in column order. An implementation may use torch
    internally but must return a numpy array, so retarget stays torch-free and model-agnostic.
    """

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    def forward_joints(self, params: ParamsT) -> np.ndarray: ...

    def forward_transforms(self, params: ParamsT) -> np.ndarray: ...


@runtime_checkable
class MeshBodyModel[ParamsT](BodyModel[ParamsT], Protocol):
    """A :class:`BodyModel` that also returns the posed mesh, for mesh-derived contact patches.

    ``forward_vertices`` returns world-frame mesh vertices ``(T, V, 3)`` (the linear-blend-skinned
    body surface). :func:`smpl_foot_patch` uses it to author a sole ``Patch`` from real foot
    vertices instead of an approximate bone-frame rectangle.
    """

    def forward_vertices(self, params: ParamsT) -> np.ndarray: ...


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


def smpl_mocap_track[ParamsT, SubjectsT: Subjects](
    model: BodyModel[ParamsT],
    params: ParamsT,
    subjects: SubjectsT,
    timestamps: np.ndarray,
) -> MocapTrack[SubjectsT]:
    """Drive a typed scene from a SMPL body model, so contact detection + sync reuse for SMPL.

    Each ``Segment`` in ``subjects`` whose ``mocap_name`` is one of the model's ``joint_names`` is
    posed by that bone's forward kinematics (``forward_transforms``); patches authored on it
    (``geometry=`` -> ``Face``) and its markers then behave exactly as in a mocap scene. The result
    is an ordinary :class:`~retarget.demo.mocap.MocapTrack` -- ``detect_contacts(...)`` on its
    patches, the marker/segment queries, and ``estimate_sync`` all work unchanged. SMPL becomes
    "just another subject" in the typed/fungeom substrate.
    """
    transforms = np.asarray(model.forward_transforms(params), dtype=np.float64)  # (T, J, 4, 4)
    times = np.asarray(timestamps, dtype=np.float64)
    index = {name: i for i, name in enumerate(model.joint_names)}
    segment_poses: dict[SegmentKey, SegmentPoseTrajectory] = {}
    subject: Subject[Any]
    segment: Segment[Any, Any]
    for subject_name, subject in _schema_items(subjects):
        for segment_name, segment in _schema_items(subject.segments):
            bone = segment.mocap_name
            if bone is None or bone not in index:
                raise KeyError(
                    f"segment {subject_name!r}/{segment_name!r} has mocap_name {bone!r}, not a "
                    f"joint of the model (available: {', '.join(model.joint_names)})"
                )
            world = transforms[:, index[bone]]  # (T, 4, 4)
            poses = tuple(
                RigidTransform.from_rotation_translation(rotation=world[t, :3, :3], translation=world[t, :3, 3])
                for t in range(world.shape[0])
            )
            segment_poses[SegmentKey(subject_name, segment_name)] = SegmentPoseTrajectory(poses)
    return MocapTrack(subjects=subjects, state=SceneState(segment_poses=segment_poses), timestamps=times)


def smpl_foot_patch[ParamsT](
    model: MeshBodyModel[ParamsT],
    rest_params: ParamsT,
    bone: str,
    vertex_indices: Sequence[int],
    *,
    label: str = "sole",
    padding: float = 0.0,
) -> Patch:
    """Author a sole ``Patch`` from a SMPL foot bone's mesh vertices, in the bone rest frame.

    ``vertex_indices`` select the foot-sole vertices of the model mesh (from the SMPL body-part
    segmentation -- real-model data); they are taken at ``rest_params`` (use zero pose and your
    subject's betas), transformed into the bone's rest frame, and fit to a fungeom ``Face`` -- an
    oriented plane (normal pointing away from the bone, toward the contact) + their convex-hull
    footprint (optionally ``padding``-expanded). The patch rides the SMPL bone via
    :func:`smpl_mocap_track`, so ``detect_contacts`` sees a real foot surface rather than a
    bone-frame rectangle. Author the foot ``Segment`` with ``mocap_name = bone`` and this patch.
    """
    transforms = np.asarray(model.forward_transforms(rest_params), dtype=np.float64)  # (T, J, 4, 4)
    vertices = np.asarray(model.forward_vertices(rest_params), dtype=np.float64)  # (T, V, 3)
    names = model.joint_names
    if bone not in names:
        raise KeyError(f"bone {bone!r} is not a joint of the model (available: {', '.join(names)}).")
    rest_bone = transforms[0, names.index(bone)]  # (4, 4)
    rotation, translation = rest_bone[:3, :3], rest_bone[:3, 3]
    world = vertices[0][list(vertex_indices)]  # (K, 3) world-frame rest foot vertices
    local = (world - translation) @ rotation  # (K, 3) bone-local = R^T @ (v - t)
    # The sole vertices are already concrete (bone-local), so the Face is plain fungeom data -- no
    # callable needed; their labels would be pure noise, so build the bundle label-free. The
    # outward normal points away from the bone origin (down, toward the ground).
    points = Point3Bundle.of(
        [Point3.at(float(local[i, 0]), float(local[i, 1]), float(local[i, 2])) for i in range(local.shape[0])]
    )
    plane = points.fit_plane().facing(Point3.at(0.0, 0.0, 0.0)).flipped()
    footprint = Region2.hull(points.in_frame(plane))
    if padding:
        footprint = footprint.offset(padding)
    return Patch(label=label, geometry=Face.on(plane, footprint))
