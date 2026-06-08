from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from retarget.core.enums import MarkerId, PatchId, SegmentId, SubjectId
from retarget.core.handles import MarkerHandle, PatchHandle
from retarget.core.specs import PatchSpec, SceneSpec, SegmentSpec, SubjectSpec
from retarget.core.state import SceneState, SegmentPoseTrajectory
from retarget.core.transform import RigidTransform
from retarget.core.types import Vec3


@dataclass(frozen=True, slots=True)
class SceneView:
    """Scene-level view over a static scene spec and runtime scene state."""

    spec: SceneSpec
    state: SceneState

    def subject(self, subject: SubjectId | SubjectSpec) -> SubjectView:
        """Return a subject-scoped view over this scene state."""
        if isinstance(subject, SubjectSpec):
            subject_spec = subject
        else:
            subject_spec = self.spec.subject(subject)
        return SubjectView(subject_spec=subject_spec, state=self.state)

    @overload
    def segment[M: MarkerId, P: PatchId](
        self,
        subject: SubjectId | SubjectSpec,
        segment: SegmentSpec[M, P],
    ) -> SegmentView[M, P]: ...

    @overload
    def segment(
        self,
        subject: SubjectId | SubjectSpec,
        segment: SegmentId,
    ) -> SegmentView[Any, Any]: ...

    def segment(
        self,
        subject: SubjectId | SubjectSpec,
        segment: SegmentSpec[Any, Any] | SegmentId,
    ) -> SegmentView[Any, Any]:
        """
        Return a segment view within a subject namespace.

        Passing a SegmentSpec preserves marker/patch generic types.
        Passing a SegmentId performs runtime lookup and returns SegmentView[Any, Any].
        """
        return self.subject(subject).segment(segment)


@dataclass(frozen=True, slots=True)
class SubjectView:
    """Subject-scoped view over a static subject spec and runtime scene state."""

    subject_spec: SubjectSpec
    state: SceneState

    @property
    def subject_id(self) -> SubjectId:
        """The subject identifier for this viewed subject."""
        return self.subject_spec.subject

    @overload
    def segment[M: MarkerId, P: PatchId](
        self,
        segment: SegmentSpec[M, P],
    ) -> SegmentView[M, P]: ...

    @overload
    def segment(
        self,
        segment: SegmentId,
    ) -> SegmentView[Any, Any]: ...

    def segment(
        self,
        segment: SegmentSpec[Any, Any] | SegmentId,
    ) -> SegmentView[Any, Any]:
        """
        Return a segment view for this subject.

        Passing a SegmentSpec preserves marker/patch generic types.
        Passing a SegmentId performs runtime lookup through this subject's
        SubjectSpec and returns SegmentView[Any, Any], because a bare SegmentId
        does not carry marker/patch type parameters.
        """
        if isinstance(segment, SegmentSpec):
            segment_spec = segment
        else:
            # Some concrete SubjectSpec subclasses may have fields whose names would
            # otherwise shadow base-class lookup methods. Call the base method explicitly
            # so SegmentId lookup always uses SubjectSpec.segment(...).
            segment_spec = SubjectSpec.segment(self.subject_spec, segment)
        return SegmentView(
            subject_id=self.subject_id,
            spec=segment_spec,
            trajectory=self.state.pose(
                subject=self.subject_id,
                segment=segment_spec.segment,
            ),
        )


@dataclass(frozen=True, slots=True)
class SegmentView[M: MarkerId, P: PatchId]:
    """Typed runtime view of a segment within one subject namespace."""

    subject_id: SubjectId
    spec: SegmentSpec[M, P]
    trajectory: SegmentPoseTrajectory

    @property
    def segment_id(self) -> SegmentId:
        """The segment identifier for this viewed segment."""
        return self.spec.segment

    def pose_at(self, timestep: int) -> RigidTransform:
        """Return the world-from-segment transform at one timestep."""
        return self.trajectory.at(timestep)

    def marker(self, marker: M) -> MarkerView[M, P]:
        """Return a typed runtime view of one marker on this segment."""
        return MarkerView(
            segment_view=self,
            handle=self.spec.marker(marker),
        )

    def patch(self, patch: P) -> PatchView[M, P]:
        """Return a typed runtime view of one patch on this segment."""
        return PatchView(
            segment_view=self,
            handle=self.spec.patch(patch),
        )

    def point_to_world(self, point_segment: Vec3, timestep: int) -> Vec3:
        """Transform a segment-frame point into the world frame."""
        return self.pose_at(timestep).apply(point_segment)

    def vector_to_world(self, vector_segment: Vec3, timestep: int) -> Vec3:
        """Transform a segment-frame vector into the world frame."""
        return self.pose_at(timestep).rotation @ vector_segment


@dataclass(frozen=True, slots=True)
class MarkerView[M: MarkerId, P: PatchId]:
    """Runtime view of a model marker attached to a typed segment."""

    segment_view: SegmentView[M, P]
    handle: MarkerHandle[M]

    @property
    def position_segment(self) -> Vec3:
        """Marker position in the segment frame."""
        return self.segment_view.spec.marker_position(self.handle.marker)

    def position_world_at(self, timestep: int) -> Vec3:
        """Marker position in the world frame at one timestep."""
        return self.segment_view.point_to_world(
            point_segment=self.position_segment,
            timestep=timestep,
        )


@dataclass(frozen=True, slots=True)
class PatchView[M: MarkerId, P: PatchId]:
    """Runtime view of a contact patch attached to a typed segment."""

    segment_view: SegmentView[M, P]
    handle: PatchHandle[P]

    @property
    def spec(self) -> PatchSpec[P]:
        """Segment-local patch specification."""
        return self.segment_view.spec.patch_spec(self.handle.patch)

    @property
    def transform_segment_patch(self) -> RigidTransform:
        """Segment-from-patch transform."""
        return self.spec.transform_segment_patch

    def transform_world_patch_at(self, timestep: int) -> RigidTransform:
        """World-from-patch transform at one timestep."""
        return self.segment_view.pose_at(timestep).compose(
            self.transform_segment_patch
        )

    def contact_point_world_at(self, timestep: int) -> Vec3:
        """Patch origin/contact point in the world frame at one timestep."""
        return self.transform_world_patch_at(timestep).translation

    def normal_world_at(self, timestep: int) -> Vec3:
        """Patch normal in the world frame at one timestep."""
        return self.transform_world_patch_at(timestep).rotation[:, 2]

    def world_to_patch_at(self, point_world: Vec3, timestep: int) -> Vec3:
        """Transform a world-frame point into this patch frame."""
        return self.transform_world_patch_at(timestep).inverse().apply(point_world)

    def contains_world_point_at(
        self,
        point_world: Vec3,
        timestep: int,
        *,
        plane_tolerance: float = 1e-3,
    ) -> bool:
        """Return whether a world point lies inside this patch at a timestep."""
        point_patch = self.world_to_patch_at(
            point_world=point_world,
            timestep=timestep,
        )

        if abs(float(point_patch[2])) > plane_tolerance:
            return False

        return self.spec.region.contains(np.asarray(point_patch[:2]))
