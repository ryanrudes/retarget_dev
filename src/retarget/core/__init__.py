"""Core primitives shared across retarget."""

from retarget.core.axes import (
    CoordinateAxis,
    SemanticAxis,
    SignedAxis,
    AxisConvention,
    Z_UP_AXES,
    Y_UP_AXES,
    MUJOCO_AXES,
    ISAAC_AXES,
)
from retarget.core.enums import (
    NameId,
    SubjectId,
    SegmentId,
    MarkerId,
    PatchId,
    TrackId,
    MarkerRole,
    RotationFormat,
    PoseFormat,
    QuaternionOrder,
)
from retarget.core.handles import (
    MarkerHandle,
    PatchHandle,
)
from retarget.core.contact_region import (
    ContactRegion,
    RectangularRegion,
    PolygonalRegion,
)
from retarget.core.keys import SegmentKey
from retarget.core.specs import (
    MarkerSpec,
    MarkerSetSpec,
    PatchDeclarationSpec,
    PatchSpec,
    PatchCalibrationSpec,
    SegmentSpec,
    SubjectSpec,
    SceneSpec,
)
from retarget.core.state import (
    SegmentPoseTrajectory,
    SceneState,
)
from retarget.core.schema import (
    GeneratedIds,
    Marker,
    Markers,
    Patch,
    Patches,
    Segment,
    Segments,
    Subject,
    Subjects,
    build_scene,
    marker_external_name,
    marker_from_external_name,
    marker_from_vicon_name,
    segment_external_name,
    subject_external_name,
)
from retarget.core.targets import MarkerTarget, PatchTarget, SegmentTarget
from retarget.core.transform import (
    RigidTransform,
)
from retarget.core.translation import (
    MarkerTranslation,
    BodyFrameTranslation,
    SemanticAxisTranslation,
)
from retarget.core.types import (
    Vec2,
    Vec3,
    Vec4,
    Vec6,
    Mat3,
    Points2,
    Points3,
    TimeVec3,
    TimeEntityVec3,
    TimeMat3,
    TimeQuat,
    TimeBool,
    TimeEntityBool,
)
from retarget.core.views import (
    SceneView,
    SubjectView,
    SegmentView,
    MarkerView,
    PatchView,
)

__all__ = [
    # Enums / IDs
    "NameId",
    "SubjectId",
    "SegmentId",
    "MarkerId",
    "PatchId",
    "TrackId",
    "MarkerRole",
    "RotationFormat",
    "PoseFormat",
    "QuaternionOrder",

    # Axes
    "SignedAxis",
    "CoordinateAxis",
    "SemanticAxis",
    "AxisConvention",
    "Z_UP_AXES",
    "Y_UP_AXES",
    "MUJOCO_AXES",
    "ISAAC_AXES",

    # Types
    "Vec2",
    "Vec3",
    "Vec4",
    "Vec6",
    "Mat3",
    "Points2",
    "Points3",
    "TimeVec3",
    "TimeEntityVec3",
    "TimeMat3",
    "TimeQuat",
    "TimeBool",
    "TimeEntityBool",

    # Transform
    "RigidTransform",

    # Handles / keys
    "MarkerHandle",
    "PatchHandle",
    "SegmentKey",

    # Contact regions
    "ContactRegion",
    "RectangularRegion",
    "PolygonalRegion",

    # Authoring schema
    "Markers",
    "Patches",
    "Segments",
    "Subjects",
    "Marker",
    "Patch",
    "Segment",
    "Subject",
    "build_scene",
    "GeneratedIds",
    "marker_external_name",
    "marker_from_external_name",
    "marker_from_vicon_name",
    "segment_external_name",
    "subject_external_name",

    # Specs
    "MarkerSpec",
    "MarkerSetSpec",
    "PatchDeclarationSpec",
    "PatchSpec",
    "PatchCalibrationSpec",
    "SegmentSpec",
    "SubjectSpec",
    "SceneSpec",

    # State
    "SegmentPoseTrajectory",
    "SceneState",

    # Targets
    "SegmentTarget",
    "MarkerTarget",
    "PatchTarget",

    # Translation
    "MarkerTranslation",
    "BodyFrameTranslation",
    "SemanticAxisTranslation",

    # Views
    "SceneView",
    "SubjectView",
    "SegmentView",
    "MarkerView",
    "PatchView",
]
