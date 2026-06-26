"""Core primitives shared across retarget."""

from retarget.core.axes import (
    CoordinateAxis,
    SemanticAxis,
    SignedAxis,
    SignedSemanticAxis,
    AxisConvention,
    Z_UP_AXES,
    Y_UP_AXES,
    MUJOCO_AXES,
    ISAAC_AXES,
)
from retarget.core.enums import (
    MarkerRole,
    RotationFormat,
    PoseFormat,
    QuaternionOrder,
)
from retarget.core.keys import SegmentKey
from retarget.core.state import (
    SegmentPoseTrajectory,
    SceneState,
)
from retarget.core.schema import (
    Marker,
    Markers,
    Patch,
    Patches,
    Segment,
    Segments,
    Subject,
    Subjects,
    bind_scene,
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

__all__ = [
    # Value enums
    "MarkerRole",
    "RotationFormat",
    "PoseFormat",
    "QuaternionOrder",
    # Axes
    "SignedAxis",
    "SignedSemanticAxis",
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
    # Keys
    "SegmentKey",
    # Authoring schema + bound runtime surface
    "Markers",
    "Patches",
    "Segments",
    "Subjects",
    "Marker",
    "Patch",
    "Segment",
    "Subject",
    "bind_scene",
    # State
    "SegmentPoseTrajectory",
    "SceneState",
    # Targets (stable runtime keys)
    "SegmentTarget",
    "MarkerTarget",
    "PatchTarget",
    # Translation
    "MarkerTranslation",
    "BodyFrameTranslation",
    "SemanticAxisTranslation",
]
