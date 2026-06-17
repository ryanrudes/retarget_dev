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
    MarkerRole,
    RotationFormat,
    PoseFormat,
    QuaternionOrder,
)
from retarget.core.contact_region import (
    ContactRegion,
    RectangularRegion,
    PolygonalRegion,
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
from retarget.core.calibration import calibrate_patch_transform
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

    # Contact regions
    "ContactRegion",
    "RectangularRegion",
    "PolygonalRegion",

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

    # Calibration (backend helper)
    "calibrate_patch_transform",

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
