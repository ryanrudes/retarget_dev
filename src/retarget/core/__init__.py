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
    MarkerRole,
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
from retarget.core.targets import PatchTarget
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
    "MarkerRole",
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

    # Specs
    "MarkerSpec",
    "MarkerSetSpec",
    "PatchSpec",
    "PatchCalibrationSpec",
    "SegmentSpec",
    "SubjectSpec",
    "SceneSpec",

    # State
    "SegmentPoseTrajectory",
    "SceneState",

    # Targets
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
