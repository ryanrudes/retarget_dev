"""Backend/manual geometry query example for the real Vicon scene.

Uses enum-keyed ``Demonstration`` access and internal mocap helpers such as
``_subject``, ``_segment``, ``_marker_positions``, and ``_patch_points`` for
backend validation. Those underscore helpers are not the public typed mapping
API.

The preferred public authoring example is ``new_api_example.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend_specs.ground_estimation_loader import load_ground_estimation_demo
from backend_specs.vicon_scene import LEFT_SHOE_SEGMENT
from backend_specs.vicon_vocab import (
    LeftShoeMarkerId,
    LeftShoePatchId,
    ViconSubjectId,
)
from demo_vocab import GroundEstimationTrackId

from retarget.core import RotationFormat
from retarget.demo import mocap as mocap_mod


REPO_ROOT = Path(__file__).resolve().parents[2]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"


def main() -> None:
    demo = load_ground_estimation_demo(UNBAGGED_DIR)
    clip = demo.slice_time(0.0, 1.0)
    mocap = clip[GroundEstimationTrackId.MOCAP]
    assert isinstance(mocap, mocap_mod.MocapTrack | mocap_mod.MocapTrackView)

    # Internal mocap query helpers (backend/manual only; not public API):
    left_shoe = (
        mocap._subject(ViconSubjectId.LEFT_SHOE)
        ._segment(LEFT_SHOE_SEGMENT)
    )

    heel_obs = left_shoe._marker_positions(LeftShoeMarkerId.HEEL)
    heel_model = left_shoe._marker_positions(
        LeftShoeMarkerId.HEEL,
        modeled=True,
    )
    heel_vel = left_shoe._marker_velocities(LeftShoeMarkerId.HEEL)
    shoe_quat = left_shoe.rotations(
        format=RotationFormat.QUATERNION_XYZW,
    )
    sole_target = left_shoe.segment_view.patch_target(LeftShoePatchId.SOLE)
    sole_point = left_shoe._patch_points(LeftShoePatchId.SOLE)
    sole_normal = left_shoe._patch_normals(LeftShoePatchId.SOLE)

    print("Clip timesteps:", len(mocap.timestamps))
    print("Heel observed:", heel_obs)
    print("Heel modeled:", heel_model)
    print("Heel velocity:", heel_vel)
    print("Shoe quaternion:", shoe_quat)
    print("Sole target:", sole_target)
    print("Sole point:", sole_point)
    print("Sole normal:", sole_normal)


if __name__ == "__main__":
    main()
