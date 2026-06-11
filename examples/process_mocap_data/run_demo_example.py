from __future__ import annotations

from pathlib import Path

from mocap_specs import LEFT_SHOE_SEGMENT
from mocap_vocab import (
    LeftShoeMarkerId,
    LeftShoePatchId,
    ViconSubjectId,
)

from retarget.core import RotationFormat
from retarget.demo import mocap as mocap_mod

from demo_specs import load_ground_estimation_demo
from demo_vocab import GroundEstimationTrackId

REPO_ROOT = Path(__file__).resolve().parents[2]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"

demo = load_ground_estimation_demo(UNBAGGED_DIR)
clip = demo.slice_time(0.0, 1.0)
mocap = clip.get_track(GroundEstimationTrackId.MOCAP)
assert isinstance(mocap, mocap_mod.MocapTrack | mocap_mod.MocapTrackView)

left_shoe = mocap.subject(ViconSubjectId.LEFT_SHOE).segment(
    LEFT_SHOE_SEGMENT
)

heel_obs = left_shoe.marker_positions(LeftShoeMarkerId.HEEL)
heel_model = left_shoe.marker_positions(
    LeftShoeMarkerId.HEEL,
    modeled=True,
)
heel_vel = left_shoe.marker_velocities(LeftShoeMarkerId.HEEL)
shoe_quat = left_shoe.rotations(
    format=RotationFormat.QUATERNION_XYZW,
)
sole_point = left_shoe.patch_points(LeftShoePatchId.SOLE)
sole_normal = left_shoe.patch_normals(LeftShoePatchId.SOLE)

print("Clip timesteps:", len(mocap.timestamps))
print("Heel observed:", heel_obs)
print("Heel modeled:", heel_model)
print("Heel velocity:", heel_vel)
print("Shoe quaternion:", shoe_quat)
print("Sole point:", sole_point)
print("Sole normal:", sole_normal)
