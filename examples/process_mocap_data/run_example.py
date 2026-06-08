from __future__ import annotations

from pathlib import Path

from mocap_specs import VICON_SCENE
from mocap_vocab import (
    LeftShoeMarkerId,
    LeftShoePatchId,
    LeftShoeSegmentId,
    ViconSubjectId,
)

from retarget.core import SceneView
from retarget.io import (
    UnbaggedDirectory,
    iter_vicon_marker_frames,
    load_scene_state,
    marker_position,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"

export = UnbaggedDirectory(UNBAGGED_DIR)
state = load_scene_state(export, VICON_SCENE)
scene = SceneView(spec=VICON_SCENE, state=state)

# You can resolve by SegmentSpec for stronger static typing:
# left_shoe = scene.subject(ViconSubjectId.LEFT_SHOE).segment(LEFT_SHOE_SEGMENT)
# Or resolve by SegmentId for ergonomic runtime lookup:
left_shoe = scene.subject(ViconSubjectId.LEFT_SHOE).segment(LeftShoeSegmentId.LEFT_SHOE)
timestep = 0

heel = left_shoe.marker(LeftShoeMarkerId.HEEL)
sole = left_shoe.patch(LeftShoePatchId.SOLE)

heel_world = heel.position_world_at(timestep)
sole_point_world = sole.contact_point_world_at(timestep)
sole_normal_world = sole.normal_world_at(timestep)

for marker_frame in iter_vicon_marker_frames(export):
    observed_heel = marker_position(
        marker_frame,
        segment=left_shoe,
        marker=LeftShoeMarkerId.HEEL,
    )

    print(observed_heel)

print("Timesteps:", state.num_timesteps)
print("Heel world (model):", heel_world)
print("Heel world (observed):", observed_heel)
print("Sole point world:", sole_point_world)
print("Sole normal world:", sole_normal_world)
print("Scene subjects:", tuple(VICON_SCENE.iter_subjects()))
print("Scene segments:", tuple(VICON_SCENE.iter_segments()))
