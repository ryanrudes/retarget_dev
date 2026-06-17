from __future__ import annotations

from pathlib import Path

from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    RigidTransform,
    Segment,
    Segments,
    Subject,
    Subjects,
    build_scene,
)
from retarget.demo import Tracks, build_demonstration, load_mocap_track
from retarget.demo.mocap import MocapTrack
from retarget.io import UnbaggedDirectory


# ----------------------------
# User-authored typed schema
# ----------------------------


class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker
    lateral: Marker
    medial: Marker


class ShoePatches(Patches):
    sole: Patch
    heel_contact: Patch
    toe_contact: Patch


class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]


class HandMarkers(Markers):
    wrist: Marker
    index_tip: Marker
    thumb_tip: Marker


class HandPatches(Patches):
    palm: Patch
    index_contact: Patch


class HandSegments(Segments):
    hand: Segment[HandMarkers, HandPatches]


class MocapSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]
    right_shoe: Subject[ShoeSegments]
    right_hand: Subject[HandSegments]


class GroundEstimationSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]


class GroundEstimationTracks(Tracks):
    mocap: MocapTrack


# ----------------------------
# User-authored concrete instance
# ----------------------------

subjects = MocapSubjects(
    left_shoe=Subject(
        vicon_name="Left_Shoe_Improved",
        segments=ShoeSegments(
            shoe=Segment(
                vicon_name="Left_Shoe_Improved",
                markers=ShoeMarkers(
                    heel=Marker(vicon_name="left_shoe_heel"),
                    toe=Marker(vicon_name="left_shoe_toe"),
                    lateral=Marker(vicon_name="left_shoe_lateral"),
                    medial=Marker(vicon_name="left_shoe_medial"),
                ),
                patches=ShoePatches(
                    sole=Patch.rectangular(
                        label="sole",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.10,
                        height=0.25,
                        frame="sole_frame",
                    ),
                    heel_contact=Patch.rectangular(
                        label="heel_contact",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.05,
                        height=0.05,
                        frame="heel_frame",
                    ),
                    toe_contact=Patch(label="toe_contact_display"),
                ),
            )
        ),
    ),
    right_shoe=Subject(
        vicon_name="Right_Shoe_Improved",
        segments=ShoeSegments(
            shoe=Segment(
                vicon_name="Right_Shoe_Improved",
                markers=ShoeMarkers(
                    heel=Marker(vicon_name="right_shoe_heel"),
                    toe=Marker(vicon_name="right_shoe_toe"),
                    lateral=Marker(vicon_name="right_shoe_lateral"),
                    medial=Marker(vicon_name="right_shoe_medial"),
                ),
                patches=ShoePatches(
                    sole=Patch.rectangular(
                        label="sole",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.10,
                        height=0.25,
                        frame="sole_frame",
                    ),
                    heel_contact=Patch.rectangular(
                        label="heel_contact",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.05,
                        height=0.05,
                        frame="heel_frame",
                    ),
                    toe_contact=Patch(label="toe_contact_display"),
                ),
            )
        ),
    ),
    right_hand=Subject(
        vicon_name="Right_Hand_Improved",
        segments=HandSegments(
            hand=Segment(
                vicon_name="Right_Hand_Improved",
                markers=HandMarkers(
                    wrist=Marker(vicon_name="right_hand_wrist"),
                    index_tip=Marker(vicon_name="right_index_tip"),
                    thumb_tip=Marker(vicon_name="right_thumb_tip"),
                ),
                patches=HandPatches(
                    palm=Patch.rectangular(
                        label="palm",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.08,
                        height=0.08,
                        frame="palm_frame",
                    ),
                    index_contact=Patch.rectangular(
                        label="index_contact",
                        transform_segment_patch=RigidTransform.identity(),
                        width=0.03,
                        height=0.03,
                        frame="index_contact_frame",
                    ),
                ),
            )
        ),
    ),
)


# ----------------------------
# Compile into runtime spec
# ----------------------------

scene = build_scene(subjects)

left_shoe_spec = scene.subject("left_shoe")
shoe_spec = left_shoe_spec.segment("shoe")

heel_handle = shoe_spec.marker("heel")
sole_handle = shoe_spec.patch("sole")
sole_label = shoe_spec.patch_label("sole")
toe_target = shoe_spec.patch_target("toe_contact")
# shoe_spec.patch_spec("toe_contact") would raise because that patch is declaration-only.

shoe_target = shoe_spec.segment_target()
heel_target = shoe_spec.marker_target("heel")
sole_target = shoe_spec.patch_target("sole")

print("Runtime handles:", heel_handle, sole_handle)
print("Runtime patch label:", sole_label)
print("Runtime targets:", shoe_target, heel_target, sole_target, toe_target)


# ----------------------------
# Real-data handoff
# ----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
UNBAGGED_DIR = REPO_ROOT / "bags" / "ground_estimation" / "unbagged"

if UNBAGGED_DIR.is_dir():
    # The sample bag only contains the left shoe, so build a loadable subset
    # from the same authored schema instead of forcing absent subjects through
    # the loader.
    loadable_subjects = GroundEstimationSubjects(left_shoe=subjects["left_shoe"])
    loadable_scene = build_scene(loadable_subjects)
    root = UnbaggedDirectory(UNBAGGED_DIR)
    mocap = load_mocap_track(root, loadable_scene).with_rebased_time()
    demo = build_demonstration(GroundEstimationTracks(mocap=mocap))
    # demo.get_track("mocap") remains the dynamic lookup path.
    mocap = demo.typed_tracks["mocap"]  # MocapTrack
    left_shoe = mocap.subjects["left_shoe"]
    shoe = left_shoe.segments["shoe"]
    heel = shoe.markers["heel"]
    heel_positions = heel.positions()  # TimeVec3
    sole = shoe.patches["sole"]
    sole_points = sole.points()  # TimeVec3
    print("Loaded mocap timestamps:", mocap.timestamps[:5])
    print("Heel positions:", heel_positions[:3])
    print("Sole points:", sole_points[:3])
else:
    print(f"Skipping bag-backed demo; no local bag found at {UNBAGGED_DIR}")


# ----------------------------
# Stable runtime keys
# ----------------------------

# These are stable keys for runtime data structures:
#
# contacts: dict[PatchTarget, BoolArray]
# marker_observations: dict[MarkerTarget, FloatArray]
# segment_poses: dict[SegmentTarget, SegmentPoseTrajectory]
