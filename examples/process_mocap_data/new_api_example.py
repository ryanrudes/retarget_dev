from __future__ import annotations

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


# ----------------------------
# User-authored concrete instance
# ----------------------------

subjects = MocapSubjects(
    left_shoe=Subject(
        segments=ShoeSegments(
            shoe=Segment(
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
                    toe_contact=Patch(
                        label="toe_contact_display",
                    ),
                ),
            )
        )
    ),
    right_shoe=Subject(
        segments=ShoeSegments(
            shoe=Segment(
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
                    toe_contact=Patch(
                        label="toe_contact_display",
                    ),
                ),
            )
        )
    ),
    right_hand=Subject(
        segments=HandSegments(
            hand=Segment(
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
        )
    ),
)


# ----------------------------
# Compile into runtime spec
# ----------------------------

scene = build_scene(subjects)


# ----------------------------
# Runtime usage
# ----------------------------

left_shoe = scene.subject("left_shoe")
shoe = left_shoe.segment("shoe")

heel_marker = shoe.marker("heel")
sole_patch = shoe.patch("sole")
toe_target = shoe.patch_target("toe_contact")

shoe_target = shoe.segment_target()
heel_target = shoe.marker_target("heel")
sole_target = shoe.patch_target("sole")

# shoe.patch("toe_contact") would raise clearly because that patch is declaration-only.


# These are stable keys for runtime data structures:
#
# contacts: dict[PatchTarget, BoolArray]
# marker_observations: dict[MarkerTarget, FloatArray]
# segment_poses: dict[SegmentTarget, SegmentPoseTrajectory]
