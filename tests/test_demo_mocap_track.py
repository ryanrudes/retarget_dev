from __future__ import annotations

import numpy as np
import pytest

from fungeom import Face, Point3, Point3Bundle, Region2

from dataclasses import dataclass

from conftest import (
    SEGMENT,
    SUBJECT,
    demo_segment,
    make_demo_patch_target,
    make_mocap_track,
)
from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    RigidTransform,
    RotationFormat,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    Segments,
    Subject,
    Subjects,
)
from retarget.demo.contact import ContactTrack
from retarget.demo.mocap import MocapTrack
from retarget.io import ViconMarkersFrame

_POSITIONS = {
    "heel": np.array([0.0, 0.0, 0.0]),
    "toe": np.array([1.0, 0.0, 0.0]),
    "mid": np.array([0.0, 1.0, 0.0]),
}


@dataclass(frozen=True, slots=True)
class _TwoPatchPatches(Patches):
    sole: Patch
    heel_cap: Patch


@dataclass(frozen=True, slots=True)
class _TwoPatchMarkers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker


@dataclass(frozen=True, slots=True)
class _TwoPatchSegments(Segments):
    segment: Segment[_TwoPatchMarkers, _TwoPatchPatches]


@dataclass(frozen=True, slots=True)
class _TwoPatchSubjects(Subjects):
    subject: Subject[_TwoPatchSegments]


def _two_patch_track() -> MocapTrack[_TwoPatchSubjects]:
    heel = Marker(mocap_name="heel", position_segment=_POSITIONS["heel"])
    toe = Marker(mocap_name="toe", position_segment=_POSITIONS["toe"])
    mid = Marker(mocap_name="mid", position_segment=_POSITIONS["mid"])
    # Two data-form patches sharing the +z fitted plane: a 1x1 sole and a 0.5x0.5 heel cap lifted 5cm.
    plane = Point3Bundle.of([heel, toe, mid]).fit_plane().facing(Point3.at(0.0, 0.0, 1.0))
    sole_face = Face.on(plane, Region2.rectangle(1.0, 1.0))
    heel_cap_face = Face.on(plane.offset(0.05), Region2.rectangle(0.5, 0.5))
    subjects = _TwoPatchSubjects(
        subject=Subject(
            segments=_TwoPatchSegments(
                segment=Segment(
                    markers=_TwoPatchMarkers(heel=heel, toe=toe, mid=mid),
                    patches=_TwoPatchPatches(
                        sole=Patch(label="sole", geometry=sole_face),
                        heel_cap=Patch(label="heel_cap", geometry=heel_cap_face),
                    ),
                )
            )
        )
    )
    poses = tuple(RigidTransform.from_translation(np.array([float(i), 0.0, 0.0])) for i in range(3))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses)})
    return MocapTrack(subjects=subjects, state=state, timestamps=np.arange(3, dtype=np.float64) * 0.1)


# -- construction / timeline ---------------------------------------------------


def test_mocap_track_validates_timestamp_length() -> None:
    track = make_mocap_track()
    with pytest.raises(ValueError, match="does not match"):
        MocapTrack(subjects=track.subjects, state=track.state, timestamps=np.array([0.0, 0.1]))


def test_mocap_track_rejects_duplicate_timestamps() -> None:
    track = make_mocap_track()
    with pytest.raises(ValueError, match="strictly increasing"):
        MocapTrack(
            subjects=track.subjects,
            state=track.state,
            timestamps=np.array([0.0, 0.1, 0.1]),
            marker_frames=track.marker_frames,
        )


def test_slice_time_returns_mocap_track_with_expected_timestamps() -> None:
    track = make_mocap_track()
    clip = track.slice_time(0.1, 0.25)
    assert isinstance(clip, MocapTrack)
    np.testing.assert_allclose(clip.timestamps, np.array([0.1, 0.2]))


def test_nearest_index_on_empty_track_raises() -> None:
    track = make_mocap_track().slice_time(100.0, 101.0)
    with pytest.raises(IndexError, match="empty"):
        track.nearest_index(0.0)


# -- segment / marker / patch queries ------------------------------------------


def test_translations_and_rotations_shapes() -> None:
    segment = demo_segment(make_mocap_track())
    assert segment.translations().shape == (3, 3)
    assert segment.rotations().shape == (3, 3, 3)
    assert segment.rotations(format=RotationFormat.QUATERNION_XYZW).shape == (3, 4)


def test_marker_positions_modeled_and_observed() -> None:
    segment = demo_segment(make_mocap_track())
    observed = segment.markers.heel.positions()
    modeled = segment.markers.heel.positions(modeled=True)
    assert observed.shape == (3, 3)
    np.testing.assert_allclose(observed[:, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(modeled[:, 0], np.array([0.0, 1.0, 2.0]))


def test_batch_marker_positions_and_as_dict() -> None:
    segment = demo_segment(make_mocap_track())
    stacked = segment.marker_positions([segment.markers.heel, segment.markers.toe])
    assert stacked.shape == (3, 2, 3)
    by_name = segment.marker_positions([segment.markers.heel, segment.markers.toe], as_dict=True)
    assert set(by_name) == {"heel", "toe"}
    np.testing.assert_allclose(by_name["heel"], segment.markers.heel.positions(), equal_nan=True)
    assert segment.marker_positions([]).shape == (3, 0, 3)


def test_marker_velocities_shapes() -> None:
    segment = demo_segment(make_mocap_track())
    velocities = segment.markers.heel.velocities(modeled=True)
    assert velocities.shape == (3, 3)
    np.testing.assert_allclose(velocities[:, 0], 10.0, rtol=1e-5)


def test_missing_observed_marker_is_nan() -> None:
    track = make_mocap_track()
    frames = list(track.marker_frames or ())
    blanked = (frames[0], ViconMarkersFrame(stamp_seconds=frames[1].stamp_seconds, markers=()), frames[2])
    track = MocapTrack(
        subjects=track.subjects,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=blanked,
    )
    observed = demo_segment(track).markers.heel.positions()
    assert np.isnan(observed[1]).all()


def test_unobserved_marker_name_returns_nan() -> None:
    # 'toe'/'mid' are declared but the demo frames only observe 'heel'.
    observed = demo_segment(make_mocap_track()).markers.toe.positions()
    assert np.isnan(observed).all()


def test_observed_marker_requires_marker_frames() -> None:
    track = make_mocap_track(include_markers=False)
    with pytest.raises(ValueError, match="marker_frames"):
        demo_segment(track).markers.heel.positions()


def test_patch_points_and_normals_match_manual_formula() -> None:
    segment = demo_segment(make_mocap_track())
    sole = segment.patches.sole
    rotations = segment.rotations()
    translations = segment.translations()
    # make_mocap_track's first frame is identity rotation + zero translation, so points()[0]
    # and normals()[0] are the segment-local patch origin/normal; every frame must be exactly
    # that rigidly transported by the segment pose.
    local_point = np.asarray(sole.points()[0], dtype=np.float64)
    local_normal = np.asarray(sole.normals()[0], dtype=np.float64)
    expected_points = np.einsum("tij,j->ti", rotations, local_point) + translations
    expected_normals = np.einsum("tij,j->ti", rotations, local_normal)
    np.testing.assert_allclose(sole.points(), expected_points)
    np.testing.assert_allclose(sole.normals(), expected_normals)


def test_declaration_only_patch_points_raise() -> None:
    sole_segment = demo_segment(make_mocap_track())
    with pytest.raises(ValueError, match="no calibrated geometry"):
        sole_segment.patches.toe.points()


def test_multi_patch_points_and_normals() -> None:
    segment = demo_segment(_two_patch_track())
    stacked = segment.patch_points([segment.patches.sole, segment.patches.heel_cap])
    assert stacked.shape == (3, 2, 3)
    np.testing.assert_allclose(stacked[:, 0, :], segment.patches.sole.points())
    by_name = segment.patch_normals([segment.patches.sole, segment.patches.heel_cap], as_dict=True)
    assert set(by_name) == {"sole", "heel_cap"}


def test_empty_slice_query_shapes() -> None:
    segment = demo_segment(make_mocap_track().slice_time(100.0, 101.0))
    assert segment.translations().shape == (0, 3)
    assert segment.markers.heel.positions().shape == (0, 3)
    assert segment.marker_positions([segment.markers.heel, segment.markers.toe]).shape == (0, 2, 3)
    assert segment.patches.sole.points().shape == (0, 3)
    assert segment.patch_points([segment.patches.sole]).shape == (0, 1, 3)


def test_rebase_time_at_construction() -> None:
    track = make_mocap_track()
    rebased = MocapTrack(
        subjects=track.subjects,
        state=track.state,
        timestamps=track.timestamps + 100.0,
        marker_frames=track.marker_frames,
        rebase_time=True,
    )
    np.testing.assert_allclose(rebased.timestamps, track.timestamps)


def test_rebase_time_rebases_attached_contacts() -> None:
    track = make_mocap_track()
    offset_timestamps = track.timestamps + 100.0
    contacts = ContactTrack(
        timestamps=offset_timestamps,
        contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
    )
    rebased = MocapTrack(
        subjects=track.subjects,
        state=track.state,
        timestamps=offset_timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
        rebase_time=True,
    )
    np.testing.assert_allclose(rebased.timestamps, track.timestamps)
    np.testing.assert_allclose(rebased.contacts.timestamps, track.timestamps)


def test_patch_contacts_via_attached_contact_track() -> None:
    track = make_mocap_track()
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={make_demo_patch_target("sole"): np.array([True, False, True])},
    )
    track = MocapTrack(
        subjects=track.subjects,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    segment = demo_segment(track)
    np.testing.assert_array_equal(segment.patches.sole.contacts(), np.array([True, False, True]))
    assert segment.patch_contacts([segment.patches.sole]).shape == (3, 1)


def test_patch_contacts_without_track_raises() -> None:
    segment = demo_segment(make_mocap_track())
    with pytest.raises(ValueError, match="No contact track"):
        segment.patches.sole.contacts()
