from __future__ import annotations

import numpy as np
import pytest

from retarget.core import (
    MarkerId,
    MarkerSetSpec,
    PatchCalibrationSpec,
    PatchId,
    RectangularRegion,
    RotationFormat,
    RigidTransform,
    SceneState,
    SegmentKey,
    SegmentPoseTrajectory,
    SegmentSpec,
    Z_UP_AXES,
)
from retarget.core.targets import PatchTarget
from retarget.core import PatchHandle
from conftest import (
    DEMO_SEGMENT_SPEC,
    DemoMarkerId,
    DemoPatchId,
    DemoSceneSpec,
    DemoSegmentId,
    DemoSubjectId,
    DemoSubjectSpec,
    make_mocap_track,
)

from retarget.demo.contact import ContactTrack
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame


class TwoPatchId(PatchId):
    SOLE = "sole"
    HEEL_CAP = "heel_cap"


TWO_PATCH_SEGMENT_SPEC = (
    SegmentSpec(
        segment=DemoSegmentId.SEGMENT,
        marker_type=DemoMarkerId,
        patch_type=TwoPatchId,
        axis_convention=Z_UP_AXES,
        marker_set=MarkerSetSpec(marker_type=DemoMarkerId),
        marker_positions_segment={
            DemoMarkerId.HEEL: np.array([0.0, 0.0, 0.0]),
            DemoMarkerId.TOE: np.array([1.0, 0.0, 0.0]),
            DemoMarkerId.MID: np.array([0.0, 1.0, 0.0]),
        },
        patch_calibrations={
            TwoPatchId.SOLE: PatchCalibrationSpec(
                patch=TwoPatchId.SOLE,
                markers=(DemoMarkerId.HEEL, DemoMarkerId.TOE, DemoMarkerId.MID),
                region=RectangularRegion(width=1.0, height=1.0),
            ),
            TwoPatchId.HEEL_CAP: PatchCalibrationSpec(
                patch=TwoPatchId.HEEL_CAP,
                markers=(DemoMarkerId.HEEL, DemoMarkerId.TOE, DemoMarkerId.MID),
                region=RectangularRegion(width=0.5, height=0.5),
                normal_offset=0.05,
            ),
        },
    ).with_built_patches()
)


def make_two_patch_mocap_track() -> MocapTrack:
    scene_spec = DemoSceneSpec(
        subject_spec=DemoSubjectSpec(
            subject=DemoSubjectId.SUBJECT,
            segment_spec=TWO_PATCH_SEGMENT_SPEC,
        )
    )
    poses = tuple(
        RigidTransform.from_translation(np.array([float(i), 0.0, 0.0]))
        for i in range(3)
    )
    state = SceneState(
        segment_poses={
            SegmentKey(DemoSubjectId.SUBJECT, DemoSegmentId.SEGMENT): SegmentPoseTrajectory(
                poses=poses,
            )
        }
    )
    timestamps = np.arange(3, dtype=np.float64) * 0.1
    return MocapTrack(
        scene_spec=scene_spec,
        state=state,
        timestamps=timestamps,
    )


def test_mocap_track_validates_timestamp_length() -> None:
    track = make_mocap_track()
    with pytest.raises(ValueError, match="does not match"):
        MocapTrack(
            scene_spec=track.scene_spec,
            state=track.state,
            timestamps=np.array([0.0, 0.1]),
        )


def test_mocap_track_rejects_duplicate_timestamps() -> None:
    track = make_mocap_track()
    with pytest.raises(ValueError, match="strictly increasing"):
        MocapTrack(
            scene_spec=track.scene_spec,
            state=track.state,
            timestamps=np.array([0.0, 0.1, 0.1]),
            marker_frames=track.marker_frames,
        )


def test_mocap_track_slice_time_returns_expected_indices() -> None:
    track = make_mocap_track()
    view = track.slice_time(0.1, 0.25)
    np.testing.assert_allclose(view.timestamps, np.array([0.1, 0.2]))
    assert view.indices == (1, 2)


def test_nearest_index_on_empty_mocap_track_raises() -> None:
    template = make_mocap_track()
    track = MocapTrack(
        scene_spec=template.scene_spec,
        state=SceneState(),
        timestamps=np.array([], dtype=np.float64),
    )
    with pytest.raises(IndexError, match="cannot query nearest_index on an empty MocapTrack"):
        track.nearest_index(0.0)


def test_nearest_index_on_empty_mocap_view_raises() -> None:
    track = make_mocap_track()
    view = track.slice_time(100.0, 101.0)
    with pytest.raises(IndexError, match="cannot query nearest_index on an empty MocapTrackView"):
        view.nearest_index(0.0)


def test_segment_lookup_by_segment_id_and_spec() -> None:
    track = make_mocap_track()
    by_id = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    by_spec = track._subject(DemoSubjectId.SUBJECT)._segment(DEMO_SEGMENT_SPEC)
    assert by_id.segment_view.spec is DEMO_SEGMENT_SPEC
    assert by_spec.segment_view.spec is DEMO_SEGMENT_SPEC


def test_translations_and_rotations_shapes() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    translations = segment.translations()
    rotations = segment.rotations()
    quats = segment.rotations(format=RotationFormat.QUATERNION_XYZW)
    assert translations.shape == (3, 3)
    assert rotations.shape == (3, 3, 3)
    assert quats.shape == (3, 4)


def test_marker_positions_modeled_and_observed() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    observed = segment._marker_positions(DemoMarkerId.HEEL)
    modeled = segment._marker_positions(DemoMarkerId.HEEL, modeled=True)
    assert observed.shape == (3, 3)
    assert modeled.shape == (3, 3)
    np.testing.assert_allclose(observed[:, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(modeled[:, 0], np.array([0.0, 1.0, 2.0]))


def test_single_element_sequence_query_shapes() -> None:
    track = make_mocap_track()
    sole_target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={sole_target: np.array([True, False, True])},
    )
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    assert segment._marker_positions(DemoMarkerId.HEEL).shape == (3, 3)
    assert segment._marker_positions([DemoMarkerId.HEEL]).shape == (3, 1, 3)
    assert segment._marker_positions([]).shape == (3, 0, 3)
    assert segment._marker_velocities(DemoMarkerId.HEEL).shape == (3, 3)
    assert segment._marker_velocities([DemoMarkerId.HEEL]).shape == (3, 1, 3)
    assert segment._patch_points(DemoPatchId.SOLE).shape == (3, 3)
    assert segment._patch_points([DemoPatchId.SOLE]).shape == (3, 1, 3)
    assert segment._patch_normals(DemoPatchId.SOLE).shape == (3, 3)
    assert segment._patch_normals([DemoPatchId.SOLE]).shape == (3, 1, 3)
    assert segment._marker_positions("heel").shape == (3, 3)
    assert segment._marker_velocities("heel").shape == (3, 3)
    assert segment._patch_points("sole").shape == (3, 3)
    assert segment._patch_normals("sole").shape == (3, 3)
    assert segment._patch_velocities("sole").shape == (3, 3)
    assert segment._patch_contacts(DemoPatchId.SOLE).shape == (3,)
    assert segment._patch_contacts([DemoPatchId.SOLE]).shape == (3, 1)
    assert segment._patch_contacts("sole").shape == (3,)


def test_missing_observed_marker_returns_nan_rows() -> None:
    track = make_mocap_track()
    frames = list(track.marker_frames or ())
    missing_frame = ViconMarkersFrame(
        stamp_seconds=frames[1].stamp_seconds,
        markers=(),
    )
    marker_frames = (frames[0], missing_frame, frames[2])
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=marker_frames,
    )
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    observed = segment._marker_positions(DemoMarkerId.HEEL)
    assert np.isnan(observed[1]).all()


def test_multiple_marker_query_and_return_dict() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    stacked = segment._marker_positions([DemoMarkerId.HEEL, DemoMarkerId.TOE])
    by_id = segment._marker_positions(
        [DemoMarkerId.HEEL, DemoMarkerId.TOE],
        return_dict=True,
    )
    assert stacked.shape == (3, 2, 3)
    assert set(by_id) == {DemoMarkerId.HEEL, DemoMarkerId.TOE}
    assert by_id[DemoMarkerId.HEEL].shape == (3, 3)


def test_string_subject_and_segment_queries_resolve_to_typed_ids() -> None:
    track = make_mocap_track()
    subject = track._subject("subject")
    assert subject.subject_id is DemoSubjectId.SUBJECT
    segment = subject._segment("segment")
    assert segment.segment_view.segment_id is DemoSegmentId.SEGMENT
    assert segment._marker_positions("heel").shape == (3, 3)
    assert segment._patch_points("sole").shape == (3, 3)


def test_typed_mocap_accessors_delegate_to_dynamic_queries() -> None:
    track = make_mocap_track()

    left_shoe = track.subjects["subject"]
    assert left_shoe == track._subject("subject")

    shoe = left_shoe.segments["segment"]
    assert shoe == left_shoe._segment("segment")

    heel = shoe.markers["heel"]
    sole = shoe.patches["sole"]

    np.testing.assert_allclose(heel.positions(), shoe._marker_positions("heel"))
    np.testing.assert_allclose(sole.points(), shoe._patch_points("sole"))


def test_typed_mocap_accessors_work_on_sliced_views() -> None:
    track = make_mocap_track()
    view = track.slice_time(0.1, 0.25)

    left_shoe = view.subjects["subject"]
    assert left_shoe == view._subject("subject")
    shoe = left_shoe.segments["segment"]
    assert shoe == view._subject("subject")._segment("segment")
    heel = shoe.markers["heel"]
    sole = shoe.patches["sole"]

    assert len(heel.positions()) == len(view.timestamps)
    np.testing.assert_allclose(heel.positions(), shoe._marker_positions("heel"))
    np.testing.assert_allclose(sole.points(), shoe._patch_points("sole"))


def test_unknown_marker_string_raises_key_error() -> None:
    segment = make_mocap_track()._subject("subject")._segment("segment")
    with pytest.raises(KeyError, match="has no marker 'missing'"):
        segment._marker_positions("missing")


def test_unknown_patch_string_raises_key_error() -> None:
    segment = make_mocap_track()._subject("subject")._segment("segment")
    with pytest.raises(KeyError, match="has no patch 'missing'"):
        segment._patch_points("missing")


def test_modeled_marker_velocity_shape() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    velocities = segment._marker_velocities(DemoMarkerId.HEEL, modeled=True)
    assert velocities.shape == (3, 3)
    np.testing.assert_allclose(velocities[:, 0], 10.0, rtol=1e-5)


def test_observed_marker_velocity_shape() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    velocities = segment._marker_velocities(DemoMarkerId.HEEL)
    assert velocities.shape == (3, 3)
    np.testing.assert_allclose(velocities[:, 0], 10.0, rtol=1e-5)


def test_multi_marker_velocity_columns_match_single_marker() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    markers = [DemoMarkerId.HEEL, DemoMarkerId.TOE, DemoMarkerId.MID]
    stacked = segment._marker_velocities(markers, modeled=True)
    assert stacked.shape == (3, 3, 3)
    for column, marker in enumerate(markers):
        single = segment._marker_velocities(marker, modeled=True)
        np.testing.assert_allclose(stacked[:, column, :], single)

    observed_stacked = segment._marker_velocities(markers)
    assert observed_stacked.shape == (3, 3, 3)
    for column, marker in enumerate(markers):
        single = segment._marker_velocities(marker)
        np.testing.assert_allclose(
            observed_stacked[:, column, :],
            single,
            equal_nan=True,
        )


def test_one_timestep_marker_velocity_returns_zeros() -> None:
    track = make_mocap_track(num_timesteps=1)
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    modeled = segment._marker_velocities(DemoMarkerId.HEEL, modeled=True)
    observed = segment._marker_velocities(DemoMarkerId.HEEL)
    assert modeled.shape == (1, 3)
    assert observed.shape == (1, 3)
    np.testing.assert_allclose(modeled, 0.0)
    np.testing.assert_allclose(observed, 0.0)

    multi = segment._marker_velocities([DemoMarkerId.HEEL, DemoMarkerId.TOE], modeled=True)
    assert multi.shape == (1, 2, 3)
    np.testing.assert_allclose(multi, 0.0)


def test_empty_slice_returns_correctly_shaped_arrays() -> None:
    track = make_mocap_track()
    segment = track.slice_time(100.0, 101.0)._subject(DemoSubjectId.SUBJECT)._segment(
        DemoSegmentId.SEGMENT
    )
    assert segment.poses() == ()
    assert segment.translations().shape == (0, 3)
    assert segment.rotations().shape == (0, 3, 3)
    assert segment.rotations(format=RotationFormat.QUATERNION_XYZW).shape == (0, 4)
    assert segment._marker_positions(DemoMarkerId.HEEL).shape == (0, 3)
    assert segment._marker_positions([DemoMarkerId.HEEL, DemoMarkerId.TOE]).shape == (0, 2, 3)
    by_id = segment._marker_positions(
        [DemoMarkerId.HEEL, DemoMarkerId.TOE],
        return_dict=True,
    )
    assert by_id[DemoMarkerId.HEEL].shape == (0, 3)
    assert by_id[DemoMarkerId.TOE].shape == (0, 3)
    assert segment._marker_positions(DemoMarkerId.HEEL, modeled=True).shape == (0, 3)
    assert (
        segment._marker_positions([DemoMarkerId.HEEL, DemoMarkerId.TOE], modeled=True).shape
        == (0, 2, 3)
    )
    modeled_by_id = segment._marker_positions(
        [DemoMarkerId.HEEL, DemoMarkerId.TOE],
        modeled=True,
        return_dict=True,
    )
    assert modeled_by_id[DemoMarkerId.HEEL].shape == (0, 3)
    assert modeled_by_id[DemoMarkerId.TOE].shape == (0, 3)
    assert segment._marker_velocities(DemoMarkerId.HEEL).shape == (0, 3)
    assert segment._marker_velocities([DemoMarkerId.HEEL, DemoMarkerId.TOE]).shape == (0, 2, 3)
    velocity_by_id = segment._marker_velocities(
        [DemoMarkerId.HEEL, DemoMarkerId.TOE],
        return_dict=True,
    )
    assert velocity_by_id[DemoMarkerId.HEEL].shape == (0, 3)
    assert velocity_by_id[DemoMarkerId.TOE].shape == (0, 3)
    assert segment._marker_velocities(DemoMarkerId.HEEL, modeled=True).shape == (0, 3)
    assert (
        segment._marker_velocities([DemoMarkerId.HEEL, DemoMarkerId.TOE], modeled=True).shape
        == (0, 2, 3)
    )
    assert segment._patch_points(DemoPatchId.SOLE).shape == (0, 3)
    assert segment._patch_normals(DemoPatchId.SOLE).shape == (0, 3)
    assert segment._patch_points([DemoPatchId.SOLE]).shape == (0, 1, 3)
    assert segment._patch_normals([DemoPatchId.SOLE]).shape == (0, 1, 3)


def test_modeled_multi_marker_positions_shape_and_columns() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    markers = [DemoMarkerId.HEEL, DemoMarkerId.TOE, DemoMarkerId.MID]
    stacked = segment._marker_positions(markers, modeled=True)
    assert stacked.shape == (3, 3, 3)
    for column, marker in enumerate(markers):
        single = segment._marker_positions(marker, modeled=True)
        np.testing.assert_allclose(stacked[:, column, :], single)


def test_modeled_multi_marker_return_dict() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    markers = [DemoMarkerId.HEEL, DemoMarkerId.TOE]
    by_id = segment._marker_positions(markers, modeled=True, return_dict=True)
    assert set(by_id) == set(markers)
    for marker in markers:
        np.testing.assert_allclose(
            by_id[marker],
            segment._marker_positions(marker, modeled=True),
        )


def test_modeled_marker_positions_repeated_calls_are_equal() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    markers = [DemoMarkerId.HEEL, DemoMarkerId.TOE]
    first = segment._marker_positions(markers, modeled=True)
    second = segment._marker_positions(markers, modeled=True)
    np.testing.assert_array_equal(first, second)


def test_observed_multi_marker_positions_shape_and_columns() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    markers = [DemoMarkerId.HEEL, DemoMarkerId.TOE, DemoMarkerId.MID]
    stacked = segment._marker_positions(markers)
    assert stacked.shape == (3, 3, 3)
    for column, marker in enumerate(markers):
        single = segment._marker_positions(marker)
        np.testing.assert_allclose(
            stacked[:, column, :],
            single,
            equal_nan=True,
        )


def test_observed_multi_marker_return_dict() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    markers = [DemoMarkerId.HEEL, DemoMarkerId.TOE]
    by_id = segment._marker_positions(markers, return_dict=True)
    assert set(by_id) == set(markers)
    for marker in markers:
        np.testing.assert_allclose(
            by_id[marker],
            segment._marker_positions(marker),
            equal_nan=True,
        )


def test_occluded_observed_marker_returns_nan() -> None:
    track = make_mocap_track()
    frames = list(track.marker_frames or ())
    occluded_frame = ViconMarkersFrame(
        stamp_seconds=frames[1].stamp_seconds,
        markers=(
            MarkerObservation(
                marker_name=DemoMarkerId.HEEL.label,
                subject_name=DemoSubjectId.SUBJECT.label,
                segment_name=DemoSegmentId.SEGMENT.label,
                position_world=np.array([99.0, 99.0, 99.0]),
                occluded=True,
            ),
        ),
    )
    marker_frames = (frames[0], occluded_frame, frames[2])
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=marker_frames,
    )
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    observed = segment._marker_positions(DemoMarkerId.HEEL)
    assert np.isnan(observed[1]).all()


def test_empty_observed_marker_query_shapes() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    assert segment._marker_positions([]).shape == (3, 0, 3)
    assert segment._marker_positions([], return_dict=True) == {}


def test_patch_points_matches_scalar_oracle() -> None:
    track = make_mocap_track()
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    patch_view = segment.segment_view.patch(DemoPatchId.SOLE)
    expected = np.stack(
        [patch_view.contact_point_world_at(i) for i in range(len(track))],
        axis=0,
    )
    np.testing.assert_allclose(segment._patch_points(DemoPatchId.SOLE), expected)


def test_patch_normals_matches_scalar_oracle() -> None:
    track = make_mocap_track()
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    patch_view = segment.segment_view.patch(DemoPatchId.SOLE)
    expected = np.stack(
        [patch_view.normal_world_at(i) for i in range(len(track))],
        axis=0,
    )
    np.testing.assert_allclose(segment._patch_normals(DemoPatchId.SOLE), expected)


def test_multi_patch_points_shape_and_columns() -> None:
    segment = make_two_patch_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(
        DemoSegmentId.SEGMENT
    )
    patches = [TwoPatchId.SOLE, TwoPatchId.HEEL_CAP]
    stacked = segment._patch_points(patches)
    assert stacked.shape == (3, 2, 3)
    for column, patch in enumerate(patches):
        single = segment._patch_points(patch)
        np.testing.assert_allclose(stacked[:, column, :], single)


def test_multi_patch_normals_shape_and_columns() -> None:
    segment = make_two_patch_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(
        DemoSegmentId.SEGMENT
    )
    patches = [TwoPatchId.SOLE, TwoPatchId.HEEL_CAP]
    stacked = segment._patch_normals(patches)
    assert stacked.shape == (3, 2, 3)
    for column, patch in enumerate(patches):
        single = segment._patch_normals(patch)
        np.testing.assert_allclose(stacked[:, column, :], single)


def test_patch_points_return_dict() -> None:
    segment = make_two_patch_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(
        DemoSegmentId.SEGMENT
    )
    patches = [TwoPatchId.SOLE, TwoPatchId.HEEL_CAP]
    by_id = segment._patch_points(patches, return_dict=True)
    assert set(by_id) == set(patches)
    for patch in patches:
        np.testing.assert_allclose(by_id[patch], segment._patch_points(patch))


def test_patch_normals_return_dict() -> None:
    segment = make_two_patch_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(
        DemoSegmentId.SEGMENT
    )
    patches = [TwoPatchId.SOLE, TwoPatchId.HEEL_CAP]
    by_id = segment._patch_normals(patches, return_dict=True)
    assert set(by_id) == set(patches)
    for patch in patches:
        np.testing.assert_allclose(by_id[patch], segment._patch_normals(patch))


def test_empty_patch_query_shapes() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    assert segment._patch_points([]).shape == (3, 0, 3)
    assert segment._patch_normals([]).shape == (3, 0, 3)
    assert segment._patch_points([], return_dict=True) == {}
    assert segment._patch_normals([], return_dict=True) == {}

    empty_time = make_two_patch_mocap_track().slice_time(100.0, 101.0)._subject(
        DemoSubjectId.SUBJECT
    )._segment(DemoSegmentId.SEGMENT)
    patches = [TwoPatchId.SOLE, TwoPatchId.HEEL_CAP]
    assert empty_time._patch_points(patches).shape == (0, 2, 3)
    assert empty_time._patch_normals(patches).shape == (0, 2, 3)


def test_patch_points_repeated_calls_are_equal() -> None:
    segment = make_mocap_track()._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    first = segment._patch_points(DemoPatchId.SOLE)
    second = segment._patch_points(DemoPatchId.SOLE)
    np.testing.assert_array_equal(first, second)


def test_nan_observed_marker_positions_propagate_to_velocities() -> None:
    track = make_mocap_track()
    frames = list(track.marker_frames or ())
    missing_frame = ViconMarkersFrame(
        stamp_seconds=frames[1].stamp_seconds,
        markers=(),
    )
    marker_frames = (frames[0], missing_frame, frames[2])
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=marker_frames,
    )
    segment = track._subject(DemoSubjectId.SUBJECT)._segment(DemoSegmentId.SEGMENT)
    positions = segment._marker_positions(DemoMarkerId.HEEL)
    velocities = segment._marker_velocities(DemoMarkerId.HEEL)
    assert velocities.shape == (3, 3)
    assert np.isnan(positions[1]).all()
    assert np.isnan(velocities[0]).all()
    assert np.isnan(velocities[2]).all()
    np.testing.assert_allclose(velocities[1], np.array([10.0, 0.0, 0.0]))


def test_mocap_track_with_rebased_time() -> None:
    track = make_mocap_track()
    shifted = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps + 100.0,
        marker_frames=track.marker_frames,
    )
    rebased = shifted.with_rebased_time()
    np.testing.assert_allclose(rebased.timestamps, track.timestamps)
    assert rebased.scene_spec is shifted.scene_spec
    assert rebased.state is shifted.state
    assert rebased.marker_frames is shifted.marker_frames


def test_mocap_track_with_timestamps_rejects_attached_contacts_when_timestamps_change() -> (
    None
):
    track = make_mocap_track()
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={target: np.array([True, False, True])},
    )
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    with pytest.raises(ValueError, match="contacts are attached"):
        track.with_timestamps(track.timestamps + 100.0)


def test_mocap_track_with_timestamps_allows_unchanged_contacts() -> None:
    track = make_mocap_track()
    target = PatchTarget(
        subject=DemoSubjectId.SUBJECT,
        handle=PatchHandle(segment=DemoSegmentId.SEGMENT, patch=DemoPatchId.SOLE),
    )
    contacts = ContactTrack(
        timestamps=track.timestamps,
        contacts={target: np.array([True, False, True])},
    )
    track = MocapTrack(
        scene_spec=track.scene_spec,
        state=track.state,
        timestamps=track.timestamps,
        marker_frames=track.marker_frames,
        contacts=contacts,
    )
    updated = track.with_timestamps(track.timestamps.copy())
    np.testing.assert_allclose(updated.timestamps, track.timestamps)
    assert updated.contacts is contacts
