"""A SMPL joint track + its energy extractor, and end-to-end temporal sync against a reference."""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import dataclass

from retarget.demo import (
    BodyModel,
    Demonstration,
    EnergySignal,
    SmplTrack,
    SyncEdge,
    SyncPlan,
    Tracks,
    estimate_sync,
    smpl_joint_energy,
)


@dataclass(frozen=True)
class _StubParams:
    """Stand-in for a vendored model's variant-specific params: the joints, given directly."""

    joints: np.ndarray  # (T, J, 3)


class _StubBody:
    """A trivial BodyModel: its 'forward kinematics' just returns the joints in the params.

    Stands in for the vendored ``smpl`` library's SMPL/SMPL-X models so the seam is testable
    without torch or licensed model files.
    """

    joint_names = ("left_ankle", "root")

    def forward_joints(self, params: _StubParams) -> np.ndarray:
        return params.joints

    def forward_transforms(self, params: _StubParams) -> np.ndarray:
        frames, joints, _ = params.joints.shape
        transforms = np.broadcast_to(np.eye(4), (frames, joints, 4, 4)).copy()
        transforms[:, :, :3, 3] = params.joints
        return transforms


def test_from_body_runs_forward_kinematics() -> None:
    joints = np.random.default_rng(0).standard_normal((5, 2, 3))
    track = SmplTrack.from_body(_StubBody(), _StubParams(joints), timestamps=np.arange(5.0))
    assert isinstance(track, SmplTrack)
    assert track.joint_names == ("left_ankle", "root")
    np.testing.assert_array_equal(track.joint_positions("left_ankle"), joints[:, 0, :])
    assert isinstance(_StubBody(), BodyModel)  # the stub satisfies the runtime-checkable Protocol


def _ankle_track(shift: float = 0.0) -> SmplTrack:
    # A distinctive vertical "footstep" signature on the left ankle -- three bumps of different
    # heights at 0.5/1.0/1.4 s (unevenly spaced, so the cross-correlation peak is unambiguous);
    # the root is static. 100 Hz over 2 s.
    t = np.linspace(0.0, 2.0, 201)
    z = sum(h * np.exp(-((t - (c + shift)) ** 2) / 0.002) for c, h in ((0.5, 1.0), (1.0, 0.6), (1.4, 0.9)))
    joints = np.zeros((t.shape[0], 2, 3))
    joints[:, 0, 2] = z
    return SmplTrack(joints=joints, joint_names=("left_ankle", "root"), timestamps=t)


def test_construction_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        SmplTrack(joints=np.zeros((3, 2)), joint_names=("a", "b"), timestamps=np.arange(3.0))
    with pytest.raises(ValueError, match="joint_names"):
        SmplTrack(joints=np.zeros((3, 2, 3)), joint_names=("a",), timestamps=np.arange(3.0))
    with pytest.raises(ValueError, match="unique"):
        SmplTrack(joints=np.zeros((3, 2, 3)), joint_names=("a", "a"), timestamps=np.arange(3.0))
    with pytest.raises(ValueError, match="frames"):
        SmplTrack(joints=np.zeros((3, 1, 3)), joint_names=("a",), timestamps=np.arange(4.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        SmplTrack(joints=np.zeros((3, 1, 3)), joint_names=("a",), timestamps=np.array([0.0, 0.0, 1.0]))


def test_joint_queries() -> None:
    track = _ankle_track()
    assert track.joint_index("root") == 1
    assert track.joint_positions("left_ankle").shape == (201, 3)
    assert track.joint_velocities("left_ankle").shape == (201, 3)
    np.testing.assert_array_equal(track.joint_positions("root"), np.zeros((201, 3)))
    with pytest.raises(KeyError, match="has no joint 'missing'"):
        track.joint_index("missing")


def test_slice_time_returns_smpl_track() -> None:
    clip = _ankle_track().slice_time(0.4, 0.6)
    assert isinstance(clip, SmplTrack)
    assert clip.joints.shape[1:] == (2, 3)
    assert clip.joint_names == ("left_ankle", "root")
    assert np.all((clip.timestamps >= 0.4) & (clip.timestamps < 0.6))


def test_energy_extractor_returns_scalar_signal() -> None:
    energy = smpl_joint_energy("left_ankle", axis=2)(_ankle_track())
    assert isinstance(energy, EnergySignal)
    assert energy.values.ndim == 1 and energy.values.shape == (201,)
    assert energy.values.max() > 0.0  # the footstep moves vertically
    # 3-D speed (axis=None) agrees here since only z moves
    speed = smpl_joint_energy("left_ankle")(_ankle_track())
    np.testing.assert_allclose(speed.values, energy.values, atol=1e-9)


def test_energy_extractor_rejects_foreign_track() -> None:
    class _NotSmpl:
        pass

    with pytest.raises(TypeError, match="expects a SmplTrack"):
        smpl_joint_energy("left_ankle")(_NotSmpl())  # type: ignore[arg-type]


def test_smpl_track_syncs_to_reference_recovering_lag() -> None:
    # The video (source) ankle motion happens 0.07 s LATER than the reference; sync must recover it
    # through the existing energy/timeline machinery, with no sync-side changes for the new track.
    reference = _ankle_track(shift=0.0)
    video = _ankle_track(shift=0.07)

    class _Tracks(Tracks):
        reference: SmplTrack
        video: SmplTrack

    demo = Demonstration(_Tracks(reference=reference, video=video))
    energy = smpl_joint_energy("left_ankle", axis=2)
    plan = SyncPlan(
        reference="reference",
        edges=(
            SyncEdge(
                source="video",
                reference="reference",
                source_signal=energy,
                reference_signal=energy,
                max_lag_seconds=0.25,
            ),
        ),
    )
    (alignment,) = estimate_sync(demo, plan)
    # t_reference = t_source + offset; the source event is 0.07 s late, so offset ~ -0.07.
    assert alignment.transform.offset == pytest.approx(-0.07, abs=0.02)


def test_smpl_drives_a_typed_scene_and_contact_detection_reuses() -> None:
    # The schema-integration payoff: a SMPL bone drives a typed Segment's pose, a sole Patch
    # (geometry= -> Face) rides it, and detect_contacts runs unchanged -- SMPL is "just a subject".
    import smpl
    from fungeom import Face, Point3, Region2

    from retarget.contacts import detect_contacts, ground_plane
    from retarget.core import Marker, Markers, Patch, Patches, Segment, Segments, Subject, Subjects
    from retarget.demo import smpl_mocap_track

    model = smpl.SmplModel(smpl.synthetic_model())
    num_joints = len(model.joint_names)
    num_frames = 12
    transl = np.zeros((num_frames, 3))
    transl[:, 2] = np.linspace(0.1, 0.0, num_frames)  # descend the body to the ground
    params = smpl.SmplParams(betas=np.zeros(2), pose=np.zeros((num_frames, num_joints, 3)), transl=transl)

    def sole(seg: object) -> Face:
        plane = seg.markers["heel", "toe", "mid"].fit_plane().facing(Point3.at(0.0, 0.0, 1.0))  # type: ignore[attr-defined]
        return Face.on(plane, Region2.rectangle(0.2, 0.1))

    class FootMarkers(Markers):
        heel: Marker
        toe: Marker
        mid: Marker

    class FootPatches(Patches):
        sole: Patch

    class FootSegments(Segments):
        foot: Segment[FootMarkers, FootPatches]

    class BodySubjects(Subjects):
        body: Subject[FootSegments]

    subjects = BodySubjects(
        body=Subject(
            segments=FootSegments(
                foot=Segment(
                    mocap_name=model.joint_names[-1],  # drive the foot segment by the last bone
                    markers=FootMarkers(
                        heel=Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0])),
                        toe=Marker(mocap_name="toe", position_segment=np.array([0.2, 0.0, 0.0])),
                        mid=Marker(mocap_name="mid", position_segment=np.array([0.0, 0.1, 0.0])),
                    ),
                    patches=FootPatches(sole=Patch(label="sole", geometry=sole)),
                )
            )
        )
    )

    track = smpl_mocap_track(model, params, subjects, timestamps=np.arange(num_frames) * 0.05)
    foot = track.subjects["body"].segments["foot"]
    # the typed segment pose is exactly the SMPL bone's forward kinematics
    bone = model.forward_transforms(params)[:, -1]  # (T, 4, 4)
    np.testing.assert_allclose(foot.translations(), bone[:, :3, 3], atol=1e-9)
    # the sole patch transports by the SMPL bone pose, and contact detection reuses unchanged
    assert foot.patches["sole"].points().shape == (num_frames, 3)
    contacts = detect_contacts(foot.patches["sole"], against=ground_plane(0.0))
    assert contacts.contacts[foot.patches["sole"].target].shape == (num_frames,)


def test_smpl_mocap_track_rejects_unknown_bone() -> None:
    import smpl

    from retarget.core import Marker, Markers, Patches, Segment, Segments, Subject, Subjects
    from retarget.demo import smpl_mocap_track

    class M(Markers):
        a: Marker

    class P(Patches):
        pass

    class S(Segments):
        seg: Segment[M, P]

    class Subs(Subjects):
        body: Subject[S]

    model = smpl.SmplModel(smpl.synthetic_model())
    subjects = Subs(
        body=Subject(segments=S(seg=Segment(mocap_name="not_a_joint", markers=M(a=Marker("a")), patches=P())))
    )
    params = smpl.SmplParams(betas=np.zeros(2), pose=np.zeros((2, len(model.joint_names), 3)), transl=np.zeros((2, 3)))
    with pytest.raises(KeyError, match="not a joint of the model"):
        smpl_mocap_track(model, params, subjects, timestamps=np.arange(2.0))


def test_smpl_foot_patch_from_mesh_vertices() -> None:
    # The mesh payoff: author a sole Patch from the foot bone's actual mesh vertices (not a
    # bone-frame rectangle). A tiny model -- pelvis(0) + foot(1) at z=0.5, with 4 sole vertices
    # forming a 0.2x0.1 rectangle at z=0 -- stands in for a real SMPL foot region.
    import smpl

    from retarget.contacts import detect_contacts, ground_plane
    from retarget.core import Markers, Patch, Patches, Segment, Segments, Subject, Subjects
    from retarget.demo import smpl_foot_patch, smpl_mocap_track

    v_template = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [-0.1, -0.05, 0.0],
            [0.1, -0.05, 0.0],
            [0.1, 0.05, 0.0],
            [-0.1, 0.05, 0.0],
        ]
    )
    j_regressor = np.zeros((2, 6))
    j_regressor[0, 0] = 1.0
    j_regressor[1, 1] = 1.0
    weights = np.zeros((6, 2))
    weights[0, 0] = 1.0
    weights[1:, 1] = 1.0  # the foot + its sole vertices ride the foot bone
    body = smpl.BodyModelData(
        v_template=v_template,
        shapedirs=np.zeros((6, 3, 1)),
        j_regressor=j_regressor,
        parents=np.array([-1, 0]),
        joint_names=("pelvis", "foot"),
        posedirs=np.zeros((6, 3, 9)),
        weights=weights,
    )
    model = smpl.SmplModel(body)
    rest = smpl.SmplParams(betas=np.zeros(1), pose=np.zeros((1, 2, 3)), transl=np.zeros((1, 3)))

    patch = smpl_foot_patch(model, rest, bone="foot", vertex_indices=[2, 3, 4, 5])
    assert patch.geometry is not None

    num_frames = 10
    transl = np.zeros((num_frames, 3))
    transl[:, 2] = np.linspace(0.1, 0.0, num_frames)  # descend to the ground
    params = smpl.SmplParams(betas=np.zeros(1), pose=np.zeros((num_frames, 2, 3)), transl=transl)

    class FootMarkers(Markers):
        pass

    class FootPatches(Patches):
        sole: Patch

    class FootSegments(Segments):
        foot: Segment[FootMarkers, FootPatches]

    class BodySubjects(Subjects):
        body: Subject[FootSegments]

    subjects = BodySubjects(
        body=Subject(
            segments=FootSegments(
                foot=Segment(mocap_name="foot", markers=FootMarkers(), patches=FootPatches(sole=patch))
            )
        )
    )
    track = smpl_mocap_track(model, params, subjects, timestamps=np.arange(num_frames) * 0.05)
    sole = track.subjects["body"].segments["foot"].patches["sole"]
    # the bound patch's footprint at frame 0 is the world sole rectangle (z = the descending transl)
    boundary = np.asarray(sole.boundary_points()[0])  # (K, 3)
    np.testing.assert_allclose(boundary[:, 0].min(), -0.1, atol=1e-9)
    np.testing.assert_allclose(boundary[:, 0].max(), 0.1, atol=1e-9)
    np.testing.assert_allclose(boundary[:, 1].min(), -0.05, atol=1e-9)
    np.testing.assert_allclose(boundary[:, 1].max(), 0.05, atol=1e-9)
    np.testing.assert_allclose(boundary[:, 2], 0.1, atol=1e-9)
    # contact detection reuses on the real foot surface
    contacts = detect_contacts(sole, against=ground_plane(0.0))
    assert contacts.contacts[sole.target].shape == (num_frames,)
