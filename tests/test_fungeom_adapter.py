"""Tests for the fungeom backend adapter (retarget.fungeom).

Covers the array-level builders (occlusion -> present mask, the commuting
square, pose-grid fidelity, shape validation), the correspondence/relabel
identity transfer, read-back partiality, and the domain-level builders against
a real bound MocapTrack.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from conftest import (
    SEGMENT,
    SUBJECT,
    make_demo_subjects,
    make_mocap_track,
)

import retarget.fungeom as fg
from fungeom import (
    WORLD_FRAME,
    Point3Bundle,
    Point3BundleSignal,
    Resolvable,
    Resolver,
    TransformBundleSignal,
    Unresolvable,
    UnresolvableError,
)
from fungeom.values import RigidTransform as FgRigidTransform
from retarget.core import (
    Markers,
    Patches,
    RigidTransform,
    SceneState,
    Segment,
    SegmentKey,
    SegmentPoseTrajectory,
    Segments,
    Subject,
    Subjects,
)
from retarget.demo.mocap import MocapTrack
from retarget.io import MarkerObservation, ViconMarkersFrame

# 180 deg about z (a clean antipodal target for I) and 90 deg about z.
ROT90_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ROT180_Z = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def _is_unresolvable(node: Resolver[Any]) -> Unresolvable | None:
    decided = fg.resolvability(node)
    return decided if isinstance(decided, Unresolvable) else None


# --------------------------------------------------------------------------- #
# array-level: point_bundle_signal (marker cloud)
# --------------------------------------------------------------------------- #
def test_point_bundle_signal_occlusion_present_mask() -> None:
    times = np.array([0.0, 1.0, 2.0])
    # marker 'b' occluded (NaN) at frame index 1
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            [[2.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        ]
    )
    sig = fg.point_bundle_signal(times, frames, ["a", "b"])

    # present markers resolve to their exact coordinates
    np.testing.assert_array_equal(fg.marker_at(sig, 0.0, "b"), [10.0, 0.0, 0.0])
    np.testing.assert_array_equal(fg.marker_at(sig, 1.0, "a"), [1.0, 0.0, 0.0])

    # the occluded marker is Unresolvable (absent), reason surfaced
    dropout = _is_unresolvable(sig.at(1.0).at("b"))
    assert dropout is not None
    assert "absent" in dropout.reason

    # present(key) / count() reflect the mask at the dropout instant
    present_b = fg.resolvability(sig.at(1.0).present("b"))
    assert isinstance(present_b, Resolvable) and present_b.value is False
    count = fg.resolvability(sig.at(1.0).count())
    assert isinstance(count, Resolvable) and count.value == 1.0


def test_point_bundle_signal_commuting_square_on_support() -> None:
    times = np.array([0.0, 1.0, 2.0])
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [11.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        ]
    )
    sig = fg.point_bundle_signal(times, frames, ["a", "b"])
    # at(t).at(k) == key(k).at(t) on the support (including an interpolated t)
    for t in (0.0, 1.5, 2.0):
        np.testing.assert_allclose(
            fg.point_array(sig.at(t).at("b")),
            fg.point_array(sig.key("b").at(t)),
        )


def test_point_bundle_signal_all_finite_all_present() -> None:
    times = np.array([0.0, 1.0])
    frames = np.zeros((2, 3, 3))
    sig = fg.point_bundle_signal(times, frames, ["a", "b", "c"])
    for key in ("a", "b", "c"):
        present = fg.resolvability(sig.at(0.0).present(key))
        assert isinstance(present, Resolvable) and present.value is True


def test_point_bundle_signal_explicit_present_marks_finite_absent() -> None:
    times = np.array([0.0, 1.0])
    frames = np.zeros((2, 2, 3))  # all finite
    present = np.array([[True, True], [True, False]])  # mark b absent at frame 1
    sig = fg.point_bundle_signal(times, frames, ["a", "b"], present=present)
    assert _is_unresolvable(sig.at(1.0).at("b")) is not None
    assert _is_unresolvable(sig.at(0.0).at("b")) is None


def test_point_bundle_signal_present_cannot_override_nonfinite() -> None:
    # an explicit present=True at a NaN cell must NOT claim it present
    times = np.array([0.0, 1.0])
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
        ]
    )
    present = np.ones((2, 2), dtype=bool)
    sig = fg.point_bundle_signal(times, frames, ["a", "b"], present=present)
    assert _is_unresolvable(sig.at(1.0).at("b")) is not None


@pytest.mark.parametrize(
    ("frames", "keys"),
    [
        (np.zeros((3, 2, 3)), ["a"]),  # key count mismatch
        (np.zeros((3, 2)), ["a", "b"]),  # wrong ndim
        (np.zeros((3, 2, 4)), ["a", "b"]),  # wrong last dim
    ],
)
def test_point_bundle_signal_shape_validation(frames: np.ndarray, keys: list[str]) -> None:
    with pytest.raises(ValueError):
        fg.point_bundle_signal(np.array([0.0, 1.0, 2.0]), frames, keys)


def test_point_bundle_signal_times_length_mismatch() -> None:
    with pytest.raises(ValueError):
        fg.point_bundle_signal(np.array([0.0, 1.0]), np.zeros((3, 1, 3)), ["a"])


# --------------------------------------------------------------------------- #
# array-level: transform_bundle_signal (pose set)
# --------------------------------------------------------------------------- #
def test_transform_bundle_signal_pose_fidelity() -> None:
    times = np.array([0.0, 1.0])
    rotations = np.broadcast_to(ROT90_Z, (2, 1, 3, 3)).astype(np.float64)
    translations = np.zeros((2, 1, 3))
    translations[:, 0, :] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    sig = fg.transform_bundle_signal(times, translations, rotations, ["joint"])

    matrix = fg.joint_at(sig, 0.0, "joint")
    np.testing.assert_allclose(matrix[:3, :3], ROT90_Z)
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0])
    # fungeom 0.2.0 added the entity-axis key(): pulling one segment's pose-signal out works.
    assert hasattr(sig, "key")


def test_transform_bundle_signal_multi_entity_distinct_rotations() -> None:
    times = np.array([0.0, 1.0])
    rotations = np.zeros((2, 2, 3, 3))
    rotations[:, 0] = np.eye(3)
    rotations[:, 1] = ROT90_Z
    translations = np.zeros((2, 2, 3))
    sig = fg.transform_bundle_signal(times, translations, rotations, ["a", "b"])
    np.testing.assert_allclose(fg.joint_at(sig, 1.0, "a")[:3, :3], np.eye(3))
    np.testing.assert_allclose(fg.joint_at(sig, 1.0, "b")[:3, :3], ROT90_Z)


def test_transform_bundle_signal_nonfinite_pose_absent() -> None:
    times = np.array([0.0, 1.0])
    rotations = np.broadcast_to(np.eye(3), (2, 1, 3, 3)).astype(np.float64).copy()
    translations = np.zeros((2, 1, 3))
    translations[1, 0, :] = [np.nan, 0.0, 0.0]  # pose absent at frame 1
    sig = fg.transform_bundle_signal(times, translations, rotations, ["joint"])
    assert _is_unresolvable(sig.at(1.0).at("joint")) is not None
    assert _is_unresolvable(sig.at(0.0).at("joint")) is None


def test_transform_bundle_signal_shape_validation() -> None:
    times = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        fg.transform_bundle_signal(times, np.zeros((2, 1, 3)), np.zeros((2, 1, 3)), ["j"])
    with pytest.raises(ValueError):
        fg.transform_bundle_signal(times, np.zeros((2, 1, 3)), np.zeros((2, 1, 3, 3)), ["a", "b"])


# --------------------------------------------------------------------------- #
# correspondence + relabel (identity transfer)
# --------------------------------------------------------------------------- #
def _two_marker_bundle() -> Point3Bundle:
    times = np.array([0.0, 1.0])
    frames = np.zeros((2, 2, 3))
    frames[:, 0] = [1.0, 0.0, 0.0]
    frames[:, 1] = [0.0, 2.0, 0.0]
    return fg.point_bundle_signal(times, frames, ["a", "b"]).at(0.0)


def test_roster_map_and_relabel_identity_transfer() -> None:
    bundle = _two_marker_bundle()
    relabeled = fg.relabel(bundle, {"a": "heel", "b": "toe"})
    # values carry across unchanged; only keys are renamed
    np.testing.assert_array_equal(fg.point_array(relabeled.at("heel")), [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(fg.point_array(relabeled.at("toe")), [0.0, 2.0, 0.0])
    support = fg.resolvability(relabeled.support())
    assert isinstance(support, Resolvable)
    assert set(support.value.keys) == {"heel", "toe"}


def test_relabel_drops_out_of_domain_keys() -> None:
    bundle = _two_marker_bundle()
    relabeled = fg.relabel(bundle, fg.roster_map({"a": "heel"}))
    np.testing.assert_array_equal(fg.point_array(relabeled.at("heel")), [1.0, 0.0, 0.0])
    # 'b' was not in the map's domain -> absent from the relabeled bundle
    assert _is_unresolvable(relabeled.at("b")) is not None
    assert _is_unresolvable(relabeled.at("toe")) is not None


def test_relabel_accepts_prebuilt_roster_map() -> None:
    bundle = _two_marker_bundle()
    mapping = fg.roster_map({"a": "x", "b": "y"})
    relabeled = fg.relabel(bundle, mapping)
    np.testing.assert_array_equal(fg.point_array(relabeled.at("x")), [1.0, 0.0, 0.0])


def test_identity_map_over_support() -> None:
    bundle = _two_marker_bundle()
    ident = fg.identity_map(bundle.support())
    source = fg.resolvability(ident.source())
    assert isinstance(source, Resolvable)
    assert set(source.value.keys) == {"a", "b"}


def test_relabel_pose_bundle() -> None:
    times = np.array([0.0, 1.0])
    rotations = np.broadcast_to(np.eye(3), (2, 1, 3, 3)).astype(np.float64)
    translations = np.zeros((2, 1, 3))
    translations[:, 0, 0] = [7.0, 8.0]
    pose = fg.transform_bundle_signal(times, translations, rotations, ["segment"]).at(0.0)
    relabeled = fg.relabel(pose, {"segment": "root"})
    np.testing.assert_allclose(fg.pose_matrix(relabeled.at("root"))[:3, 3], [7.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# read-back partiality
# --------------------------------------------------------------------------- #
def test_marker_at_raises_unresolvable_with_reason() -> None:
    times = np.array([0.0, 1.0])
    frames = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[np.nan, np.nan, np.nan]],
        ]
    )
    sig = fg.point_bundle_signal(times, frames, ["a"])
    with pytest.raises(UnresolvableError) as exc:
        fg.marker_at(sig, 1.0, "a")
    assert "absent" in str(exc.value)


def test_resolvability_is_nonraising() -> None:
    times = np.array([0.0, 1.0])
    frames = np.array([[[0.0, 0.0, 0.0]], [[np.nan, np.nan, np.nan]]])
    sig = fg.point_bundle_signal(times, frames, ["a"])
    decided = fg.resolvability(sig.at(1.0).at("a"))
    assert isinstance(decided, Unresolvable)
    ok = fg.resolvability(sig.at(0.0).at("a"))
    assert isinstance(ok, Resolvable)


# --------------------------------------------------------------------------- #
# domain-level: against a real bound MocapTrack
# --------------------------------------------------------------------------- #
def _partial_occlusion_track(num: int = 3, occlude_at: int = 1) -> MocapTrack:
    poses = tuple(RigidTransform.from_translation(np.array([float(i), 0.0, 0.0])) for i in range(num))
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
    timestamps = np.arange(num, dtype=np.float64) * 0.1
    frames = tuple(
        ViconMarkersFrame(
            stamp_seconds=float(timestamps[i]),
            markers=(
                MarkerObservation(
                    marker_name="heel",
                    subject_name=SUBJECT,
                    segment_name=SEGMENT,
                    position_world=np.array([float(i), 0.0, 0.0]),
                    occluded=(i == occlude_at),
                ),
            ),
        )
        for i in range(num)
    )
    return MocapTrack(
        subjects=make_demo_subjects(),
        state=state,
        timestamps=timestamps,
        marker_frames=frames,
    )


def test_marker_cloud_signal_observed_occlusion() -> None:
    track = _partial_occlusion_track(num=3, occlude_at=1)
    sig = fg.marker_cloud_signal(track, SUBJECT, SEGMENT, markers=["heel"])
    np.testing.assert_array_equal(fg.marker_at(sig, 0.0, "heel"), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(fg.marker_at(sig, 0.2, "heel"), [2.0, 0.0, 0.0])
    # occluded exact frame -> Unresolvable
    assert _is_unresolvable(sig.at(0.1).at("heel")) is not None


def test_marker_cloud_signal_modeled_all_present() -> None:
    track = make_mocap_track(num_timesteps=3, include_markers=False)
    # modeled = R(t) . position_segment + t(t); poses are translations [i,0,0], R=I
    sig = fg.marker_cloud_signal(track, SUBJECT, SEGMENT, modeled=True)
    # default keys are all authored markers on the segment
    support = fg.resolvability(sig.at(0.0).support())
    assert isinstance(support, Resolvable)
    assert set(support.value.keys) == {"heel", "toe", "mid"}
    # heel local [0,0,0] -> world [i,0,0]; toe local [1,0,0] -> [i+1,0,0]; mid -> [i,1,0]
    np.testing.assert_allclose(fg.marker_at(sig, 0.2, "heel"), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(fg.marker_at(sig, 0.2, "toe"), [3.0, 0.0, 0.0])
    np.testing.assert_allclose(fg.marker_at(sig, 0.2, "mid"), [2.0, 1.0, 0.0])


def test_pose_signal_keys_and_values() -> None:
    track = make_mocap_track(num_timesteps=3, include_markers=False)
    sig = fg.pose_signal(track, SUBJECT)
    support = fg.resolvability(sig.at(0.0).support())
    assert isinstance(support, Resolvable)
    assert set(support.value.keys) == {SEGMENT}
    # poses are pure translations [i,0,0] with identity rotation
    matrix = fg.joint_at(sig, 0.2, SEGMENT)
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3))
    np.testing.assert_allclose(matrix[:3, 3], [2.0, 0.0, 0.0])


def test_pose_signal_reads_rotation() -> None:
    poses = (
        RigidTransform.from_rotation_translation(rotation=ROT90_Z, translation=np.array([1.0, 2.0, 3.0])),
        RigidTransform.from_rotation_translation(rotation=ROT90_Z, translation=np.array([1.0, 2.0, 3.0])),
    )
    state = SceneState(segment_poses={SegmentKey(SUBJECT, SEGMENT): SegmentPoseTrajectory(poses=poses)})
    track = MocapTrack(
        subjects=make_demo_subjects(),
        state=state,
        timestamps=np.array([0.0, 0.1]),
        marker_frames=None,
    )
    sig = fg.pose_signal(track, SUBJECT)
    matrix = fg.joint_at(sig, 0.0, SEGMENT)
    np.testing.assert_allclose(matrix[:3, :3], ROT90_Z, atol=1e-12)
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# spec gotchas: SE(3) antipodal blend, interpolated occlusion, temporal gaps
# --------------------------------------------------------------------------- #
def test_transform_bundle_signal_antipodal_blend_surfaces_reason() -> None:
    # the headline pose gotcha: an interpolated instant whose SE(3) blend is
    # opposed (180 deg apart) makes the pose-set Unresolvable; exact frames bypass it
    times = np.array([0.0, 1.0])
    rotations = np.stack([np.eye(3), ROT180_Z])[:, None, :, :]  # (2, 1, 3, 3)
    translations = np.zeros((2, 1, 3))
    sig = fg.transform_bundle_signal(times, translations, rotations, ["joint"])

    # exact frames resolve
    assert _is_unresolvable(sig.at(0.0).at("joint")) is None
    assert _is_unresolvable(sig.at(1.0).at("joint")) is None
    # the interpolated blend is Unresolvable, with fungeom's op-failure reason
    blend = _is_unresolvable(sig.at(0.5).at("joint"))
    assert blend is not None
    assert "opposed orientations" in blend.reason
    # and joint_at surfaces (does not swallow) that reason
    with pytest.raises(UnresolvableError) as exc:
        fg.joint_at(sig, 0.5, "joint")
    assert "opposed orientations" in str(exc.value)


def test_occlusion_propagates_through_interpolation() -> None:
    # marker 'b' occluded at the middle frame; interpolating ACROSS the dropout
    # must leave 'b' absent, while always-present 'a' still resolves there
    times = np.array([0.0, 1.0, 2.0])
    frames = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            [[2.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        ]
    )
    sig = fg.point_bundle_signal(times, frames, ["a", "b"])
    for t in (0.5, 1.5):
        assert _is_unresolvable(sig.at(t).at("b")) is not None
        assert _is_unresolvable(sig.at(t).at("a")) is None


@pytest.mark.parametrize("builder", ["point", "transform"])
def test_max_gap_temporal_dropout(builder: str) -> None:
    times = np.array([0.0, 1.0])  # spaced wider than max_gap
    sig: Point3BundleSignal | TransformBundleSignal
    if builder == "point":
        sig = fg.point_bundle_signal(times, np.zeros((2, 1, 3)), ["a"], max_gap=0.5)
    else:
        sig = fg.transform_bundle_signal(
            times,
            np.zeros((2, 1, 3)),
            np.broadcast_to(np.eye(3), (2, 1, 3, 3)).astype(np.float64),
            ["a"],
            max_gap=0.5,
        )
    # exact samples resolve; an instant strictly inside the temporal gap does not
    assert _is_unresolvable(sig.at(0.0).at("a")) is None
    assert _is_unresolvable(sig.at(1.0).at("a")) is None
    assert _is_unresolvable(sig.at(0.5).at("a")) is not None


# --------------------------------------------------------------------------- #
# present-mask validation + explicit present on the transform builder
# --------------------------------------------------------------------------- #
def test_transform_bundle_signal_explicit_present_marks_finite_absent() -> None:
    times = np.array([0.0, 1.0])
    rotations = np.broadcast_to(np.eye(3), (2, 1, 3, 3)).astype(np.float64)
    translations = np.zeros((2, 1, 3))  # all finite
    present = np.array([[True], [False]])  # mark joint absent at frame 1
    sig = fg.transform_bundle_signal(times, translations, rotations, ["joint"], present=present)
    assert _is_unresolvable(sig.at(1.0).at("joint")) is not None
    assert _is_unresolvable(sig.at(0.0).at("joint")) is None


@pytest.mark.parametrize("builder", ["point", "transform"])
def test_present_shape_mismatch_raises(builder: str) -> None:
    # a (T,) per-frame mask must raise, not silently mis-broadcast against (T, N)
    times = np.array([0.0, 1.0])
    bad_present = np.array([True, False])  # (T,) instead of (T, N)
    with pytest.raises(ValueError, match="present must have shape"):
        if builder == "point":
            fg.point_bundle_signal(times, np.zeros((2, 1, 3)), ["a"], present=bad_present)
        else:
            fg.transform_bundle_signal(
                times,
                np.zeros((2, 1, 3)),
                np.broadcast_to(np.eye(3), (2, 1, 3, 3)).astype(np.float64),
                ["a"],
                present=bad_present,
            )


def test_pose_signal_present_passthrough() -> None:
    track = make_mocap_track(num_timesteps=3, include_markers=False)
    present = np.array([[True], [False], [True]])  # absent at frame 1
    sig = fg.pose_signal(track, SUBJECT, present=present)
    assert _is_unresolvable(sig.at(0.1).at(SEGMENT)) is not None
    assert _is_unresolvable(sig.at(0.0).at(SEGMENT)) is None


# --------------------------------------------------------------------------- #
# non-injective relabel, frame threading, writeable read-back
# --------------------------------------------------------------------------- #
def test_relabel_collapse_is_unresolvable() -> None:
    bundle = _two_marker_bundle()
    relabeled = fg.relabel(bundle, {"a": "j", "b": "j"})  # two sources onto one target
    collapsed = _is_unresolvable(relabeled.at("j"))
    assert collapsed is not None
    assert "collapses" in collapsed.reason
    # the same non-injective map has no inverse
    assert _is_unresolvable(fg.roster_map({"a": "j", "b": "j"}).inverse()) is not None


def test_frame_override_transports_to_world() -> None:
    # a grounded child frame offset by +10x: frame-local [1,0,0] resolves to world [11,0,0]
    rig = WORLD_FRAME.child("rig", to_parent=FgRigidTransform.from_translation(np.array([10.0, 0.0, 0.0])))
    times = np.array([0.0, 1.0])
    frames = np.zeros((2, 1, 3))
    frames[:, 0] = [1.0, 0.0, 0.0]
    sig = fg.point_bundle_signal(times, frames, ["m"], frame=rig)
    np.testing.assert_allclose(fg.marker_at(sig, 0.0, "m"), [11.0, 0.0, 0.0])


def test_readback_arrays_are_owned_and_writeable() -> None:
    times = np.array([0.0, 1.0])
    sig = fg.point_bundle_signal(times, np.zeros((2, 1, 3)), ["m"])
    vec = fg.marker_at(sig, 0.0, "m")
    assert vec.flags.writeable
    vec += 1.0  # must not raise (would, if it aliased fungeom's frozen buffer)

    psig = fg.transform_bundle_signal(
        times,
        np.zeros((2, 1, 3)),
        np.broadcast_to(np.eye(3), (2, 1, 3, 3)).astype(np.float64),
        ["j"],
    )
    mat = fg.joint_at(psig, 0.0, "j")
    assert mat.flags.writeable
    mat += 1.0


# --------------------------------------------------------------------------- #
# multi-segment pose_signal: pins per-segment column assignment
# --------------------------------------------------------------------------- #
class _PMarkers(Markers):
    pass


class _PPatches(Patches):
    pass


class _TwoSegments(Segments):
    a: Segment[_PMarkers, _PPatches]
    b: Segment[_PMarkers, _PPatches]


class _TwoSegSubjects(Subjects):
    body: Subject[_TwoSegments]


def _two_segment_track() -> MocapTrack[_TwoSegSubjects]:
    subjects = _TwoSegSubjects(
        body=Subject(
            segments=_TwoSegments(
                a=Segment(markers=_PMarkers(), patches=_PPatches()),
                b=Segment(markers=_PMarkers(), patches=_PPatches()),
            )
        )
    )
    poses_a = tuple(RigidTransform.from_translation(np.array([1.0, 0.0, 0.0])) for _ in range(2))
    poses_b = tuple(
        RigidTransform.from_rotation_translation(rotation=ROT90_Z, translation=np.array([0.0, 5.0, 0.0]))
        for _ in range(2)
    )
    state = SceneState(
        segment_poses={
            SegmentKey("body", "a"): SegmentPoseTrajectory(poses=poses_a),
            SegmentKey("body", "b"): SegmentPoseTrajectory(poses=poses_b),
        }
    )
    return MocapTrack(
        subjects=subjects,
        state=state,
        timestamps=np.array([0.0, 0.1]),
        marker_frames=None,
    )


def test_pose_signal_multi_segment_column_assignment() -> None:
    track = _two_segment_track()
    sig = fg.pose_signal(track, "body")
    support = fg.resolvability(sig.at(0.0).support())
    assert isinstance(support, Resolvable)
    assert set(support.value.keys) == {"a", "b"}
    np.testing.assert_allclose(fg.joint_at(sig, 0.0, "a")[:3, 3], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(fg.joint_at(sig, 0.0, "a")[:3, :3], np.eye(3))
    np.testing.assert_allclose(fg.joint_at(sig, 0.0, "b")[:3, 3], [0.0, 5.0, 0.0])
    np.testing.assert_allclose(fg.joint_at(sig, 0.0, "b")[:3, :3], ROT90_Z, atol=1e-12)
