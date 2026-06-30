"""Tests for the contact-detection driver (single-support boolean ContactTrack)."""

from __future__ import annotations

from typing import Any

import numpy as np

from retarget.contacts import (
    CONTACT_CHANNELS,
    ContactDetector,
    ContactDetectionConfig,
    classify_feature_channels,
    detect_contacts,
    ground_plane,
)
from retarget.contacts.detect import (
    _PatchData,
    _detect_patch,
    _support_relative_angular_speed,
    _support_relative_motion,
)
from retarget.contacts.supports import TimeIndexedSupport
from retarget.contacts._quiet import detect_quiet, local_polynomial_derivative
from retarget.core.types import FloatArray
from retarget.demo.mocap import MocapTrack

from conftest import (
    make_demo_patch_target,
    make_descending_mocap_track,
    make_gliding_mocap_track,
    demo_segment,
)


def _sole(track: MocapTrack[Any]) -> Any:
    return demo_segment(track).patches.sole


def _gliding_support(track: MocapTrack[Any], glide_speed: float) -> TimeIndexedSupport:
    """A z=0 plane support that translates in x at ``glide_speed`` (rides with the patch)."""
    t = np.asarray(track.timestamps, dtype=np.float64)
    n = len(t)
    return TimeIndexedSupport(
        origins=np.column_stack([glide_speed * t, np.zeros(n), np.zeros(n)]),
        normals=np.tile([0.0, 0.0, 1.0], (n, 1)),
    )


def test_detect_contacts_descent_onto_known_plane() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    contacts = detect_contacts(_sole(track), against=ground_plane(0.0))

    target = make_demo_patch_target("sole")
    mask = contacts.state(target)
    confidence = contacts.confidence(target)

    assert mask.dtype == np.bool_
    assert confidence.shape == t.shape
    assert float(confidence.min()) >= 0.0 and float(confidence.max()) <= 1.0

    # Resting on the floor -> in contact; descending in the air -> not.
    assert mask[(t >= 1.2) & (t <= 2.8)].all()
    assert not mask[np.argmin(np.abs(t - 0.3))]


def test_detect_contacts_infers_floor_when_against_none() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    contacts = detect_contacts(_sole(track))  # default infer_ground() -> fit the floor
    mask = contacts.state(make_demo_patch_target("sole"))
    assert mask[(t >= 1.2) & (t <= 2.8)].all()


def test_detect_contacts_segment_scope_skips_geometryless_patches() -> None:
    track = make_descending_mocap_track()
    segment = demo_segment(track)
    contacts = detect_contacts(segment, against=ground_plane(0.0))
    # The demo segment declares sole (calibrated) and toe (declaration-only).
    assert make_demo_patch_target("sole") in contacts.contacts
    assert make_demo_patch_target("toe") not in contacts.contacts


def test_detect_contacts_against_moving_support() -> None:
    track = make_descending_mocap_track()
    t = track.timestamps
    # A moving floor that sits at z=0 the whole time is equivalent to the ground plane.
    n = len(t)
    support = TimeIndexedSupport(
        origins=np.zeros((n, 3)), normals=np.tile([0.0, 0.0, 1.0], (n, 1))
    )
    contacts = detect_contacts(_sole(track), against=support)
    mask = contacts.state(make_demo_patch_target("sole"))
    assert mask[(t >= 1.2) & (t <= 2.8)].all()


def test_moving_support_detects_co_moving_patch_as_contact() -> None:
    # A shoe-like patch gliding at 0.3 m/s while resting on a support that glides
    # with it: world motion is large, but relative motion is ~0. Measured relative
    # to the moving support it must read as in contact throughout -- world-frame
    # motion would wrongly score the glide as airborne (the skateboard-glide bug).
    glide_speed = 0.3
    track = make_gliding_mocap_track(glide_speed=glide_speed)
    t = track.timestamps
    support = _gliding_support(track, glide_speed)
    contacts = detect_contacts(_sole(track), against=support)
    mask = contacts.state(make_demo_patch_target("sole"))
    assert mask[(t >= 0.3) & (t <= 1.7)].all()
    assert float(mask.mean()) > 0.9


def test_moving_support_rejects_liftoff_during_glide() -> None:
    # Same gliding support, but the patch lifts off (z ramps up) after t=1.0s.
    # Subtracting the support's translation leaves a genuine normal separation, so
    # the airborne tail must NOT read as contact -- the fix is relative, not always-on.
    glide_speed = 0.3
    track = make_gliding_mocap_track(glide_speed=glide_speed, liftoff_at=1.0)
    t = track.timestamps
    support = _gliding_support(track, glide_speed)
    contacts = detect_contacts(_sole(track), against=support)
    mask = contacts.state(make_demo_patch_target("sole"))
    assert mask[(t >= 0.2) & (t <= 0.8)].all()  # resting on the glider -> contact
    assert not mask[t >= 1.5].any()             # lifted clear of it -> not contact


def test_support_relative_motion_subtracts_support_translation() -> None:
    # Unit-level: the helper yields ~zero relative velocity for a co-moving patch
    # (while world velocity is large), and reuses world motion for static supports.
    n = 200
    t = np.linspace(0.0, 2.0, n)
    glide = 0.3
    origins = np.column_stack([glide * t, np.zeros(n), np.zeros(n)])
    points = origins + np.array([0.1, 0.1, 0.0])  # a fixed offset, riding the glider
    support = TimeIndexedSupport(origins=origins, normals=np.tile([0.0, 0.0, 1.0], (n, 1)))
    resolved = ContactDetectionConfig().resolve()
    world_vel = local_polynomial_derivative(
        points, t, resolved.quiet.derivative_window_time, resolved.quiet.derivative_poly_degree
    )
    data = _PatchData(
        target=make_demo_patch_target("sole"),
        points=points,
        sample_points=points[:, None, :],
        velocities=world_vel,
        frames=np.tile(np.eye(3), (n, 1, 1)),
        quiet=detect_quiet(points, t, resolved.quiet),
        coverage=np.ones(n),
        timestamps=t,
    )

    rel_vel, rel_quiet = _support_relative_motion(data, support, resolved)
    assert float(np.linalg.norm(world_vel, axis=1).max()) > 0.25  # world motion is large
    assert float(np.linalg.norm(rel_vel, axis=1).max()) < 1e-6    # relative motion ~ 0
    assert rel_quiet.mask.all()                                   # at rest vs the support

    # A static support has nothing to subtract -> world motion reused unchanged.
    static_vel, static_quiet = _support_relative_motion(data, ground_plane(0.0), resolved)
    assert static_vel is data.velocities
    assert static_quiet is data.quiet


def _spin_about_z(angle: FloatArray) -> np.ndarray:
    """``(T, 3, 3)`` rotations of ``angle`` (rad) about the world z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    z = np.zeros_like(angle)
    o = np.ones_like(angle)
    return np.stack(
        [np.stack([c, -s, z], axis=1), np.stack([s, c, z], axis=1), np.stack([z, z, o], axis=1)],
        axis=1,
    )


def _spinning_patch_data(n: int, spin_rate: float) -> tuple[_PatchData, np.ndarray]:
    """A patch fixed at the origin but spinning about z at ``spin_rate`` rad/s.

    Returns the patch data and the matching ``(T, 3, 3)`` support frames that co-spin
    with it, so a support carrying those frames sees zero relative spin.
    """
    t = np.linspace(0.0, 2.0, n)
    frames = _spin_about_z(spin_rate * t)
    points = np.zeros((n, 3))
    resolved = ContactDetectionConfig().resolve()
    data = _PatchData(
        target=make_demo_patch_target("sole"),
        points=points,
        sample_points=points[:, None, :],
        velocities=np.zeros((n, 3)),
        frames=frames,
        quiet=detect_quiet(points, t, resolved.quiet),
        coverage=np.ones(n),
        timestamps=t,
    )
    return data, frames


def test_support_relative_angular_speed_subtracts_support_spin() -> None:
    # Unit-level: a patch spinning with the support reads ~zero relative angular speed
    # (while its world spin is large), and a support without orientation reuses the
    # world-frame angular speed exactly (the omega_support = 0 limit -- no second path).
    spin = 2.0
    data, support_frames = _spinning_patch_data(200, spin)
    n = len(data.timestamps)
    resolved = ContactDetectionConfig().resolve()
    co_spinning = TimeIndexedSupport(
        origins=np.zeros((n, 3)), normals=np.tile([0.0, 0.0, 1.0], (n, 1)), frames=support_frames
    )

    world = _support_relative_angular_speed(data, ground_plane(0.0), resolved)
    relative = _support_relative_angular_speed(data, co_spinning, resolved)
    assert float(np.median(world)) > 1.5            # large world-frame spin
    assert float(np.max(relative)) < 1e-6           # ~0 relative to the co-spinning support

    # A bare moving plane (frames=None) is the omega_support = 0 case: identical to world.
    bare = TimeIndexedSupport(origins=np.zeros((n, 3)), normals=np.tile([0.0, 0.0, 1.0], (n, 1)))
    np.testing.assert_allclose(_support_relative_angular_speed(data, bare, resolved), world)


def test_angular_speed_smoothing_suppresses_orientation_noise() -> None:
    # A clean constant-rate spin with small per-frame orientation jitter. The raw
    # finite difference rectifies the jitter into a large upward-biased speed; smoothing
    # the rotation-vector path (like the translation channel) recovers the true rate far
    # more accurately, while leaving a noise-free signal undistorted.
    from scipy.spatial.transform import Rotation

    from retarget.contacts._features import angular_speed_from_frames

    n = 400
    t = np.linspace(0.0, 2.0, n)
    rate = 1.0
    clean = _spin_about_z(rate * t)
    rng = np.random.default_rng(0)
    jitter = Rotation.from_rotvec(np.deg2rad(0.5) * rng.standard_normal((n, 3)))
    noisy = (Rotation.from_matrix(clean) * jitter).as_matrix()

    raw = angular_speed_from_frames(noisy, t)                          # finite difference
    smooth = angular_speed_from_frames(noisy, t, window_time=0.1, degree=2)
    raw_err = float(np.mean(np.abs(raw - rate)))
    smooth_err = float(np.mean(np.abs(smooth - rate)))
    assert smooth_err < 0.5 * raw_err                                 # noise markedly suppressed

    # On a noise-free constant spin, smoothing returns the exact true rate (no distortion).
    clean_smooth = angular_speed_from_frames(clean, t, window_time=0.1, degree=2)
    np.testing.assert_allclose(clean_smooth, rate, atol=1e-6)


def test_rigid_contact_on_rotating_support_stays_in_contact() -> None:
    # A patch resting at zero clearance on a support it co-rotates with (a shoe on a
    # balance board lifted and rotated in space). World-frame angular speed is large,
    # but relative to the support it is ~0, so the rigid contact must read as unbroken.
    # Without the support's orientation the same spin would collapse the score -> the
    # fix is relative, not always-on.
    spin = 2.0
    data, support_frames = _spinning_patch_data(200, spin)
    n = len(data.timestamps)
    resolved = ContactDetectionConfig().resolve()
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))

    oriented = TimeIndexedSupport(origins=np.zeros((n, 3)), normals=normals, frames=support_frames)
    mask, _ = _detect_patch(data, oriented, resolved)
    assert mask.all()

    # Drop the orientation: the inherited world spin now reads as motion -> not contact.
    bare = TimeIndexedSupport(origins=np.zeros((n, 3)), normals=normals)
    bare_mask, _ = _detect_patch(data, bare, resolved)
    assert not bare_mask.any()


def test_min_contact_time_removes_short_rest() -> None:
    track = make_descending_mocap_track()
    # The rest lasts ~2.2s; require a longer minimum so it is dropped entirely.
    contacts = detect_contacts(
        _sole(track),
        against=ground_plane(0.0),
        config=ContactDetectionConfig(min_contact_time=5.0),
    )
    assert not contacts.state(make_demo_patch_target("sole")).any()


def test_detector_reuse_matches_function() -> None:
    track = make_descending_mocap_track()
    config = ContactDetectionConfig(contact_confidence=0.9)
    detector = ContactDetector(config)
    target = make_demo_patch_target("sole")

    via_detector = detector.detect(_sole(track), against=ground_plane(0.0))
    via_function = detect_contacts(_sole(track), against=ground_plane(0.0), config=config)
    np.testing.assert_array_equal(via_detector.state(target), via_function.state(target))


def test_classify_feature_channels_explains_the_same_decision() -> None:
    # The diagnostic surfaces the per-channel signals behind detection. It must use the
    # identical pipeline -- its confidence equals what classify() scores -- and the
    # clearance channel must carry the verdict for a shoe descending onto a known floor.
    track = make_descending_mocap_track()
    t = np.asarray(track.timestamps)
    against = {"floor": ground_plane(0.0)}

    channels = classify_feature_channels(_sole(track), against=against)
    target = make_demo_patch_target("sole")
    floor = channels[target]["floor"]

    assert tuple(floor.raw.keys()) == CONTACT_CHANNELS
    assert tuple(floor.residual.keys()) == CONTACT_CHANNELS
    assert floor.confidence.shape == t.shape

    # Same pipeline as classify(): the diagnostic explains exactly that decision.
    scores = ContactDetector().classify(_sole(track), against=against).scores[target]["floor"]
    np.testing.assert_allclose(floor.confidence, scores)

    # Clearance is the story: many σ off while airborne, ~0 once resting on the floor.
    airborne = int(np.argmin(np.abs(t - 0.3)))
    resting = (t >= 1.4) & (t <= 2.6)
    clearance = np.abs(np.asarray(floor.residual["clearance"]))
    assert clearance[airborne] > 3.0
    assert float(np.median(clearance[resting])) < 1.0
