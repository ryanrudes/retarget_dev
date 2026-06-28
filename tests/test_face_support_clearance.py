"""Bounded contact clearance against a Face-backed patch support.

A patch support (e.g. a board deck) is evaluated as a fungeom ``FaceSignal``: clearance respects
the support's footprint edge (a query off the edge reads as a gap, not contact), unlike the
infinite-plane :class:`~retarget.contacts.supports.TimeIndexedSupport`. Inside the footprint the
two agree exactly. Occluded footprint samples drop out (partiality).
"""

from __future__ import annotations

import numpy as np
import pytest

from fungeom import Direction3, Face, Plane, Point3, Region2

from conftest import demo_segment, make_mocap_track
from retarget.contacts.detect import _FaceSupport, _evaluate_support, _moving_support_from, detect_contacts
from retarget.contacts.supports import TimeIndexedSupport
from retarget.core.targets import PatchTarget

TARGET = PatchTarget(subject="board", segment="deck", patch="surface")


def _square_face() -> Face:
    """A 2x2 square contact face on the z=0 plane, +z outward normal."""
    plane = Plane.through(Point3.at(0.0, 0.0, 0.0), Direction3.of(0.0, 0.0, 1.0))
    return Face.on(plane, Region2.rectangle(2.0, 2.0))


def _identity_support(num: int, *, translations: np.ndarray | None = None) -> _FaceSupport:
    eye = np.broadcast_to(np.eye(3), (num, 3, 3)).copy()
    trans = np.zeros((num, 3)) if translations is None else np.asarray(translations, dtype=np.float64)
    return _FaceSupport(
        face=_square_face(),
        timestamps=np.arange(num, dtype=np.float64),
        translations=trans,
        rotations=eye,
        origins=trans,
        normals=np.tile([0.0, 0.0, 1.0], (num, 1)),
        frames=eye,
        name="surface",
    )


def test_inside_footprint_matches_perpendicular() -> None:
    # Inside the footprint, bounded clearance == the signed perpendicular distance: a gap above
    # is positive, penetration below is negative -- exactly the infinite-plane behavior.
    support = _identity_support(3)
    above = np.tile([0.0, 0.0, 0.05], (3, 1))[:, None, :]
    below = np.tile([0.0, 0.0, -0.03], (3, 1))[:, None, :]
    clr_above, _ = _evaluate_support(support, above, above[:, 0], TARGET)
    clr_below, _ = _evaluate_support(support, below, below[:, 0], TARGET)
    np.testing.assert_allclose(clr_above, 0.05, atol=1e-6)
    np.testing.assert_allclose(clr_below, -0.03, atol=1e-6)


def test_off_edge_sample_reads_as_gap_not_contact() -> None:
    # Two samples: one inside at height 0.2, one OFF the edge (x=5) at plane height. The off-edge
    # sample is ~4 m of lateral gap, so the closest *bounded* approach is the inside 0.2 -- whereas
    # an infinite plane would call the off-edge sample contact (clearance 0).
    support = _identity_support(2)
    samples = np.zeros((2, 2, 3))
    samples[:, 0, :] = [0.0, 0.0, 0.2]
    samples[:, 1, :] = [5.0, 0.0, 0.0]
    clr, _ = _evaluate_support(support, samples, samples[:, 0], TARGET)
    np.testing.assert_allclose(clr, 0.2, atol=1e-3)


def test_bounded_differs_from_infinite_plane_off_edge() -> None:
    # The same off-edge sample: the infinite-plane support reads it as contact (0.0), the bounded
    # Face support reads it as a gap. This is the behavior the bounded clearance changes.
    samples = np.zeros((2, 1, 3))
    samples[:, 0, :] = [5.0, 0.0, 0.0]  # at plane height but far off the 2x2 footprint
    plane_support = TimeIndexedSupport(origins=np.zeros((2, 3)), normals=np.tile([0.0, 0.0, 1.0], (2, 1)))
    clr_plane, _ = _evaluate_support(plane_support, samples, samples[:, 0], TARGET)
    clr_face, _ = _evaluate_support(_identity_support(2), samples, samples[:, 0], TARGET)
    np.testing.assert_allclose(clr_plane, 0.0, atol=1e-9)  # infinite plane: contact
    assert np.all(clr_face > 3.9)  # bounded: ~4 m lateral gap


def test_off_edge_footprint_follows_support_pose() -> None:
    # The support Face moves with its pose: at frame 1 the support has translated to x=10, so a
    # query at (10, 0, 0.1) is inside it (clearance 0.1) while one left behind at the origin is a
    # large gap. This exercises the FaceSignal transport in the support.
    support = _identity_support(2, translations=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    follows = np.array([[[0.0, 0.0, 0.1]], [[10.0, 0.0, 0.1]]])
    stays = np.array([[[0.0, 0.0, 0.1]], [[0.0, 0.0, 0.1]]])
    clr_follows, _ = _evaluate_support(support, follows, follows[:, 0], TARGET)
    clr_stays, _ = _evaluate_support(support, stays, stays[:, 0], TARGET)
    np.testing.assert_allclose(clr_follows, [0.1, 0.1], atol=1e-5)
    assert clr_stays[0] == pytest.approx(0.1, abs=1e-5)
    assert clr_stays[1] > 5.0  # left behind the moved support


def test_respects_in_plane_rotation_of_support() -> None:
    # A support spun about its own normal must rotate its footprint, not just re-centre it. The
    # 2x2 square spun 45 deg becomes a diamond reaching sqrt(2)~1.414 on the x-axis, so a query at
    # x=1.2 (plane height) is INSIDE it -> ~0 clearance; x=1.5 is just past the rotated vertex -> a
    # small gap. (Against the un-rotated square x=1.2 would be a ~0.2 lateral gap -- the fungeom
    # 0.2.2 transformed_by rotation fix is what makes this correct.)
    c = s = np.cos(np.pi / 4)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])[None]
    support = _FaceSupport(
        face=_square_face(),
        timestamps=np.array([0.0]),
        translations=np.zeros((1, 3)),
        rotations=rotation,
        origins=np.zeros((1, 3)),
        normals=np.array([[0.0, 0.0, 1.0]]),
        frames=rotation,
        name="surface",
    )
    inside = np.array([[[1.2, 0.0, 0.0]]])
    outside = np.array([[[1.5, 0.0, 0.0]]])
    clr_inside, _ = _evaluate_support(support, inside, inside[:, 0], TARGET)
    clr_outside, _ = _evaluate_support(support, outside, outside[:, 0], TARGET)
    np.testing.assert_allclose(clr_inside, 0.0, atol=1e-6)
    assert clr_outside[0] == pytest.approx(1.5 - np.sqrt(2.0), abs=1e-3)  # distance to the rotated vertex


def test_occluded_sample_drops_out() -> None:
    # A footprint sample that is NaN (occluded) at a frame cannot be the closest approach; the
    # remaining samples decide. A frame where *every* sample is occluded is NaN (honest gap).
    support = _identity_support(3)
    samples = np.zeros((3, 2, 3))
    samples[:, 0, :] = [0.0, 0.0, 0.1]
    samples[:, 1, :] = [0.0, 0.0, 0.05]
    samples[0, 1, :] = np.nan  # frame 0: sample 1 occluded -> falls back to sample 0 (0.1)
    samples[1, :, :] = np.nan  # frame 1: both occluded -> NaN
    clr, _ = _evaluate_support(support, samples, np.zeros((3, 3)), TARGET)
    assert clr[0] == pytest.approx(0.1, abs=1e-6)
    assert np.isnan(clr[1])
    assert clr[2] == pytest.approx(0.05, abs=1e-6)


def test_patch_support_resolves_to_face_support_and_detects() -> None:
    # Integration: a bound Patch passed as the support routes through _moving_support_from to a
    # _FaceSupport (not the infinite-plane TimeIndexedSupport), and a full detect runs end-to-end.
    track = make_mocap_track()
    sole = demo_segment(track).patches["sole"]
    support = _moving_support_from(sole, up_axis=2)
    assert isinstance(support, _FaceSupport)
    assert support.name == "sole"
    contacts = detect_contacts(sole, against=sole)
    assert sole.target in contacts.contacts
    assert contacts.contacts[sole.target].shape == (len(track.timestamps),)
