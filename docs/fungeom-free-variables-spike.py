"""THROWAWAY SPIKE — prove the 'patch = fungeom data over free-variable markers' model.

It stands in for a *future* fungeom capability (free-variable leaves + `resolve(env)`) with a
~25-line `Deferred` proxy, rewrites the real `_sole` patch (tests/test_patch_geometry_callable.py)
as pure DATA over typed Marker symbols, and proves the resolved Face is identical to today's
callable Face through the real retarget pipeline (face_signal -> frame()/boundary()).

Run: PYTHONPATH=src uv run python <thisfile>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from fungeom import Face, Point3, Point3Bundle, Region2

from retarget.core import Marker
from retarget.core.geometry import SegmentGeometry, face_signal, sampling_at, segment_geometry_at
from retarget.core.targets import SegmentTarget

# --------------------------------------------------------------------------------------------------
# THE SHIM: a stand-in for "fungeom Point3.free() + resolve(env)". A `Deferred` is a lazy fungeom
# value; `__getattr__` records any fluent op (fit_plane/in_frame/offset/facing/flipped/...) and
# replays it on the resolved receiver, so resolve(env) reconstructs the *real* fungeom call graph
# with free Marker leaves substituted from `env`. This is exactly what fungeom would do natively.
# --------------------------------------------------------------------------------------------------


def _resolve(value: Any, env: dict[int, Point3]) -> Any:
    if isinstance(value, Deferred):
        return value.resolve(env)
    if isinstance(value, Marker):  # a free-variable leaf: the marker IS the variable
        return env[id(value)]
    return value


@dataclass(frozen=True)
class Deferred:
    _build: Callable[[dict[int, Point3]], Any]

    def resolve(self, env: dict[int, Point3]) -> Any:
        return self._build(env)

    def __getattr__(self, name: str) -> Callable[..., Deferred]:
        if name.startswith("_"):
            raise AttributeError(name)

        def fluent(*args: Any) -> Deferred:
            return Deferred(lambda env: getattr(self.resolve(env), name)(*[_resolve(a, env) for a in args]))

        return fluent


# fungeom constructors mirrored as Deferred-returning helpers (the classmethod surface):
def bundle(*points: Any) -> Deferred:
    return Deferred(lambda env: Point3Bundle.from_map({str(i): _resolve(p, env) for i, p in enumerate(points)}))


def hull(region_points: Deferred) -> Deferred:
    return Deferred(lambda env: Region2.hull(_resolve(region_points, env)))


def face_on(plane: Deferred, region: Deferred) -> Deferred:
    return Deferred(lambda env: Face.on(_resolve(plane, env), _resolve(region, env)))


# --------------------------------------------------------------------------------------------------
# (A) TODAY — the patch is an imperative callable over stringly-typed marker keys.
# --------------------------------------------------------------------------------------------------
def sole_callable(seg: SegmentGeometry) -> Face:
    markers = seg.markers["heel", "toe", "mid"]
    plane = markers.fit_plane()
    return Face.on(plane, Region2.hull(markers.in_frame(plane)))


# --------------------------------------------------------------------------------------------------
# (B) THE IDEAL — the patch is pure fungeom DATA over typed Marker symbols. No callable, no strings.
# --------------------------------------------------------------------------------------------------
heel = Marker(mocap_name="heel", position_segment=np.array([0.0, 0.0, 0.0]))
toe = Marker(mocap_name="toe", position_segment=np.array([1.0, 0.0, 0.0]))
mid = Marker(mocap_name="mid", position_segment=np.array([0.0, 1.0, 0.0]))

_foot = bundle(heel, toe, mid)
_plane = _foot.fit_plane()
sole_face_data = face_on(_plane, hull(_foot.in_frame(_plane)))  # <- a value you can hold and pass


def _env(*markers: Marker) -> dict[int, Point3]:
    return {id(m): Point3.at(*(float(x) for x in m.position_segment)) for m in markers}


# --------------------------------------------------------------------------------------------------
# PROOF — resolve both to a Face, push each through the REAL pipeline, assert identical.
# --------------------------------------------------------------------------------------------------
def probe(face: Face) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.array([0.0, 1.0])
    translations = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    rotations = np.stack([np.eye(3), Rotation.from_euler("z", 90, degrees=True).as_matrix()])
    signal = face_signal(face, timestamps, translations, rotations)
    sampling = sampling_at(timestamps)
    frames = np.asarray(signal.frame().resolve_over(sampling), dtype=np.float64)
    vertices, _present = signal.boundary().resolve_over(sampling)
    return frames, np.asarray(vertices, dtype=np.float64)


positions = {"heel": np.array([0.0, 0.0, 0.0]), "toe": np.array([1.0, 0.0, 0.0]), "mid": np.array([0.0, 1.0, 0.0])}
old_face = sole_callable(segment_geometry_at(SegmentTarget("body", "seg"), positions))
new_face = sole_face_data.resolve(_env(heel, toe, mid))

old_frames, old_boundary = probe(old_face)
new_frames, new_boundary = probe(new_face)

assert_allclose(old_frames, new_frames, atol=1e-12)
assert_allclose(old_boundary, new_boundary, atol=1e-12)
print("[1] resolved Face is IDENTICAL to the callable Face through the real pipeline:")
print(f"      frame() match: {np.allclose(old_frames, new_frames)}   "
      f"boundary() match: {np.allclose(old_boundary, new_boundary)}")
print(f"      patch origin @frame0 = {old_frames[0, :3, 3]}  (expected centroid ~[0.333,0.333,0])")

# [2] partiality is honest: an unbound marker -> a clear failure naming the missing var.
try:
    sole_face_data.resolve(_env(heel, toe))  # 'mid' missing from the environment
except KeyError:
    print("[2] unbound marker -> resolve fails (fungeom would return Unresolvable naming 'mid').")

# [3] typo-safety: a mis-typed SYMBOL is a NameError (statically a mypy [name-defined] error),
#     vs seg.markers['mdi'] which is a silent string until a bind-time KeyError.
try:
    eval("bundle(heel, toe, mdi)")  # noqa: S307 - 'mdi' is undefined on purpose
except NameError as exc:
    print(f"[3] misspelled symbol -> {type(exc).__name__}: {exc}  (caught statically by mypy)")

print("\nAUTHORING, side by side:")
print("  today:  Patch(geometry=lambda seg: Face.on(seg.markers['heel','toe','mid'].fit_plane(), ...))")
print("  ideal:  Patch(face = face_on(bundle(heel,toe,mid).fit_plane(), hull(...)))   # data; heel/toe/mid typed")
print("\nSPIKE PASSED.")
