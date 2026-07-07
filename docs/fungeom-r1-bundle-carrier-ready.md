# fungeom response — R1's vectorized carriers are ready (markers **and** poses)

**From:** the **fungeom** repo (`~/GitHub/functional_api`). **To:** the retarget session.
**Re:** [`fungeom-integration-roadmap.md`](fungeom-integration-roadmap.md) → **R1** ("markers + poses as
fungeom signals; un-orphan the adapter"). Markers in **v0.2.3**, **poses in v0.3.0** (both on PyPI).

> **Update (v0.3.0):** the pose half is now done too — see **"Poses — `pose_signal` is now fast as
> well"** below. The earlier "pose_signal is still per-instant / deferred" note is superseded:
> `TransformBundleSignal` had *no* `resolve_over` at all, and now has both a generic one and a dense
> `from_matrices` carrier. **R1 is fully unblocked on the fungeom side.**

## The R1 ask — delivered, with zero adapter API change

R1 wanted *"a vectorized `Point3BundleSignal` carrier from a dense `(T, N, 3)` array (the analog of
`TransformSignal.from_matrices`)"* because `from_frames` was per-instant.

It turns out `from_frames` **already stored** the dense `(T, N, 3)` array — the cost was that
`resolve_over` materialized `T·N` per-frame `Point3` objects on the way out. So rather than add a new
constructor, I made the **existing `from_frames` carrier fast**: `Point3BundleSignal.resolve_over` (and
`ScalarBundleSignal.resolve_over` — the contact-clearance field) now read the dense array back in one
batched numpy interpolation.

**This means your adapter needs no migration.** `marker_cloud_signal` / `point_bundle_signal` already
call `from_frames` + `resolve_over`; they just got fast. Wire R1 as planned.

## Perf (T=5000, N=50, the realistic marker case)

| readback | before | after |
|---|---:|---:|
| `Point3BundleSignal.resolve_over`, same grid | 66 ms | **~2.7 ms (~25×)** |
| `Point3BundleSignal.resolve_over`, between-sample grid | 803 ms | **~2.6 ms (~300×)** |
| `ScalarBundleSignal.resolve_over` (contact field) | 40 ms | ~2 ms |

(Lesson #1 honored — measured before, measured after.) Comfortably within a small constant of the
~8 ms `TransformSignal` carrier, and grid-independent (the between-sample case is now as fast as the
exact-grid one).

## Exactness — bit-for-bit, including occlusion

The fast path is **provably equivalent** to the per-instant readback at the values you'll use:
exact knots short-circuit (the reconstruction contract), and an interior target is the
**key-intersection** lerp of its two bracketing frames — so coordinates *and* the `(T, N)` present
mask match the old `resolved_grid` to machine precision (≤ 6e-16), occluded cells `nan` + `False`
exactly as before. Anything the shortcut doesn't model — a `hold`/`nearest` kernel, a `max_gap`
(interior gaps), an off-domain target — transparently **falls back** to the generic path (so
off-support still raises `UnresolvableError`, occlusion partiality still flows). 1329 tests, 100%
coverage.

Repro:

```python
import numpy as np
from fungeom import Point3BundleSignal, Sampling
T, N = 5000, 50
times = np.arange(T) * 0.01
cloud = Point3BundleSignal.from_frames(times, np.random.randn(T, N, 3),
                                       present=np.random.rand(T, N) > 0.1)
values, mask = cloud.resolve_over(Sampling.at_times(times))   # (T, N, 3), (T, N)  — ~2.7 ms
```

## Poses — `pose_signal` is now fast as well (v0.3.0)

`TransformBundleSignal` previously had **no `resolve_over` at all** — so `pose_signal` couldn't be
bulk-read to `(T, N, 4, 4)` arrays, and `pose_bundle_signal` was forced to wrap `T·N`
`Transform.known(...)` objects for `from_frames` (slow at *both* construction and readback). Both are
fixed:

- **`TransformBundleSignal.resolve_over(onto) → ((T, N, 4, 4), (T, N) mask)`** now exists (the pose
  companion to the cloud readbacks; occluded joints `nan` / `False`).
- **`TransformBundleSignal.from_matrices(times, (T, N, 4, 4), keys, present)`** is the dense batch
  carrier (analog of `TransformSignal.from_matrices`): it stores the raw `(T, N, 4, 4)` array — **no
  `T·N` `Transform` wrappers** — and its `resolve_over` short-circuits an exact-knot grid (resolving
  onto the signal's own times — your R1 case) to a **bit-exact** matrix copy, else reads back via a
  batched per-joint quaternion slerp + numpy translation lerp.

**Adapter change for `pose_bundle_signal` (one line):** you already have dense `translations (T,N,3)`
+ `rotations (T,N,3,3)` — stack them into `(T, N, 4, 4)` and call `from_matrices` instead of building
the `Transform` grid for `from_frames`. That kills the `T·N`-object construction cost too.

Perf (T=5000, N=50, onto the own grid): `from_frames` per-instant **313 ms → `from_matrices` ~6 ms
(~54×)**, bit-exact. A genuine *between-sample* pose resample is ~107 ms (the scipy quaternion
round-trip over `T·N`) — fine for R4 resampling; if you need that hot, ping me and I'll precompute the
quaternions at construction. Exactness verified to ≤7e-16 vs the per-instant blend (incl. occlusion).

```python
import numpy as np
from fungeom import TransformBundleSignal, Sampling
T, N = 5000, 50
times = np.arange(T) * 0.01
mats = ...  # (T, N, 4, 4) from your translations + rotations
poses = TransformBundleSignal.from_matrices(times, mats, keys=segment_names, present=mask)
stack, present = poses.resolve_over(Sampling.at_times(times))   # (T, N, 4, 4), (T, N)  — ~6 ms
```

## On R4 (so you can plan)

- **R4 (discrete resampling) is already covered:** build the signal with `via=Interpolation.nearest`
  or `Interpolation.hold`; `resample`/`resolve_over` honor it (through the generic path).
- **R4 (contact-bool) caveat:** `BoolSignal` is a separate three-valued hierarchy with **no
  `resample`/`reparameterize`** today. The clean path is to resample/reparameterize the underlying
  *scalar* clearance `ScalarSignal` (which supports both) and re-threshold (`.lt(0)`) — no new fungeom
  API needed. If you'd rather resample the bool directly, that's a small additive ask for the R4 round.

R1 is unblocked. Ping with a repro if `marker_cloud_signal`/`pose_signal` `resolve_over` doesn't hit
these numbers at your real T, N.
