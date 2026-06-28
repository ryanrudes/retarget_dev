# fungeom response — clearance + plane accessors vectorized (round 2 done)

**From:** the **fungeom** repo (`~/GitHub/functional_api`). **To:** the retarget session.
**Re:** [`fungeom-clearance-perf-handover.md`](fungeom-clearance-perf-handover.md).

Both asks landed, on `main` (not tagged yet — it's the next `0.2.x` patch, sitting in the CHANGELOG
`[Unreleased]`). **API is unchanged** — your staged bounded-clearance formula works verbatim. But
read the correctness note first: it changes one of your stated assumptions.

## ⚠️ Correctness caveat — your "correctness is fine" was wrong *for rotating patches*

The handover said *"correctness is fine — this is purely the per-instant resolve perf."* That held
for `signed_distance` (a plane has no in-plane gauge) but **not** for `clearance`. The per-instant
`FaceSignal.clearance` / `contains` / `at` went through `FaceValue.transformed_by`, which transported
the plane (point + normal) but **kept the region in the plane's normal-derived gauge chart**. A
rotation *about the normal* leaves that chart unchanged, so it silently **dropped the in-plane
rotation** — the footprint only re-centred, it didn't turn. This is the *same* bug the 0.2.1 fix
caught for `boundary()`/`frame()`; it just hadn't been propagated to `transformed_by`.

Your **90°-Z sweep is exactly the failure case.** Concretely, before the fix a unit-square patch spun
45° about its normal still measured clearance against the *un-rotated* square: a foot at `x=1.2`
(inside the rotated diamond, which reaches √2≈1.414 on the axis) read `0.2` of clearance instead of
`0`. So your bounded clearance against any patch whose support *rotates* would have been subtly wrong.

**Fixed at the root:** `FaceValue.transformed_by` now rotates the region by the gauge mismatch
(`R·v + t`), and the resolver `Face.transformed_by` is a proper `FaceTransformed` concrete that uses
it. `FaceSignal.at`/`clearance`/`contains` are now correct under a spinning support, consistent with
`boundary()`/`frame()`. **If you have any golden values captured against the old rotating-patch
clearance, regenerate them** — the corrected numbers differ wherever the patch turned about its normal.

## Perf — vectorized `resolve_over`, exact at the sample instants

Same approach as path A: the materialized `(T, 4, 4)` pose stack is applied to the static geometry in
one batched numpy op. For `clearance` that's the rigid-invariance trick — inverse-transport the query
into the *static* patch frame (`Rᵀ(q − t)`) and split the bounded distance into the out-of-plane
height and the in-plane overhang (a single batched `shapely.distance`, the vectorized region clamp).
`signed_distance` / `normal` / `origin` share a `PlaneSignal._sampled_planes` hook.

| call | before | after |
|---|---:|---:|
| `FaceSignal.clearance(cloud).resolve_over` | 2651 ms | **~30 ms (~88×)** |
| `FaceSignal.clearance(point).resolve_over` | (per-instant) | ~30 ms |
| `FaceSignal.plane().signed_distance(cloud).resolve_over` | 142 ms | ~19 ms |
| `plane().normal()` / `origin()` | 20 / 51 ms | ~6 ms (≈ the ~8 ms carrier) |

(warm, T=5000, K=5, the handover's scene.) Each **exact-matches the per-instant `.at()` values at the
sample instants** to machine precision — including the rotated 45°/90° instants — verified in
`tests/primitives/test_face_signal.py::test_clearance_and_plane_readbacks_match_per_instant_under_rotation`.
As with `boundary()`/`frame()`, *between* samples it interpolates the pose (not the scalar output);
resolve onto your track's own timestamps (you already do) and it's exact.

Partiality is unchanged: occluded cloud members stay `nan` + `present=False`; an empty-region face
raises `UnresolvableError` from `resolve_over` (matching the per-instant Unresolvable). Off-support
pose/query still raise eagerly.

## You're unblocked

Wire in the bounded clearance — `perp = plane().signed_distance(footprint)` and `bounded =
clearance(footprint)` are both now fast *and* (with the rotation fix) correct for spinning supports.
Ping back if anything in the readback shapes or the partiality surface doesn't match what you staged.
