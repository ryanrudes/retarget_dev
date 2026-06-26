# fungeom backend adapter — handoff spec

**Status:** plan only, nothing built in retarget yet. Authored from the **fungeom**
session that just shipped the seam (fungeom commit `ebf2c15`), so the fungeom surface
below is exact as of that commit. The adapter itself is to be built **in a dedicated
retarget session** (this repo's own AGENTS.md conventions apply — typed-first, enum-free,
TypedDict schemas, time-major arrays, frozen+slotted dataclasses, `Mapping` for read-only).

> Read order for the retarget session: this file → fungeom's `~/GitHub/functional_api/README.md`
> (the Combinators table) and `docs/collections.md` (the spine), then this repo's
> `AGENTS.md` + `docs/{axes,scene}.md`.

---

## 0. What this is

fungeom (`~/GitHub/functional_api`, package `fungeom`) is now feature-complete as the
*decidable geometry/temporal/collections backend* retarget needs. The seam that makes it
retarget's backend is **rung 3** (`Roster`/`RosterMap`) plus the two collection-over-time
signals. This adapter is the thin retarget-side layer that maps retarget's authored scene +
recorded arrays onto fungeom resolvers, and reads results back.

**fungeom carries identity + geometry + time. It does NOT do the numeric retarget transfer**
(where a target joint actually *goes* given source motion). That estimator is parked numerics,
exactly as DTW/ICP are — fungeom *calls* such a thing, it does not *be* it. `RosterMap` is the
**identity** correspondence (which marker is which joint); the geometric map layers on top.

---

## 1. The fungeom surface the adapter uses (exact, as of `ebf2c15`)

Everything is a lazily-evaluated **resolver**. `x.decide()` → `Resolvable(value)` |
`Unresolvable(reason)`; `x.resolve()` → the value (raises `UnresolvableError` if partial).
Partiality is first-class — an occluded marker, a disjoint sync, an antipodal blend are
`Unresolvable`, never exceptions. Import from `fungeom` (facades) and `fungeom.values`
(value types, for `isinstance`/raw construction).

### 1a. Positions over time — `Point3BundleSignal` (a marker cloud)

```python
from fungeom import Point3BundleSignal, Interpolation, Boundary
Point3BundleSignal.from_frames(
    times,                 # (T,) array of seconds
    frames,                # (T, N, 3) array of marker positions
    keys=None,             # marker names, len N (defaults to 0..N-1)
    frame=WORLD_FRAME,     # one shared CoordinateFrame for the whole cloud
    via=Interpolation.linear,
    outside=Boundary.undefined,
    max_gap=None,          # seconds; samples spaced wider are a TEMPORAL dropout
    present=None,          # (T, N) bool occlusion mask; False = marker absent that frame
) -> Point3BundleSignal
```
- `.at(t) -> Point3Bundle` — the cloud at one instant (bridges to the static algebra).
- `.key(marker) -> Point3Signal` — one marker's trajectory (the entity-axis slice); gaps where occluded; **only valid under the default reconstruction** (linear + undefined), else Unresolvable.
- inherited: `.over() -> Interval`, `.support() -> Coverage`, `.resample(Sampling)`, `.restrict(Interval|Coverage)`, `.shift(Duration|float)`, `.reparameterize(TimeMap|TimeWarp)`.
- Occlusion mask falls out for free: interpolating across a dropout leaves that marker absent; an exact frame returns it. `at(t).at(k) == key(k).at(t)` (the commuting square) holds on the support.

### 1b. Rotations/poses over time — `TransformBundleSignal` (a skeleton's joints)

```python
from fungeom import TransformBundleSignal, Transform
from fungeom.values import RigidTransform
from scipy.spatial.transform import Rotation

TransformBundleSignal.from_frames(
    times,                 # (T,) seconds
    frames,                # (T, N) GRID OF Transform RESOLVERS — NOT a raw array (see Gotchas)
    keys=None,             # joint names, len N
    via=Interpolation.linear,
    outside=Boundary.undefined,
    max_gap=None,
    present=None,          # (T, N) bool occlusion mask
) -> TransformBundleSignal
```
- `.at(t) -> TransformBundle` — the pose-set at one instant.
- **No `.key()`** — the SE(3) blend is *partial* (slerp across opposed orientations) and strict over that op-failure, so the entity-axis square can't hold. Get one joint at an instant with `at(t).at(k)`.
- inherited time-ops: same as `Point3BundleSignal`.
- **Building poses** (retarget has `(T, N, 3, 3)` rotation matrices or `(T, N, 4)` quaternions):
  - from a 3×3 rotation + translation: `Transform.known(RigidTransform.from_rotation(Rotation.from_matrix(R3x3), t3))`
  - from a quaternion: `Transform.known(RigidTransform.from_rotation(Rotation.from_quat(q4), t3))`
  - from a 4×4 homogeneous: `Transform.known(RigidTransform.from_matrix(M4x4))`
  - (`Rotation.from_quat` uses scipy's `[x,y,z,w]` order — confirm retarget's quaternion layout.)

### 1c. Static collections (at one instant) — `Point3Bundle` / `TransformBundle`

`.at(key) -> Point3|Transform`, `.present(key) -> Bool`, `.count() -> Scalar`,
`.support() -> Roster`, `.where(keys)`, `.relabel(RosterMap)`.
`Point3Bundle` extras: `.centroid()`, `.displacement_to(other) -> Vec3Bundle`,
`.distance_to(other) -> ScalarBundle`, `.transformed_by(Transform)`, `.fit_plane()`, `.fit_line()`.
Construct directly (outside a signal) with `Point3Bundle.from_map({key: Point3}, roster=…)` /
`.from_array((N,3), keys, frame)`; `TransformBundle.of([...], keys)` / `.from_map(...)`.

### 1d. The identity seam — `Roster` / `RosterMap`

```python
from fungeom import Roster, RosterMap
Roster.of(keys); Roster.empty
roster.union/intersection/difference(other); roster.count() -> Scalar; roster.contains(key) -> Bool

RosterMap.of({source_key: target_key})          # the marker↔joint correspondence (landmarks)
RosterMap.identity(roster_or_keys); RosterMap.known(value)
m @ inner            # compose (total; domain narrows to keys that chain through)
m.inverse()          # PARTIAL: Unresolvable if non-injective (two sources share a target)
m.source() -> Roster; m.target() -> Roster; m.maps(key) -> Bool

cloud.relabel(m)     # THE IDENTITY TRANSFER: re-key a bundle source→target; drops out-of-domain
                     # keys; carries the occlusion mask; partial if it collapses two keys onto one
```

### 1e. Time alignment & resampling (sync between recordings)

- `TimeMap.aligning(source, target)` — one landmark → offset (unit rate).
- `TimeMap.through((s0,t0),(s1,t1))` — two landmarks → offset + rate (drift). Partial if sources coincide.
- `signal.reparameterize(TimeMap | TimeWarp)` — apply the recovered map; `TimeWarp.through(knots)` for monotonic content warps.
- `signal.resample(Sampling.uniform(Interval.between(Instant.at(a), Instant.at(b)), count))` — onto a common rate.
- `Coverage` / `signal.support()` — honest gaps; querying a dropout is Unresolvable.

---

## 2. The mapping (retarget data → fungeom)

| retarget datum | fungeom |
| --- | --- |
| `(T, N, 3)` marker positions + names + occlusion | `Point3BundleSignal.from_frames(times, pos, keys=names, present=mask)` |
| `(T, N, 3, 3)` / `(T, N, 4)` joint rotations | `TransformBundleSignal.from_frames(times, grid_of_Transform, keys=joints, present=mask)` (wrap each pose, §1b) |
| marker set ↔ target skeleton joints | `RosterMap.of({marker: joint})` |
| identity transfer (re-key source onto target) | `source_cloud.relabel(map)` |
| sync two recordings from landmarks | `TimeMap.aligning/through` → `sig.reparameterize(tm)` |
| resample onto a common clock | `sig.resample(Sampling.uniform(...))` |
| "where is this defined?" | `sig.support()` (Coverage), `bundle.present(k)`, every `decide()` |
| nearest/centroid/distances at an instant | `cloud.at(t).centroid()/.distance_to(...).min()` etc. |

---

## 3. Adapter shape (plan — defer internals to retarget's AGENTS.md)

A new module (suggest `src/retarget/fungeom/` or wherever the house style puts integration
layers) with, roughly:

1. **`add fungeom as a dependency`** — `pyproject.toml`: a path/editable dep on
   `~/GitHub/functional_api` (or git). fungeom is Python 3.13, deps `numpy`/`scipy`/`rich`.
   Confirm the version-pinning convention this repo uses.
2. **demonstration → signals**: functions that take retarget's compiled scene/demonstration
   (the frozen dataclasses bound from TypedDict schemas) and emit `Point3BundleSignal` (markers)
   and `TransformBundleSignal` (segment/joint poses), using the authored string names as keys
   and the existing occlusion/validity tracks as the `present` mask. Keep retarget's time-major
   `(T, …)` arrays as the native input — they slot straight into `from_frames`.
3. **correspondence → `RosterMap`**: build from retarget's authored marker↔joint mapping.
4. **transfer**: `relabel` for the identity step; leave a clearly-marked seam where the
   *numeric* geometric estimator plugs in (out of scope, parked — mirror how retarget already
   treats its other numeric kernels).
5. **read-back**: helpers to pull values out (`.at(t).at(k).resolve()`, `.support()`), surfacing
   `Unresolvable` reasons rather than swallowing them (matches retarget's typed-first honesty).

Keep the adapter **thin and typed**: it translates names/arrays ↔ fungeom resolvers and nothing
more. No geometry logic lives here — that's fungeom's job.

---

## 4. Gotchas (learned building the fungeom side)

- **`TransformBundleSignal.from_frames` takes a `(T, N)` grid of `Transform` *resolvers*, not a
  raw array** (SE(3) has no ergonomic raw form — same reason `TransformBundle` has no
  `from_array`). Build the grid by wrapping each matrix/quat via `Transform.known(RigidTransform.from_*)`.
- **The SE(3) bundle blend is strict over op-failure**: if *any* present-in-both joint is
  antipodal (opposed orientations) at an interpolated instant, the *whole* interpolated pose-set
  is `Unresolvable` (an op-failure is never disguised as absence). Rare for real mocap (adjacent
  frames are close), but the adapter should surface the reason, not crash. Exact-frame samples
  bypass the blend and always resolve.
- **No `key()` on `TransformBundleSignal`** (see §1b). Use `at(t).at(k)` for a single joint.
- **One shared frame per cloud** — `Point3Bundle`/`Point3BundleSignal` carry a single coordinate
  frame for the whole cloud (like a signal's one timeline). Per-marker frames are out of scope.
- **Construction is strict**: an unresolvable member fails the *whole* bundle/frame (a detached
  frame, a degenerate rotation). Occlusion must be a deliberate `present=` mask, never a silent
  drop. So validate/ground inputs before building, or expect a build-time `Unresolvable`.
- **`relabel` is identity only** — it renames keys and carries values across unchanged; it does
  NOT move geometry. That's the parked numeric transfer.
- **Quaternion order**: `Rotation.from_quat` is scipy's `[x,y,z,w]`. Check retarget's layout.
- **Keys are bare `Hashable`** — retarget's authored string names ARE the identity; no enums
  (matches retarget's enum-free model). `Roster`/`RosterMap` align by key identity, never position.

---

## 5. Open questions for the retarget session

1. Exact array layouts & dtypes of the compiled demonstration (marker `(T,N,3)`? rotation `(T,N,3,3)` vs quat `(T,N,4)`? where does the occlusion/validity mask live?).
2. Where the authored marker↔joint correspondence is declared (so `RosterMap.of` reads from the source of truth, not a hand-written dict).
3. The fungeom dependency mechanism this repo wants (editable path vs pinned git).
4. Whether sync/resampling should move onto fungeom's `TimeMap`/`Sampling`/`Coverage` now, or stay in retarget's existing `demo_sync_resampling` path for this first pass (the adapter can do positions/poses first, time later).
5. The seam contract for the future numeric transfer estimator (its input = source `RosterMap` + clouds; its output = a fitted geometric map — design when pulled, like fungeom did for rung 3).

---

*fungeom side is done & green (955 tests, 100% cov, commit `ebf2c15`). This adapter is the
last mile, and it lives here.*
