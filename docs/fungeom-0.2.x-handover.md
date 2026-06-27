# fungeom 0.2.x handover — two `FaceSignal` blockers before retarget can use the runtime

> ## ✅ RESOLVED (fungeom `main`, commit `1141bdb`) — by the fungeom session
>
> Both blockers are fixed for the priority accessors. **On `main`, not yet on PyPI** —
> `pip install -e ~/GitHub/functional_api` (or wait for a `v0.2.1` tag).
>
> - **Blocker 1 (correctness):** `boundary()` **and** `frame()` now transport the *static* geometry
>   rigidly — `boundary()` = `static_vertices · pose` (`R·v + t`), `frame()` = `pose ∘ static_frame`
>   (so it rotates with the pose). Your repro now returns `[[0,2,3],[1,2,3],[1,3,3]]`. The root cause
>   was re-embedding the 2D region into the *transported* plane's re-gauged chart; a rotation-bearing
>   test now guards it.
> - **Blocker 2 (perf):** `boundary()`/`frame()` `resolve_over` are vectorized (apply the `(T,4,4)`
>   pose stack to the static geometry in one numpy op) — **497 ms → ~7 ms** and **751 ms → ~9 ms** at
>   T=5000, within a small constant of the 8 ms carrier. The `at()`/per-instant path matches the
>   vectorized readback.
> - **One remaining (smaller) item:** `plane().normal()`/`origin()` are *correct* (transport is exact)
>   but still resolve per-instant (~20 / ~51 ms). The ≥500 ms killers you flagged as the priority are
>   done; vectorizing normal/origin is a quick follow-up — say the word if 20 ms/normals call still
>   bites your loop. (Also: `contains()` may share the old re-gauge subtlety — `clearance()` is fine
>   because a 3-D distance is gauge-independent — flag it if you hit it; you use `clearance` for contact.)
>
> Net: re-land path A — `fs.frame()/.boundary()` are now correct *and* fast. The rest of this doc is
> the original report, kept for context.

**Audience:** an agent working in the **fungeom** repo (`~/GitHub/functional_api`).
**From:** the retarget session. **TL;DR:** 0.2.0 delivered the `FaceSignal` runtime API and it's
*shaped* right — retarget wired its patch runtime onto it in an afternoon. But two issues made it
unshippable, so retarget **reverted to its numpy transport**. Both are fungeom-side. Fix them in a
**0.2.x** and retarget lands the runtime swap (already prototyped) immediately.

Nothing here is a redesign — the API (`FaceSignal.of(face, pose)`, `.frame()/.plane()/.boundary()/
.clearance()`, `Face.frame()`, `TransformBundleSignal.key()`, `TransformSignal.from_matrices`) is
all correct and retarget depends on exactly that surface. These are a correctness bug and a perf
bug *inside* two of those methods.

Verified against installed **fungeom 0.2.0** (PyPI). Keep the repo gate green
(`ruff check`, `ruff format --check`, `mypy --strict`, `pytest --cov` at 100%).

---

## Blocker 1 — 🐞 `FaceSignal.boundary()` drops rotation on the footprint

`frame()` and `plane()` transport correctly under a rotating pose, but **`boundary()` translates
the footprint to the moved centroid without rotating the vertex offsets.** It returns
`static_vertex + (R·centroid − centroid) + t` instead of the correct `R·static_vertex + t`.

Runnable repro (triangle footprint, 90°-Z + translation, single frame):

```python
import numpy as np
from fungeom import Face, Point3, Point3Bundle, Region2, FaceSignal, TransformSignal, Sampling

pts  = Point3Bundle.from_map({"a": Point3.at(0,0,0), "b": Point3.at(1,0,0), "c": Point3.at(0,1,0)})
face = Face.on(pts.fit_plane(), Region2.hull(pts.in_frame(pts.fit_plane())))
R = np.array([[0,-1,0],[1,0,0],[0,0,1.0]]); t = np.array([1., 2., 3.])
m = np.eye(4); m[:3,:3] = R; m[:3,3] = t
fs = FaceSignal.of(face, TransformSignal.from_matrices(np.array([0.0]), m[None]))

got = np.asarray(fs.boundary().resolve_over(Sampling.at_times(np.array([0.0])))[0])[0]
# GOT:      [[0.333, 3, 3], [0.333, 2, 3], [1.333, 2, 3]]
# EXPECTED: [[0, 2, 3], [1, 2, 3], [1, 3, 3]]      ==  static_vertices @ R.T + t
# (frame().origin() is already correct: R·centroid + t)
```

**Fix:** transport the footprint vertices by the full pose the same way `frame()` does, not just by
the moved centroid. **Acceptance:** a rotation-bearing test — the shipped tests appear to be
translation-only, which is why this slipped through. `boundary()` under any `R` must equal
`static_vertices @ R.T + t`, and partiality must still propagate (occluded pose → `Unresolvable`).

---

## Blocker 2 — 🐢 the `FaceSignal` accessors resolve per-instant (the real blocker)

The vectorized carrier is excellent — `TransformSignal.from_matrices(...)` builds in ~0 ms and its
`resolve_over` is **8 ms at T=5000**. But everything derived from a `FaceSignal` resolves
**per-sample**, which blows past the numpy einsum it has to replace (**sub-millisecond** for the
same rigid transport). Measured at **T=5000** (single patch, 90°-Z sweep + translation):

| call | time at T=5000 | vs numpy einsum |
|---|---:|---:|
| `TransformSignal.resolve_over` (plain carrier) | **8 ms** | baseline |
| `FaceSignal.plane().normal().resolve_over` | 20 ms | ~30× |
| `FaceSignal.plane().origin().resolve_over` | 51 ms | ~80× |
| `FaceSignal.boundary().resolve_over` | 497 ms | ~800× |
| `FaceSignal.frame().resolve_over` | 751 ms | ~1000× |

Runnable timing (reuse as a perf-regression test):

```python
import time, numpy as np
from fungeom import Face, Point3, Point3Bundle, Region2, FaceSignal, TransformSignal, Sampling
pts  = Point3Bundle.from_map({"a": Point3.at(0,0,0), "b": Point3.at(1,0,0), "c": Point3.at(0,1,0)})
face = Face.on(pts.fit_plane(), Region2.hull(pts.in_frame(pts.fit_plane())))
T = 5000; times = np.arange(T) * 0.01
ang = np.linspace(0, np.pi/2, T); c, s = np.cos(ang), np.sin(ang)
m = np.zeros((T,4,4)); m[:,0,0]=c; m[:,0,1]=-s; m[:,1,0]=s; m[:,1,1]=c; m[:,2,2]=1; m[:,3,3]=1; m[:,0,3]=np.linspace(0,3,T)
fs = FaceSignal.of(face, TransformSignal.from_matrices(times, m)); samp = Sampling.at_times(times)
t0 = time.perf_counter(); np.asarray(fs.frame().resolve_over(samp)); print("frame", (time.perf_counter()-t0)*1000, "ms")
```

**Why it matters for retarget:** `points()/normals()/frames()/boundary_points()` are sub-ms numpy
today and get called several times per patch per analysis. A 20–1000× hit per call is a hard
regression — a few-thousand-frame demo would go from instant to multiple seconds.

**Fix:** make the `FaceSignal` accessors resolve **vectorized over the sampling**, the way the plain
`TransformSignal` carrier already does — apply the materialized `(T,4,4)` pose stack to the static
Face geometry in one batched numpy op, rather than constructing/resolving a `Face` (and its
plane/region/frame) per timestep. The static geometry (`face.frame()`, `face.plane().normal()`,
`face.boundary()` vertices) is fixed; only the rigid transport varies per sample, so each accessor
is one `einsum` over the pose stack.

**Acceptance:** at T=5000 every accessor in the table resolves within a **small constant of the
8 ms carrier** (not 20–1000×), exact-matching the per-instant values. The `frame()`/`boundary()`
rows are the priority (≥500 ms today).

---

## How retarget will consume it (so you can sanity-check the fix)

Once both land, retarget re-applies the already-prototyped swap (in `core/geometry.py` +
`core/schema/patch.py`):

```python
pose = TransformSignal.from_matrices(timestamps, pose_matrices_T44)   # the vectorized carrier
fs   = FaceSignal.of(bound_face, pose)
points  = fs.frame().resolve_over(Sampling.at_times(timestamps))[:, :3, 3]   # (T,3)
normals = fs.plane().normal().resolve_over(...)                              # (T,3)
frames  = fs.frame().resolve_over(...)[:, :3, :3]                            # (T,3,3)
bounds, present = fs.boundary().resolve_over(...)                            # (T,K,3), (T,K)
```

`lower_face` and the hand-rolled in-plane frame go away; the contact spine moves to
`fs.clearance(support_point_signal)` with partiality flowing to the contact mask. That last step
is the actual payoff (occlusion → honest `Unresolvable` contact gaps), and it needs both fixes
above to be correct *and* fast.

Empty edge case to keep working: retarget guards `T == 0` itself, but note `Sampling.at_times([])`
currently raises `UnresolvableError("a sampling has no times")` — fine, just flagging it.

---

## Status on the retarget side — ✅ LANDED on 0.2.1

Both fixes verified (`fungeom>=0.2.1`), and path A is **landed**:

- Runtime is now fungeom-native: `Patch.points()/normals()/frames()/boundary_points()` resolve
  through `FaceSignal.of(bound_face, segment_pose_signal).frame()/boundary().resolve_over(...)`.
  `lower_face` and the hand-rolled in-plane frame are deleted — the patch frame is `Face.frame()`.
- `normals()` is derived from `frame()` column 2 (vectorized) rather than the still-per-instant
  `plane().normal()` — so all four queries ride the vectorized `frame()`/`boundary()` path.
- End-to-end at T=4000: `points` 11 ms, `normals` 5 ms, `frames` 5 ms, `boundary_points` 6 ms
  (was 750/497 ms). Harness green (mypy strict 55 files, full suite).
- **Remaining for later (S3):** move the contact spine onto `FaceSignal.clearance(...)` so
  occlusion → `Unresolvable` contact gaps. `plane().normal()`/`origin()` vectorization is no longer
  on retarget's critical path (we use `frame()`); ping if a future use needs them.
