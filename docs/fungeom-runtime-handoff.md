# fungeom handoff — what retarget needs to put the patch *runtime* on fungeom

> ## ✅ DELIVERED (fungeom `main`, commit `ccf5daf`) — by the fungeom session
>
> All of P0/P1/P2 below are built, tested, and on `main` (1319 tests, 100% coverage). **Not yet
> on PyPI** — `pip install -e ~/GitHub/functional_api` (or wait for the next tagged release) to get
> it. The moving patch:
>
> ```python
> from fungeom import FaceSignal
> patch = FaceSignal.of(face, pose)        # static Face + a TransformSignal  (= FaceSignal.of, per the design call)
> patch.plane().origin().resolve_over(s)   # (T,3)  patch points
> patch.plane().normal().resolve_over(s)   # (T,3)  normals
> patch.frame().resolve_over(s)            # (T,4,4) patch frames   (Face.frame deletes your invented frame)
> patch.boundary().resolve_over(s)         # ((T,K,3), mask) footprint vertices
> patch.clearance(foot_signal)             # ScalarSignal  (point) / pass a Point3BundleSignal -> ScalarBundleSignal (T,K)
> patch.contains(foot_signal)              # BoolSignal  (footprint membership over time)
> ```
>
> - **P0 transport family:** `Plane.transformed_by(Transform)`, `Face.transformed_by(Transform)`,
>   `Face.frame()`, `Face.boundary()`, `Face.clearance(Point3Bundle)`. (Static `Point3.transformed_by`
>   transport already worked via `Point3Signal.lift(p).transformed_by(P)`.) I put transport on the
>   *signal* side — `FaceSignal.of(face, pose)`, not `Face.transformed_by(pose)` — to keep the
>   layering acyclic; retarget only depended on the output surface, which is unchanged.
> - **P0 `FaceSignal`** — the moving patch (surface above). Partiality flows: an occluded/off-support
>   query is `Unresolvable`, never a NaN.
> - **P1 `TransformBundleSignal.key(j)` → `TransformSignal`** — pull one segment's pose-signal out of
>   the per-segment bundle to feed transport. (The earlier "no key()" worry was wrong: the square
>   holds under linear interpolation.)
> - **P2 `TransformSignal.from_matrices(times, (T,4,4))`** — the vectorized batch carrier; `resolve_over`
>   is **~50× faster** than the per-object path on an interpolating grid (8 ms vs 440 ms at T=5000),
>   exact-matching it. **Acceptance met.** Wrap your dense `(T,4,4)` segment pose with this.
> - **`FrameSignal`** (option 2) intentionally **not** built — deferred as the larger generalization,
>   as you leaned.
>
> Net: you can delete `lower_face` + the invented frame and read the runtime off
> `FaceSignal(F, P).<…>.resolve_over(track_sampling)`. The rest of this doc is the original ask, kept
> for context.

---

> ## ⚠ TWO BLOCKERS (fungeom 0.2.0) — found wiring retarget's runtime onto `FaceSignal`
>
> path A was implemented against 0.2.0 and then **reverted** (retarget stays on the numpy
> transport) because of two `FaceSignal` issues. Both must be fixed for path A to land.
>
> ### 1. 🐞 BUG — `FaceSignal.boundary()` drops rotation on the footprint
>
> `FaceSignal.frame()`/`.plane()` transport correctly under a rotating pose, but
> **`FaceSignal.boundary()` translates the footprint to the moving centroid without rotating the
> vertex offsets** — it returns `static_vertex + (R·centroid − centroid) + t` instead of
> `R·static_vertex + t`.
>
> ```python
> import numpy as np
> from fungeom import Face, Point3, Point3Bundle, Region2, FaceSignal, TransformSignal, Sampling
> pts = Point3Bundle.from_map({"a": Point3.at(0,0,0), "b": Point3.at(1,0,0), "c": Point3.at(0,1,0)})
> face = Face.on(pts.fit_plane(), Region2.hull(pts.in_frame(pts.fit_plane())))
> R = np.array([[0,-1,0],[1,0,0],[0,0,1.0]]); t = np.array([1.,2.,3.])
> m = np.eye(4); m[:3,:3] = R; m[:3,3] = t
> fs = FaceSignal.of(face, TransformSignal.from_matrices(np.array([0.0]), m[None]))
> b = np.asarray(fs.boundary().resolve_over(Sampling.at_times(np.array([0.0])))[0])[0]
> # GOT:      [[0.333, 3, 3], [0.333, 2, 3], [1.333, 2, 3]]
> # EXPECTED: [[0, 2, 3], [1, 2, 3], [1, 3, 3]]   ( = static @ R.T + t ; frame().origin is correct)
> ```
>
> `frame().origin` is correct (`R·centroid + t`), so transport the footprint vertices by the full
> pose the way `frame()` does, not just by the moved centroid. The shipped tests look
> translation-only; a rotation-bearing case catches it.
>
> ### 2. 🐢 PERF — `FaceSignal` accessors resolve per-instant (not vectorized)
>
> The vectorized carrier is great (`TransformSignal.from_matrices` build ≈ 0 ms;
> `TransformSignal.resolve_over` ≈ **8 ms** at T=5000). But everything derived from a
> `FaceSignal` resolves per-sample, blowing past the numpy einsum it must replace
> (**sub-millisecond** for the same transport). At **T=5000**:
>
> | call | time | vs numpy einsum |
> |---|---|---|
> | `TransformSignal.resolve_over` (plain) | 8 ms | baseline carrier |
> | `FaceSignal.plane().normal().resolve_over` | 20 ms | ~30× |
> | `FaceSignal.plane().origin().resolve_over` | 51 ms | ~80× |
> | `FaceSignal.boundary().resolve_over` | 497 ms | ~800× |
> | `FaceSignal.frame().resolve_over` | 751 ms | ~1000× |
>
> retarget's `points()/normals()/frames()/boundary_points()` are sub-ms numpy today and get
> called several times per patch per analysis, so a 20–1000× hit per call is a hard regression.
> The fix is to make the `FaceSignal` accessors resolve **vectorized over the sampling** the way
> the plain `TransformSignal` carrier does (apply the `(T,4,4)` pose stack to the static
> Face geometry in one numpy op), rather than constructing/resolving a Face per timestep.
>
> **retarget impact / status:** path A **reverted**; the whole patch runtime stays on the numpy
> transport (`lower_face` + einsum, fast + correct). Once both land in a 0.2.x, re-landing path A
> is the small swap in `core/geometry.py` + `core/schema/patch.py` already prototyped here.

---

**Audience:** an agent working in the **fungeom** repo (`~/GitHub/functional_api`).
**From:** the retarget session, after the substrate migration landed.
**Status of retarget:** patches are authored as `geometry=` callables returning a fungeom
`Face` (✅ shipped, on `master`). But the *runtime* is still numpy: at bind time retarget
calls `lower_face(Face) → (RigidTransform, boundary (K,3))` and then transports that per-frame
with its own einsum. fungeom is currently a **bind-time authoring DSL, not the runtime
substrate.** This note is the concrete list of fungeom additions that would let retarget make
the runtime fungeom-native and delete `lower_face` (+ its invented frame).

Everything below was checked against the installed **fungeom 0.1.0** (PyPI). I've split it into
"already there — don't rebuild" and "needed", with signatures and acceptance.

---

## The one thing retarget actually needs: a *moving patch*

A patch is a static `Face` **fixed in a segment frame that moves over time**. The segment pose
over time is a `TransformSignal` `P`. retarget needs, from `(F: Face, P: TransformSignal)`, the
patch's world geometry **as fungeom signals**, then materialized to `(T, …)` arrays at the
track timestamps. Mapping from retarget's query methods to what they'd call:

| retarget query | output | wants from fungeom |
|---|---|---|
| `patch.points()` | `(T,3)` | patch origin over time → `Point3Signal` → `resolve_over` |
| `patch.normals()` | `(T,3)` | plane normal over time → `Direction3Signal` → `resolve_over` |
| `patch.frames()` | `(T,3,3)` | patch frame over time → `TransformSignal` → `resolve_over` |
| `patch.boundary_points()` | `(T,K,3)` | footprint vertices over time → `Point3BundleSignal` → `resolve_over` |
| contact clearance | `(T,)` / `(T,K)` | `FaceSignal.clearance(point signal)` → `ScalarSignal`/`ScalarBundleSignal` |

The payoff (why this is worth fungeom work): **runtime decidability/partiality** — an occluded
support marker makes a clearance `Unresolvable`, so a contact gap is *honest* instead of a
silently-transported numpy NaN; the moving patch becomes a first-class object; and `lower_face`
+ the hand-rolled frame convention disappear.

---

## Already there — do NOT rebuild

Verified present in 0.1.0:

- The temporal types: `Point3Signal`, `Direction3Signal`, `Vec3Signal`, `PlaneSignal`,
  `ScalarSignal`, `ScalarBundleSignal`, `BoolSignal`, `Point3BundleSignal`, `TransformSignal`,
  `TransformBundleSignal`.
- `Point3Signal`: `.lift` (static → constant signal), `.transformed_by(TransformSignal)`,
  `.map`, `.resolve_over(Sampling) → ndarray`, derivatives (`.velocity`, `.speed`).
- `Point3BundleSignal`: `.transformed_by(...)`, `.key(k) → Point3Signal`, `.fit_plane`,
  `.resolve_over`.
- `PlaneSignal` (from `Point3BundleSignal.fit_plane`), `.signed_distance(Point3Signal)`.
- `Face` (static): `.plane()`, `.region()`, `.clearance(...)`, `.contains(...)`,
  `.closest_point(...)`. `Plane.frame(origin, tangent) → Transform` (static).
- `resolve_over(Sampling)` on the plain signals (→ `(T,)`/`(T,3)`/`(T,4,4)`), bundle signals
  (→ `(values, (T,N) present mask)`, nan = occluded).

So the temporal substrate is *mostly built*. The gaps are specifically about (a) **a Face that
moves**, (b) **transporting static geometry by a pose-signal**, and (c) **two
construction/extraction ergonomics + perf**.

---

## Needed for path A

### P0 — `FaceSignal` (the moving patch). The central ask.

A temporal `Face`: a `Face` whose plane moves with a `TransformSignal` while its `Region2`
(plane-local) stays fixed. Construct by transporting a static `Face` by a pose signal:

```python
Face.transformed_by(pose: TransformSignal) -> FaceSignal     # static Face fixed in a moving frame
# or, equivalently, a constructor:
FaceSignal.of(face: Face, pose: TransformSignal) -> FaceSignal
```

`FaceSignal` surface retarget will call (mirror the static `Face`, lifted to signals):

```python
FaceSignal.plane()      -> PlaneSignal
FaceSignal.region()     -> Region2                       # static (plane-local)
FaceSignal.frame()      -> TransformSignal               # see P1
FaceSignal.boundary()   -> Point3BundleSignal            # footprint vertices over time (-> boundary_points)
FaceSignal.clearance(p: Point3Signal | Point3BundleSignal)
                        -> ScalarSignal | ScalarBundleSignal
FaceSignal.contains(p: Point3Signal | Point3BundleSignal)
                        -> BoolSignal | BoolBundleSignal  # (or expose via clearance<=0 if no BoolBundleSignal)
FaceSignal.resolve_over / decide                          # partiality propagates from p and the pose
```

**Why:** this single type backs `points`/`normals`/`frames`/`boundary`/clearance and carries
partiality end-to-end. **Acceptance:** `Face.on(plane, region).transformed_by(P).clearance(q)`
resolves to the same numbers as the current numpy spine on a known case, and is `Unresolvable`
when `q` (or the pose) is.

### P0 — transport of static geometry by a `TransformSignal`

`Point3Signal` already composes this (`.lift(p).transformed_by(P)`); complete the family so the
Face pieces work:

```python
Plane.transformed_by(pose: TransformSignal)        -> PlaneSignal
Point3Bundle.transformed_by(pose: TransformSignal) -> Point3BundleSignal   # boundary corners
Direction3.rotated_by(pose: TransformSignal)       -> Direction3Signal     # normals (rotation only)
```

(`Point3.transformed_by(TransformSignal) -> Point3Signal` would be nice sugar over
`Point3Signal.lift(p).transformed_by(P)`, but the `lift` path already works for points.)
Note: `Point3Bundle`/`Plane` currently have **no** `lift`/`transformed_by(TransformSignal)`,
so this is genuinely missing for the bundle/plane/face that the boundary + clearance need.

### P1 — `Face.frame()` / `FaceSignal.frame()` (canonical patch frame)

```python
Face.frame()      -> Transform        # origin = region centroid; +z = plane normal; +x = a STABLE in-plane axis
FaceSignal.frame() -> TransformSignal
```

**Why:** `patch.frames()` needs a real source, and this **deletes `lower_face`'s invented
frame** (today retarget fabricates +x from the plane's uv-x to fill the legacy `[x,y,normal]`
slot). The only contract retarget needs is *determinism* (same Face → same frame) and that
column 2 is the normal. **Acceptance:** `Face.frame().rotation[:,2] == plane.normal()`, stable
under re-evaluation.

### P1 — `TransformBundleSignal.key(k) -> TransformSignal`

`Point3BundleSignal` already has `.key`; `TransformBundleSignal` does **not**. retarget builds
the per-segment pose bundle and needs to pull one segment's pose-signal out to feed transport.

```python
TransformBundleSignal.key(k: Hashable) -> TransformSignal
```

### P2 — vectorized `TransformSignal` construction (perf; the viability gate)

retarget has the segment pose as a dense `(T, 4, 4)` array (thousands of frames). Today the only
path is per-instant `Transform` wrappers via `from_samples` — the "40k-wrapper" cost flagged in
earlier audits. A batch carrier makes path A actually viable at scale:

```python
TransformSignal.from_matrices(times: ndarray (T,), matrices: ndarray (T,4,4)) -> TransformSignal
# backed by a vectorized RigidTransform carrier so resolve_over stays O(T) in numpy, not O(T) python objects
```

**Acceptance:** building + `resolve_over` for `T≈5000` is within a small constant of the current
numpy einsum (not 100×). This is the P1/P2 batch carrier from the original needs doc; it gates
whether path A is worth shipping at all.

---

## The design question for you (your call)

There are two idiomatic ways to express "static geometry fixed in a moving frame":

1. **Transport family + `FaceSignal`** (what P0 above assumes): `X.transformed_by(P)` for each
   geometry type, plus a `FaceSignal` type. Smallest, most direct fit for retarget.
2. **`FrameSignal`** — the temporal sibling of your static `Frame` (so `Scalar→ScalarSignal`,
   `Transform→TransformSignal`, **`Frame→FrameSignal`**). A patch's geometry authored *in* a
   `FrameSignal` is world-resolvable over time for free, and transport falls out as a special
   case. More unifying and arguably more in the spirit of "every primitive has a temporal
   family" — but a bigger abstraction, and you'd likely build the transport machinery underneath
   it anyway.

I lean toward **(1) ship the transport family + `FaceSignal` now**, and consider `FrameSignal`
later as the generalization — but you own fungeom's design; pick whichever keeps the resolver
algebra clean. retarget only depends on the **output surface** (a moving patch I can ask for
`points/normals/frame/boundary/clearance` as signals + `resolve_over`), not on how you get there.

---

## Non-goals / parked (don't build for this)

- Smoothing, hysteresis, statistical/iterative estimators — those stay numeric in retarget
  (parked for hidden modeling opinion / not-yet-resident, not for being "statistical"; admission is
  fungeom's own call — [`functional_api/docs/substrate-membership.md`](../../functional_api/docs/substrate-membership.md)).
- `BoolBundleSignal` is optional — retarget can derive per-support contact from
  `clearances.min().le(0)` if you'd rather not add it.
- The orphaned `retarget.fungeom` signals adapter (`marker_cloud_signal`/`pose_signal`/`relabel`)
  is retarget's problem to wire in once `FaceSignal` exists — no action needed on your side.

---

## How retarget consumes it (the end state, for context)

Once P0/P1 land (P2 for scale): `_bind_patch` stores the static `Face` (no `lower_face`); the
segment runtime exposes a `TransformSignal` per segment; `patch.points()/normals()/frames()/
boundary_points()` become `FaceSignal(F, P).<…>.resolve_over(track_sampling)`; the contact spine
reads `face_signal.clearance(support_point_signal)` with partiality flowing to the contact mask.
`lower_face`, the invented frame, and the bind-time numpy collapse all go away.
