# fungeom needs for the retarget substrate — MASTER HANDOFF (retarget → fungeom)

**Status:** complete forward inventory, authored from the **retarget** session for a
**fungeom** session to implement. This is the umbrella over `docs/region2-handoff.md` (the
single biggest item): it lists **every** net-new fungeom primitive/combinator the substrate
migration needs, so the whole set can be built up front instead of discovered one at a time.
Derived from an exhaustive 5-domain sweep + a completeness-critic pass against live fungeom
source. IDs (G#/C#/T#/S#/P#/X#) are stable handles.

> Read order (fungeom session): this file → `docs/region2-handoff.md` → fungeom `README.md`
> (combinator table, rung-3 precedent) → fungeom `CHECKLIST.md` (add-a-primitive procedure).
> retarget repo: `~/GitHub/retargeting_from_scratch`; fungeom source under
> `~/GitHub/functional_api/src/fungeom/primitives`.

## The governing rule (what belongs in fungeom)

**Exact / closed-form / combinatorial ⇒ fungeom. Statistical / iterative / smoothing ⇒
parked numeric (retarget-side, consuming fungeom values).** This single rule resolves every
contested item below. Note the consequence: SVD plane/Kabsch fits, finite-difference
derivatives, convex hull, polygon clipping, bundle median/argmin are **exact** → in scope;
Savitzky-Golay/RMS smoothing, RANSAC, SE(3) Fréchet mean, hysteresis, chi-squared confidence,
heightmaps are **statistical/iterative** → parked.

---

## 1. Architecture decisions (settle FIRST — they shape everything)

These cross fungeom's "closed resolver menu" house style. **All six are now LOCKED decisions
(retarget-side best judgement) — build to them, don't re-litigate — with ONE exception: A4
(finite-difference derivative) is the fungeom session's philosophy call** (it brushes fungeom's
"numerics out of scope" line; our preference is ship-as-exact-combinator, but if you judge it
numerics we keep it parked in retarget — clean either way). Each item's "Rec:" is the locked call.

- **A1 — Generic typed map / lift (the keystone).** A public `Bundle.map_*(fn)` and
  `Signal.map(f)/lift([sigs], f)`. "Define geometry any way imaginable" *requires* an open
  escape hatch; `decide_mapped`/`decide_lifted` already exist internally, so the public form
  is thin. **Rec: ship it**, alongside operand-side named broadcasts (`Plane.signed_distance
  (bundle)`, `Region2.contains(bundle)`) for the hot closed paths. The generic map *is* the
  new house style for the open substrate. (Blocks the open-ness of S2/S3.)
- **A2 — Sanction vectorized ndarray readback.** `Signal.resolve_over(Sampling) → ndarray`
  (+ bundle `→ (T,N,·)` & present mask). retarget's `marker.positions() → (T,3)` contract
  needs a vectorized exit; a per-instant Python loop is non-viable at 40 k samples. **Rec:
  sanction as public API** (not just a hidden escape hatch). Gates S1 viability.
- **A3 — Vectorized `RigidTransform` batch carrier.** A batch carrier backed by `(T,N,4,4)`
  (or `(T,N,3,3)+(T,N,3)`) arrays consumed directly by the bundle-signals — *not* merely a
  `Transform.from_matrices(...) → tuple[Transform,…]` (which keeps the 40 k Python objects).
  The spike's 143 ms / 40 k-wrapper build cost only disappears with the carrier. **Rec: batch
  carrier.** Gates S1 viability.
- **A4 — Finite-difference derivative in fungeom.** `Point3Signal.velocity()/speed()`,
  `ScalarSignal.derivative()`, `TransformSignal.angular_velocity()` (closed-form SO(3) log —
  Rodrigues, not manifold-iterative). FD on the sample grid with a fixed scheme is exact →
  in scope. **Rec: ship.** Smoothing derivatives (Savitzky-Golay) stay parked. Blocks S3 speed
  channels.
- **A5 — `BoolSignal`.** Whole-timeline boolean: `ScalarSignal.lt/le(thr) → BoolSignal`,
  `& | ~`, `at(t) → Bool` (Unresolvable in a gap — keeps three-valued contact honest),
  `when_true() → Coverage`, `first_true() → Instant`. A `Coverage` alone cannot carry the
  occluded/undefined third state. **Rec: ship** (single-threshold; hysteresis stays parked).
  Blocks S3.
- **A6 — Affine `Transform2`.** Make `Transform2` carry shear/affine (or add `Affine2`) so
  `Region2.transformed_by` supports oblique plane→plane projection (cross-patch composition,
  M1). Today it is rigid-only and that composition is unbuildable. **Rec: yes** (lower
  urgency — only blocks cross-patch/deck-projection definitions).

---

## 2. Build order by tier

Implement top-down; **B2/B3/P gate the migration**, E/N are catalogued so they're not
surprises but built on demand.

| tier | meaning | items |
|---|---|---|
| **B2** | blocks S2 (open patch algebra) | G1 (Region2), G3 (Plane↔2D bridge), G4 (Point2Bundle), G5 (affine Transform2 + Plane.projection_to), G7, G8, G9, G10, G12, C7, T2 (lift, keystone), T8 (PlaneSignal) |
| **B3** | blocks S3 (contacts spine) | G2 (Region2 point-distance), G6 (Face), G13, C1 (BoolBundle), C2, C3 (map), C4, C5 (argmin→key), C6 (nearest→key), T1 (transport), T3 (BoolSignal), T4 (derivative), T5 (temporal reductions), T6, T9 (bundle-signal folds) |
| **P** | perf — gates S1 viability | P1 (batch carrier), P2, P3 (resolve_over), P4 |
| **E** | ergonomic | G11 (Kabsch), G14, T7, T10, X1, C8, S2 |
| **N** | deferred / build-on-demand (catalogued, not required up front) | G15, C9, C10, C11, T11, T12, X2, S1 |

---

## 3. The inventory (by area; tier in brackets)

### 3a. Static geometry
- **G1 [B2] `Region2` rung** — full spec in `docs/region2-handoff.md` (rect/disc/polygon/hull;
  ∪ ∩ \\ ; offset; contains/boundary/sample/corners/area/centroid/bounds; transformed_by).
- **G2 [B3] `Region2.signed_distance(Point2)→Scalar` + `nearest_boundary_point(Point2)→Point2`**
  — balance-board / ZMP stability margin (inside +, outside −). Empty → Unresolvable. *(Extends
  the Region2 contract — see region2-handoff §6.)*
- **G3 [B2] Plane↔2-D bridge** — `Plane.to_local(Point3)→Point2`, `Plane.embed(Point2)→Point3`
  (+ bundle broadcasts). Ungrounded frame → Unresolvable. The load-bearing 3D↔2D link for
  `Region2.hull(markers.in_frame(plane))` and lifting region samples back to world.
- **G4 [B2] `Point2Bundle`** + `Point3Bundle.in_frame(plane|frame)→Point2Bundle`.
- **G5 [B2] affine `Transform2`/`Affine2` + `Plane.projection_to(Plane)→Affine2`** (A6) —
  oblique plane→plane projection for cross-patch / patch-on-deck region composition.
  Parallel/degenerate planes → Unresolvable.
- **G6 [B3] `Face`/`OrientedRegion3`** = `Plane` + `Region2`; `closest_point(Point3)` (clamped
  into the region, like `Segment.project` vs `Line.project`), `clearance(Point3)→Scalar`. The
  *honest* bounded-patch clearance when the foot is beside, not above, the patch. *(Extends the
  Region2 contract.)*
- **G7 [B3] `Plane.signed_distance/project/contains(Point3Bundle)`** → `ScalarBundle`/
  `Point3Bundle`/`BoolBundle` (operand-side broadcast, per-key occlusion partiality).
- **G8 [B2] `Transform.aligning(a:Direction3, b:Direction3)` / `Direction3.rotation_to`** —
  shortest-arc rotation; antipodal → Unresolvable.
- **G9 [B2] `Transform.from_axes(x, y, z?, origin)`** — triad → rigid transform (z = x×y
  default); parallel/left-handed → Unresolvable. (The patch frame from plane normal + tangent.)
- **G10 [B2] `Transform.look_at(eye, target, up)`** — eye==target or up∥view → Unresolvable.
- **G12 [B2] `Vec3.scalar_triple(b, c)→Scalar`** — signed volume / winding sign. Total.
- **G13 [B3] `Direction3.signed_angle_to(other, *, about:Direction3)→Scalar`** — signed
  in-plane angle (RH about axis); vanishing ⟂-component → Unresolvable.
- **G14 [E] `Plane.frame_hinted(origin, tangent_hint)`** — project hint, fallback
  `any_perpendicular` (matches retarget `fit_patch_frame`). Total.
- **G11 [E] `Transform.aligning(src:Point3Bundle, dst:Point3Bundle)` (Kabsch)** — closed-form
  SVD rigid fit, **no scale, no RANSAC** (RANSAC + SE(3) Fréchet mean stay parked). <3
  non-collinear → Unresolvable.
- **G15 [N] long tail** — `Transform.orthonormalized()`; plane-mirror `Point3.reflect_across_plane`
  / `Direction3.reflect_across`; `Line.parameter_of` / `Line.closest_point_to(Line)` /
  `distance_to(Line)`; `Plane.angle_to(Plane)` (dihedral); `Region2` support-function /
  Minkowski-sum; world→local `Point3.relative_to(frame)`.

### 3b. Bundles / collections (single instant)
- **C1 [B3] `BoolBundle`** — key-aligned `and_/or_/not_`, `any()/all()→Bool`, `count()→Scalar`
  (output of G7 / `Region2.contains`).
- **C2 [B3] `Bundle.presence_mask()→BoolBundle`, `all_present()/any_present()→Bool`** —
  occlusion mask as a value.
- **C3 [B3] generic per-member map** `Point3Bundle.map_scalar/map_point/map_vec3(fn)` (A1) —
  the open escape hatch; thin public lift of `decide_mapped`.
- **C4 [B3] `Point3Bundle.distances_to(p:Point3)→ScalarBundle`** (one-query broadcast).
- **C5 [B3] `ScalarBundle.argmin()/argmax()→key` + `min_entry()/max_entry()→(key,Scalar)`** —
  which corner/marker is extreme. Empty → Unresolvable; tie → roster-order (see scope Q5).
- **C6 [B3] `Point3Bundle.nearest_to(p)→key` + `closest_point_to(p)→Point3`.**
- **C7 [B2] `Point3Bundle.bounds(frame)→(Point2,Point2)`** (AABB/extent in a frame).
- **C8 [E] predicate subsets** — `where_present()`, `masked_by(BoolBundle)`, `where(Roster)`.
- **C9–C11 [N]** — `InstantBundle` (+ argmin) for "which corner penetrates first" (M3); exact
  reductions `ScalarBundle.median()`, `Point3Bundle.diameter()/spread()`, `smallest(k)`,
  `weighted_centroid`; collection long tail (`Vec2Bundle`/`PlaneBundle`/`FrameBundle`, roster
  set-ops).

### 3c. Over-time signals
- **T1 [B/S1-core] transport family** — `Point3Signal.transformed_by(TransformSignal)`,
  `Point3BundleSignal.transformed_by(TransformBundleSignal|TransformSignal)`,
  `Vec3Signal.transformed_by`, `Direction3Signal.rotated_by`. Lift local geometry through a
  pose signal to world; off pose-support / ungrounded → Unresolvable. *(Removes the adapter's
  numeric precompute; with A3 carrier this is the modeled-marker path.)*
- **T2 [B2/B3 keystone] general lift** — `Signal.map(f, blend, on=…)` + N-ary `lift([sigs], f)`
  (A1's temporal twin). Aligns on intersected support; `f(t)` partiality flows.
- **T3 [B3] `BoolSignal`** (A5).
- **T4 [B3] finite-difference derivative** (A4) — `Point3Signal.velocity()/speed()`,
  `Vec3Signal.derivative()`, `ScalarSignal.derivative()`, `TransformSignal.angular_velocity()
  /angular_speed()`. Explicit scheme (central default); <2 samples / across gap → Unresolvable.
- **T5 [B3] temporal reductions over `Interval`/`Coverage`** — `ScalarSignal.min_over/max_over/
  mean_over/integral_over→Scalar`, `argmin_over/argmax_over→Instant`. Exact for piecewise-linear;
  window∩support empty → Unresolvable.
- **T6 [B3] `ScalarSignal` const/threshold broadcast** — `constant(v, over)`, `offset(c)/scale(c)`,
  compare-by-constant.
- **T7 [E] named over-time lifts** — `Vec3Signal.norm()→ScalarSignal`, `Vec3Signal.dot→ScalarSignal`,
  `Direction3Signal.angle_to→ScalarSignal`, pointwise `Signal.minimum/maximum(other)` (M2/M6).
- **T8 [B2/B3] `PlaneSignal`** (+ `LineSignal`) — `normal()→Direction3Signal`,
  `signed_distance(Point3Signal)→ScalarSignal`; **`Point3BundleSignal.fit_plane()→PlaneSignal`**
  (batched SVD — also closes the 11× per-instant-fit gap). Moving patch surface over time.
- **T9 [B3] generic `BundleSignal[V]`** → `ScalarBundleSignal`/`BoolBundleSignal` (+Vec3/Dir3)
  with folds `min/max/mean→ScalarSignal`, `any/all→BoolSignal`, `count→ScalarSignal` (M7 —
  footprint-min-clearance / "any corner in contact" over time).
- **T10 [E]** `TransformBundleSignal.key(k)→TransformSignal`, `Point3BundleSignal.centroid()→
  Point3Signal/.spread()`.
- **T11–T12 [N]** categorical/key signal "which corner lowest over time" (M8, scope Q6);
  `Signal.concat/overlay`.

### 3d. Time base / sampling
- **S1 [N]** `Sampling.union/merge/intersection/restrict/subsample/uniform_over`.
- **S2 [E]** `Coverage.close_gaps(Duration)/drop_shorter_than(Duration)` — exact 1-D
  morphological closing/opening (the temporal twin of `Region2.offset`).
- *(Sync **estimation** — xcorr / DTW — stays parked; fungeom consumes a `TimeMap`/`TimeWarp`,
  never fits one.)*

### 3e. Performance / readback (gate S1 viability)
- **P1 [P] vectorized `RigidTransform` batch carrier** (A3) — the only thing that removes the
  40 k Python wrappers (143 ms spike cost).
- **P2 [P] array constructors** — `TransformSignal.from_matrices`, `TransformBundleSignal.from_arrays`,
  vectorized build inside `Point3BundleSignal.from_frames`.
- **P3 [P] `Signal.resolve_over(Sampling)→ndarray`** (A2; V-agnostic `(T,)/(T,3)/(T,4,4)`) +
  bundle `resolve_over → (T,N,·) + (T,N) present mask`; `ScalarBundle.to_array()`.
- **P4 [P]** store time base as ndarray inside `ExplicitSampling` (drop `tuple[float,…]`);
  vectorized `linear`/`hold` resample fast path.

### 3f. Per-instant scalar/bool
- **X1 [E] `Bool.select(if_true, if_false)→Scalar`** (+Vec3/Point3), strict propagation —
  branching primitive.
- **X2 [N] `Scalar.approx(other, tol)/eq/ne→Bool`.**

---

## 4. Explicitly PARKED (non-goals — stay retarget-side, consume fungeom values)

Drawing this boundary is as important as the additions. fungeom does **not** grow these:

- **Smoothing/regularized derivatives** — Savitzky-Golay, local-polynomial, RMS-window
  (vs. the exact FD combinator A4, which *does* ship).
- **Robust/iterative fits** — RANSAC, SE(3) Fréchet mean / transform averaging (vs. exact
  SVD `fit_plane`/Kabsch, which *do* ship).
- **Statistical scoring** — quiet detection, noise/ChannelNoise calibration, chi-squared
  confidence fusion, mask cleaning.
- **Stateful filters** — contact **hysteresis** (two-threshold) — the single-threshold
  `BoolSignal` is the fungeom form.
- **Non-analytic surfaces** — heightmaps / terrain / scalar-field supports.
- **Estimation/orchestration** — alignment cross-correlation, DTW, `SyncPlan` networkx graph
  validation/shortest-path; support-state labels; contact plan.

---

## 5. Open scope-decision questions (human / fungeom session)

All LOCKED (retarget best judgement); A1–A6 in §1 cover the big ones, these are the rest:

1. **argmin/nearest ties (C5/C6/C9)** — LOCKED: empty → `Unresolvable`; tie → **deterministic
   first-in-roster-order** (a contact/decision layer needs an answer, not ambiguity).
2. **Region2 reach** — LOCKED: **include G2** (point-to-region signed distance — balance margin)
   **and G6** (`Face` clamped clearance). Support **convex + simple-polygon**; non-convex /
   Minkowski-sum deferred (N-tier, on demand).
3. **`BoolSignal` crossing** — LOCKED: **sub-sample exact** threshold crossing of the linear
   interpolant (matches the exact/decidable ethos), not hold-based sample-grid spans.
4. **`FrameSignal` / `PlaneSignal`** — LOCKED: **ship `PlaneSignal`** (S2/S3, needed);
   **defer `FrameSignal`** in favor of the T1 transport family until a definition needs a
   reusable moving frame.
5. **Median/diameter/spread/order** — LOCKED: **in-scope exact reductions**, but **N-tier**
   (build on demand; not blocking).
6. **Bundle-of-events / categorical signals** (M3 `InstantBundle`, M8 key-signal) — LOCKED:
   **deferred (N-tier)**; retarget handles "which corner first/lowest" numerically near-term.

---

## 6. Sequencing (which tiers unblock which retarget stage)

- **retarget S1** (substrate foundation) needs **P1–P3 (perf carrier + resolve_over)** and
  **T1 (transport)** to be viable; otherwise blocked only by adapter-internal work. Can start
  the non-perf parts now; the carrier lands the modeled-marker path cleanly.
- **retarget S2** (open patch algebra) needs the **B2** set — above all **G1 Region2**, the
  **G3 Plane↔2D bridge**, **G4 Point2Bundle**, **T2 lift**, **T8 PlaneSignal**, and (for
  cross-patch) **G5 affine Transform2**.
- **retarget S3** (contacts spine) needs the **B3** set — **G2/G6** (region distance + Face),
  **T3 BoolSignal**, **T4 derivative**, **T9 bundle-signal folds**, **C1/C5/C6** (BoolBundle,
  argmin/nearest→key).

*This is the complete forward inventory. `docs/region2-handoff.md` carries the G1/G2/G6 detail.
Build the B2/B3/P tiers; E/N are catalogued so the migration hits no surprise gaps, built on
demand. Editable path flows each landed item into retarget live.*
