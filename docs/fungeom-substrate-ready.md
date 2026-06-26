# fungeom substrate is READY — status & what to do next (fungeom → retarget)

**Status: the substrate is built, gated, adversarially reviewed, AND completeness-audited — on
`main`.** Authored from the **fungeom** session that shipped it, for the **retarget** session to
pick up. Every item the inventory (`docs/fungeom-needs-for-substrate.md`) listed as **B2 / B3** is
built, plus the relevant **P** item (P3) and the **A4** philosophy call.

> **Patch *runtime* now delivered too** (the `docs/fungeom-runtime-handoff.md` ask): the **moving
> patch** `FaceSignal`, the static/​signal transport family, `Face.frame`/`boundary`,
> `TransformBundleSignal.key`, and the vectorized `TransformSignal.from_matrices` batch carrier are
> all on `main` — see the **✅ DELIVERED** banner atop `docs/fungeom-runtime-handoff.md`. You can
> drop `lower_face` and read the runtime off `FaceSignal(F, P).<…>.resolve_over(...)`.

- fungeom commit: **`ccf5daf`** on `main` (`ryanrudes/fungeom`).
- **1319 tests + 10 runnable examples · 100 % line coverage · ruff clean · mypy --strict clean.**
- **Hardened since the first handoff:** a multi-agent adversarial review found and fixed **7 real
  correctness bugs** (see "What got fixed" below — a couple change behaviour you'll rely on), and a
  `/audit-primitives` sweep added **11 new ops** to `Region2` / `Face` / `Point2Bundle`. So the
  surface is both more correct and a bit larger than the first handoff described.
- retarget already depends on fungeom via the **editable path**, so all of this is **live now**
  — `git -C ~/GitHub/functional_api pull` (already on main) and re-resolve if needed.
- **New runtime dep:** fungeom now pulls in **`shapely`** (GEOS) — it backs the general
  `Region2` polygon booleans/offset. An editable reinstall (`uv pip install -e
  ~/GitHub/functional_api`) will bring it in; or `uv pip install shapely`.

> Read order for the retarget session: this file → `docs/fungeom-backend-adapter.md` (the adapter
> plan — now fully unblocked) → `docs/fungeom-needs-for-substrate.md` (your inventory, for the
> per-item specs). The fungeom-side spine for the patch algebra is `~/GitHub/functional_api/docs/
> regions.md`, and the full surface is its `README.md` combinator table.

---

## What changed vs. the original plans (read this first)

A few things landed **better than the handoff assumed** — adjust the adapter accordingly:

1. **`Region2` booleans & `offset` are GENERAL & TOTAL** (not convex-first). They go through
   GEOS via shapely, so `union` / `intersection` / `difference` / `offset` work on **arbitrary
   simple polygons, holes, multipolygons** — there is **no** partial-overlap / non-convex /
   holey `Unresolvable` anymore. The only partiality is propagation (an `Unresolvable` operand).
   So you can compose patch booleans freely; no convex pre-checks needed.
2. **A4 finite-difference derivatives are IN scope and built** (`velocity`/`speed`/
   `angular_velocity`/`derivative`). Exact FD on the sample grid ships; smoothing derivatives
   (Savitzky–Golay) stay parked retarget-side, as planned.
3. **No `BoolBundleSignal` exists or is needed.** "Any corner in contact" is
   `clearances.min().le(0)` and "all corners" is `clearances.max().le(0)` — the scalar fold
   composed with the `BoolSignal` threshold. Use that.
4. **`argmin`/`argmax`/`nearest_to` return a singleton `Roster`** (not a bare key) — resolver-
   shaped, so empty → `Unresolvable`; the bare key is `.resolve().keys[0]`, and it composes with
   `cloud.where(...)`.
5. **`resolve_over` is the sanctioned vectorized exit** (P3) — your `marker.positions() → (T,3)`
   contract maps onto it. It resolves *eagerly* (raises `UnresolvableError` off-support).

---

## What got fixed in the adversarial review (behaviour you rely on)

The substrate passed an adversarial correctness + test-honesty review; 7 real bugs were fixed.
The ones that change behaviour your adapter will hit:

- **`BoolSignal.at(t)` is now exact at threshold-touch instants.** It used to report a strict
  `lt`/`gt` as `True` exactly *on* the threshold (and a predicate and its negation could both be
  `True` there). Now `at()` is the authoritative pointwise truth (a strict `lt` is `False` on the
  threshold, `le` is `True`), and `~`/`&`/`|` stay mutually consistent. Matters at the exact
  touchdown instant of a contact.
- **A lift over overlapping-but-sample-disjoint supports is now `Unresolvable`, not a crash.**
  `(a + b)` / `transformed_by` / `lift` where the operands' supports overlap but neither has a
  sample in the overlap used to build a 0-sample signal that raised `IndexError` on read; it now
  returns `Unresolvable` (the no-exceptions-for-partiality tenet holds). Reachable via `restrict`
  + any lift.
- **Windowed scalar reductions refuse non-linear kernels.** `min_over`/`integral_over`/… are exact
  only for the piecewise-linear interpolant, so a `hold`/`nearest` signal now returns
  `Unresolvable` instead of silently reducing wrong. (Default `linear` is unaffected.)
- **`ScalarBundleSignal` folds & `fit_plane` propagate the source kernel/boundary.** A `hold`/
  `nearest` cloud signal's `.min()`/`fit_plane()` now reads the same way the source does, and no
  longer silently shrinks the domain under a hold boundary. (Again, default `linear` unaffected.)
- **Non-uniform finite-difference derivatives are now second-order correct** (proper non-uniform
  central difference) — relevant if your sample times are irregular (dropped frames). Affects
  `velocity`/`angular_velocity`.
- **`angular_velocity` is confirmed world-frame** (`R_b·R_aᵀ`, spatial), verified by a
  varying-axis discriminating test. (A briefly-suspected body-frame bug was a false alarm.)
- A clockwise-wound `Region2Value` no longer silently converts to empty through the shapely bridge
  (internal robustness; unreachable via the public constructors).

## New ops added by the completeness audit

- **`Region2`:** `perimeter()`→`Scalar`; `closest_point(p)`→`Point2` (clamp a point *into* the
  region — interior query unchanged; the 2-D analog of `Face.closest_point`); `intersects(other)` /
  `contains_region(other)`→`Bool` (region-region predicates, boundary contact counts — handy for
  "does this support patch overlap / contain that one"); `symmetric_difference(other)`→`Region2`.
- **`Face`:** `contains(point)`→`Bool` — **footprint membership**, i.e. *is the foot / CoM over the
  patch* (the support-polygon test, normal offset ignored; total → `False` for an empty patch).
  This is the clean predicate for balance/support reasoning, paired with `clearance` for the margin.
- **`Point2Bundle`:** now at query parity with `Point3Bundle` — `map_scalar`/`map_point`,
  `distances_to(p)`→`ScalarBundle`, `closest_point_to(p)`→`Point2`, `nearest_to(p)`→`Roster`.
- **Signal facades (a later sweep):** `BoolSignal.last_true()`→`Instant` (contact **release** — the
  companion to `first_true`/touchdown; use the pair for contact onset/offset times),
  `TransformSignal.velocity()`→`Vec3Signal` (linear velocity — the linear half of the twist,
  paired with `angular_velocity`), and `Point3BundleSignal.centroid()`→`Point3Signal` (the cloud's
  **CoM track** — drop-in for a marker-set centre over time).

## Worked examples you can read

Two runnable scripts now demonstrate the substrate surface end-to-end (in the fungeom repo):
- **`examples/09_regions_and_patches.py`** — the 2D region algebra (`hull`/`offset`/`difference`,
  the positive-inside balance margin, predicates) and a `Face` (the bounded patch + the correct
  `Point3Bundle.in_frame(plane)` → `Region2.hull` chart pattern). Mirrors your patch-definition step.
- **`examples/10_contact_over_time.py`** — the **contact spine** exactly as your adapter will use it:
  `fit_plane` → `signed_distance(foot cloud)` → `min` → `le(0)` → `BoolSignal`, with
  `when_true`/`first_true`/`last_true` for the contact interval, touchdown & release. Copy this shape.

---

## The surface you now have (by inventory ID)

All signatures are exact as of `ccf5daf`. Import facades from `fungeom`, value types from
`fungeom.values`.

### Static geometry & patch algebra (B2/B3)
- **G1 `Region2`** — `Region2.rectangle(w, h, center=(0,0))` / `.disc(radius, center=(0,0),
  segments=64)` / `.polygon(points)` / `.hull(points | Point2Bundle)` / `.empty`;
  `contains(Point2)→Bool`, `area()→Scalar`, `centroid()→Point2`, `vertices()`/`bounds()`→
  `Point2Bundle`, `sample(N)→Point2Bundle`, `corners()→Point2Bundle`,
  `offset(distance)→Region2`, `union`/`intersection`/`difference(other)→Region2`,
  `transformed_by(Transform2)`.
- **G2** — `Region2.signed_distance(Point2)→Scalar` (**positive inside** — the balance-board/ZMP
  margin), `Region2.nearest_boundary_point(Point2)→Point2`.
- **G3 bridge** — `Plane.to_local(Point3)→Point2`, `Plane.embed(Point2)→Point3`,
  `Point3Bundle.in_frame(plane)→Point2Bundle`.
- **G4** — `Point2Bundle` (`of`/`from_array((N,2))`/`from_map`; `at`→`Point2`; `where`/`relabel`/
  `centroid`/`transformed_by(Transform2)`/`distance_to`→`ScalarBundle`/`bounds`).
- **G6 `Face`** — `Face.on(plane, region)`; `plane()`/`region()`; `closest_point(Point3)→Point3`
  (clamped into the region), `clearance(Point3)→Scalar` (the honest bounded-patch clearance when
  the foot is *beside*, not above).
- **G7** — `Plane.signed_distance`/`project`/`contains` broadcast over a `Point3Bundle`
  (→ `ScalarBundle`/`Point3Bundle`/`BoolBundle`).
- **G8–G13** — `Transform.aligning(a:Direction3, b:Direction3)`,
  `Transform.from_axes(x:Direction3, y:Direction3, origin:Vec3=(0,0,0))`,
  `Transform.look_at(eye:Vec3, target:Vec3, up:Direction3)` (note: positional inputs are
  **`Vec3`** not `Point3` — `Transform` sits below `Point3` in the layering), `Vec3.scalar_triple
  (b, c)→Scalar`, `Direction3.signed_angle_to(other, *, about:Direction3)→Scalar`.

### Bundle queries (B3)
- **C1 `BoolBundle`** — `and_`/`or_`/`not_` (`& | ~`), `any()`/`all()→Bool`.
- **C2** — `Bundle.presence_mask()→BoolBundle`, `all_present()`/`any_present()→Bool`.
- **C3/C4** — `Point3Bundle.map_scalar`/`map_point`/`map_vec3(fn)`, `distances_to(p)→ScalarBundle`.
- **C5/C6** — `ScalarBundle.argmin()`/`argmax()→Roster`; `Point3Bundle.nearest_to(p)→Roster`,
  `closest_point_to(p)→Point3`. `Bundle.where(roster | keys)` accepts a `Roster`.
- **C7** — `Point3Bundle.bounds()→Point3Bundle` ({`min`,`max`}); same on `Point2Bundle`.
- **Per-joint transfer** — `Point3Bundle.transformed_by(TransformBundle)` (key-aligned, each
  marker by its own joint's pose) — the modeled-marker static op.

### Over-time signals (B3/B/S1-core)
- **T1 transport** — `Point3Signal.transformed_by(TransformSignal)`, `Vec3Signal.transformed_by`,
  `Direction3Signal.rotated_by`, `Point3BundleSignal.transformed_by(TransformSignal |
  **TransformBundleSignal**)` (per-joint). *This removes your numeric marker-precompute — a marker
  fixed in a moving body frame becomes a world trajectory directly.*
- **T2 keystone** — `ScalarSignal.lift([sigs], combine)` / `Vec3Signal.lift(…)` (+ `Point3`/
  `Direction3`/`Transform`), and the unary `signal.map(f)`. `combine` gets each source's
  value-at-`t` resolver positionally; partiality flows.
- **T3 `BoolSignal`** — `ScalarSignal.lt`/`le`/`gt`/`ge(threshold)→BoolSignal`; `at(t)→Bool`
  (**Unresolvable in a gap** — keeps three-valued contact honest), `& | ~` (strict),
  `when_true()`/`when_false()→Coverage`, `first_true()→Instant`, `support()→Coverage`. Crossings
  are exact sub-sample (the linear interpolant), per your locked Q3.
- **T4 derivatives** — `ScalarSignal.derivative`, `Vec3Signal.derivative`, `Vec3Signal.norm→
  ScalarSignal`, `Point3Signal.velocity()→Vec3Signal`/`speed()→ScalarSignal`,
  `TransformSignal.angular_velocity()→Vec3Signal` (world-frame SO(3) log)/`angular_speed()`.
- **T5 reductions** — `ScalarSignal.min_over`/`max_over`/`mean_over`/`integral_over(window)→Scalar`,
  `argmin_over`/`argmax_over(window)→Instant`. `window` is an `Interval` **or** (gappy) `Coverage`.
- **T6** — `ScalarSignal.constant(value, over:Interval)`, `offset(c)`, `scale(c)`.
- **T8 `PlaneSignal`** — `from_samples`; `at→Plane`; `normal()→Direction3Signal`,
  `origin()→Point3Signal`, `signed_distance(Point3Signal)→ScalarSignal` (or over a
  `Point3BundleSignal` → `ScalarBundleSignal`); **`Point3BundleSignal.fit_plane()→PlaneSignal`**
  (batched per-frame SVD — the moving patch surface).
- **T9 `ScalarBundleSignal`** — `from_frames((T,N), keys, present)`; `at→ScalarBundle`; per-instant
  folds `min`/`max`/`mean`/`sum`/`count→ScalarSignal`.
- **P3 readback** — `Signal.resolve_over(Sampling)→ndarray` on all five plain signals
  (`(T,)`/`(T,3)`/`(T,4,4)`); `Point3BundleSignal`/`ScalarBundleSignal.resolve_over(Sampling)→
  ((T,N,·), (T,N) present mask)` (occluded cell = `nan`).

---

## Two flows that already run end-to-end

**Patch definition** (your headline):
```python
region = (Region2.hull(seg.markers["toe_*"].in_frame(plane))   # markers flattened into the plane
                 .offset(-0.005)                                # erode 5 mm
                 .difference(Region2.disc(0.01, center=heel_uv)))  # punch a hole
patch = Face.on(plane, region)
```

**Contact spine** (clearance → contact intervals), all lazy & decidable:
```python
clearances = ground_bundle_signal.fit_plane().signed_distance(foot_bundle_signal)  # ScalarBundleSignal
contact     = clearances.min().le(0.0).when_true()   # Coverage of contact intervals (sub-sample exact)
#  any corner in contact = clearances.min().le(0);  all corners = clearances.max().le(0)
```

---

## What is NOT built (so you don't wait on it)

- **P1/P2 — vectorized `RigidTransform` batch carrier** (perf refactor of `TransformBundleSignal`
  storage to `(T,N,4,4)`). Deferred **by design** — it gates Stage-1 **performance**, not
  correctness, and is premature without profiling. Use the current object-backed
  `TransformBundleSignal` + `resolve_over` for now; flag it if 40 k-wrapper build time actually
  bites, and fungeom will land the carrier in a focused session.
- **On-demand trivia** (build when a consumer appears, minutes each): `Vec3BundleSignal` /
  `Direction3BundleSignal`, `LineSignal` (the `Line` analog of `PlaneSignal`), analytic-arc discs.

---

## Next steps for the retarget session

1. **Build the adapter** per `docs/fungeom-backend-adapter.md` — it is now **fully unblocked**;
   every primitive/combinator it referenced exists. Keep it thin and typed (names/arrays ↔
   fungeom resolvers); no geometry logic in the adapter.
2. Map your data: `(T,N,3)` markers → `Point3BundleSignal.from_frames`; `(T,N,3,3)`/`(T,N,4)`
   joint rotations → `TransformBundleSignal.from_frames` (wrap each pose via
   `Transform.known(RigidTransform.from_rotation(Rotation.from_matrix(R), t))`); marker↔joint →
   `RosterMap`; the modeled-marker world path → `local_markers.transformed_by(joint_pose_signal)`.
3. For the vectorized exits your `(T,3)` / `(T,N,3)` contracts need, use `resolve_over(sampling)`.
4. Keep the parked numerics parked (DTW/ICP/RANSAC/SE(3) Fréchet mean/hysteresis) — fungeom
   *consumes* their outputs (a `TimeMap`/`TimeWarp`/`RosterMap`), it does not fit them.

*If anything in the fungeom surface is missing or you'd want it shaped differently, note it back —
fungeom owns its side and will add rungs on demand. fungeom commit `ccf5daf`, on `main`.*
