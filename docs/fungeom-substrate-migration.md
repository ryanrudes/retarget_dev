# fungeom-as-substrate — migration design (Stage 0 output, for review)

**Status:** design only. No core code changed. This is the reviewable artifact that
gates the migration; we proceed stage-by-stage *after* sign-off, and Stage 2 has its
own hard sign-off gate (a public-API-shape change). Authored from a Stage-0
capability-gap audit + a perf/ergonomics spike against the real fungeom at commit
`ebf2c15` and the current retarget tree.

> Read order: this file → `docs/fungeom-backend-adapter.md` (the thin adapter that
> already exists and is reused) → retarget `AGENTS.md` (forbidden patterns, the typed
> deep-chain guarantee) → fungeom `README.md` (the combinator table).

---

## 0. The goal (what changes, what stays)

Make fungeom the **modeling substrate** of retarget, so geometric definitions become
**open fungeom expressions** instead of a **closed strategy menu**. The user-level
authoring model is unchanged: typed `TypedDict` schemas (`Markers`/`Patches`/
`Segments`/`Subjects`), authored string names as identity, the typed deep chain
`demo.tracks[...].subjects[...].segments[...]`. What changes is how a geometric entity
is *defined*.

Today a patch can only be built from a fixed menu of strategy objects:

```python
# CLOSED — only the prebuilt PlaneResolver / NormalResolver / ExtentResolver exist
sole = Patch.planar(plane=plane_from("plane_rear", "plane_inner", "plane_outer"),
                    normal=axis_normal(offset=-0.004), extent=bounding_box("toe", "heel"))
```

After: a patch's geometry is **any fungeom expression**, authored as a callable over the
segment's geometry, evaluated at bind time:

```python
# OPEN — any Python + any fungeom op
def sole(seg):                      # seg.markers[...] are fungeom Point3 (segment frame)
    return seg.markers["plane_rear", "plane_inner", "plane_outer"].fit_plane().offset(-0.004)
Patch(region=sole)
```

`Patch.planar(...)` survives as **thin sugar that compiles to the same algebra**. The
internal substrate work (markers/poses as fungeom resolvers behind the binding) is the
*enabler* that makes `seg.markers[...]` real fungeom `Point3`.

**fungeom carries identity + geometry + time + decidability. It does NOT do numerics**
(estimation, filtering, robust fitting, derivatives). Those stay parked retarget-side and
*consume* fungeom values — exactly the seam the thin-adapter spec drew.

---

## 1. Decisive Stage-0 findings

### 1a. Capability matrix (audited against real fungeom source)

| Need | fungeom support | disposition |
| --- | --- | --- |
| Marker fixed in segment frame → `Point3` w/ grounding partiality | `Point3.at(x,y,z, frame=…)`, detached → `Unresolvable` | **use-fungeom** |
| Modeled world pos `R(t)·local + t(t)` (per instant) | `Point3.transformed_by(pose)` == `R@p+t`, byte-exact | **use-fungeom** |
| Observed cloud w/ occlusion mask | `Point3BundleSignal.from_frames(…, present=)` (already in adapter) | **use-fungeom** |
| `seg.markers['a','b','c']` subset | `Point3Bundle.where([...])` | **use-fungeom** |
| Plane through / fit of N markers | `Plane.through_points(a,b,c)`; `Point3Bundle.fit_plane()` (PCA, ≥3 present → else `Unresolvable`) | **use-fungeom** |
| Normal / tangential / origin / offset / patch-frame | `Plane.normal()/.origin()/.offset()/.project()/.facing()/.flipped()/.frame(origin,tangent)→Transform/.intersect()` | **use-fungeom** |
| Clearance (signed distance to support) | `Plane.signed_distance(point) → Scalar` | **use-fungeom** |
| Footprint min-clearance over corners | `ScalarBundle.min()` | **use-fungeom** |
| Contact predicate **at an instant** | `Scalar.lt/le → Bool`; occlusion → `Unresolvable` propagates natively | **use-fungeom** |
| Affine clock map (offset+rate, compose/inverse) | `TimeMap`/`AffineTimeMap` (≡ `TimelineTransform`) | **use-fungeom** |
| Discrete resample NEAREST/PREVIOUS | `Interpolation.nearest`/`.hold` | **use-fungeom** |
| Time window restrict + non-affine warp | `Signal.restrict(Interval\|Coverage)`, `TimeWarp` (bonus) | **use-fungeom** |
| **Bounded region / extent** (`RectangularRegion`, rect/disc/hull/clip + set ops) | **none — fungeom is all-infinite geometry** | **extend-fungeom: new `Region2` rung** → `docs/region2-handoff.md` |
| **Time-varying frame** (a segment frame over time) | none — `CoordinateFrame` is static; no `FrameSignal` | **extend-fungeom** (or per-instant) |
| **Over-time transport** (segment-frame `Point3` × pose signal → world `Point3Signal`) | none — `transformed_by` is per-instant only | **extend-fungeom** (adapter precomputes today) |
| **Temporal derivative** (velocities, speeds, angular speed) | **none anywhere** ("numerics out of scope") | **keep-parked-numeric** |
| **BoolSignal** (whole-timeline scalar→mask, hysteresis, discrete contact track) | none — Bool is per-instant only | **keep-parked-numeric** |
| Alignment *estimation* (xcorr), sync-graph orchestration | none (algorithms, not geometry) | **keep-parked**, emit/consume `TimeMap` |
| Robust fit (RANSAC), quiet/noise/confidence, heightmaps, contact plan | none (by design) | **keep-parked-numeric** |

### 1b. Perf/ergonomics verdict (T=5000 @ 100 Hz, N=8, K=500; vs numpy baseline)

- **Viable at real scale.** One demo's bind is ~0.16 s (pose path) / ~0.3 s (point path)
  **one-time**; cached queries are ~10–30 µs. `decide()` is **memoized on the immutable
  resolver** — resolve a signal once, reuse everywhere.
- **No over-time `PlaneSignal` exists.** `fit_plane`/`where`/`centroid` live only on the
  **static per-instant `Point3Bundle`** (`signal.at(t)`). So a patch callable **operates
  on a per-instant bundle**, and retarget lifts it over the K instants it needs. fungeom
  will not hand back a plane-over-time.
- **Dominant costs are Python object construction, not algorithms:** the `(T,N)`
  `Transform` grid (143 ms / 40 k wrappers) and per-point `Point3` grounding (272 ms).
  A bulk array constructor in fungeom would erase most of it (wishlist §6).
- Per-instant fit is ~31 µs incl. offset — **2.3× an honest per-instant eager baseline**,
  11× a fully-batched SVD. Only matters if a patch plane is lifted over *all* T (≈150 ms);
  over the realistic contact-candidate subset (K≈500) it's ~15 ms/patch.

**Recommended substrate level: per-instant callable, lifted by retarget over the K
sampled instants.** Store markers as `Point3BundleSignal` and pose as
`TransformBundleSignal`, resolve **once**, evaluate the patch callable at the instants
that matter (contact candidates, not all T). Keep T-vectorization where fungeom gives it
free: marker trajectories (`signal.key(k) → Point3Signal`), distances, displacements.

---

## 2. The substrate model

```
authored schema (unchanged)            bind time                         query / runtime
─────────────────────────────────────────────────────────────────────────────────────────
Subject/Segment/Marker/Patch   ──►  build fungeom graph behind     ──►  marker.positions() etc.
TypedDict + frozen dataclasses      the existing _binding seam          keep ()->(T,3) ndarray
                                    (mypy-invisible swap)               (eager-vectorized from graph)
```

- **Markers** → one `Point3BundleSignal` per segment (observed, occlusion-aware) and a
  segment-frame `Point3Bundle` of rest positions (modeled). Built once, resolved once.
- **Segment pose** → one `TransformBundleSignal` per subject (the existing adapter
  `pose_signal` is absorbed into the binding).
- **`SegmentGeometry` view** — the object a patch callable receives. `seg.markers["a","b"]`
  → `Point3Bundle.where([...])` at the evaluated instant; `seg.frame` → the segment frame;
  `seg.pose` → the `Transform`. This is *new public surface* (see §3 typing).
- **Bind-time grounding convention (required).** `fit_plane`/`centroid` world-anchor every
  member, so they are `Unresolvable` on a *detached* frame. Therefore the patch callable is
  evaluated with the **segment frame grounded at identity to world** (segment-local ==
  world for the fit); the resulting `Plane`/`Transform` is stored **segment-local** and
  transported per-frame by the pose at query time — exactly how `transform_segment_patch`
  works today. The redesign must encode this convention explicitly; no fungeom change.
- **A patch is an oriented surface + a bounded region.** The oriented surface is a fungeom
  `Plane` (`Plane.frame(origin, tangent) → Transform` is the patch's 2-D coordinate system);
  the region is a fungeom **`Region2`** (a new rung — bounded 2-D area, the spatial sibling
  of `Interval`/`Coverage`) expressed in that frame and lifted to 3-D by the plane frame.
  `Patch.on(plane, region)`. This replaces `RectangularRegion` + the closed `ExtentResolver`
  menu with an open, composable, decidable algebra (rect/disc/polygon/hull + union/
  intersection/difference/offset; `contains`/`boundary`/`sample`/`area`). The region rung is
  net-new fungeom work — see `docs/region2-handoff.md`; it is the one **cross-repo blocking
  dependency for Stage 2** (Stage 1 does not need it).

---

## 3. Authoring API + the typing decisions

```python
class ShoePatches(Patches):
    sole: Patch                     # schema unchanged

def sole(seg: SegmentGeometry) -> Patch:        # oriented surface + bounded region
    plane = seg.markers["plane_rear", "plane_inner", "plane_outer"].fit_plane().offset(-0.004)
    region = Region2.hull(seg.markers["toe_*"].in_frame(plane))   # any open composition
    return Patch.on(plane, region)

ShoePatches(sole=Patch(geometry=sole))          # callable; evaluated (grounded) at bind time
```

Two riskiest mypy-strict pitfalls (both must be designed up front):

1. **`SegmentGeometry.__getitem__` typing.** `seg.markers["a","b","c"] → Point3Bundle`
   variadic string-key access tends to widen to `Any`. Needs deliberate `__getitem__`
   overloads (single-key → `Point3`, tuple → `Point3Bundle`) and must not leak `Any` into
   the callable's inferred type.
2. **`Patch[RegionT]` generic repurposing → PUBLIC-API-SHAPE CHANGE → human sign-off
   gate.** Today `Patch[RegionT: ContactRegion | None]` narrows the *static region type*.
   If `region` holds `Callable[[SegmentGeometry], Plane | Region]`, the generic must be
   repurposed or dropped. Per `AGENTS.md` this is exactly a "changing the public API shape"
   item — **it does not proceed without your explicit OK.** `assert_type(segment.patches
   ["sole"], Patch)` stays green either way (it uses the bare default-parametrized `Patch`).
   Keep `region` a frozen field with `compare=False, repr=False` (callables are not
   equality-friendly) — matching today's `region` field.

`Patch.planar(...)`, `plane_from`, `axis_normal`, `bounding_box`, … survive as **thin
sugar** that builds the callable/expression. Existing authored scenes keep working.

---

## 4. Preserving the invariants

- **Typed deep chain — safe, and the cleanest part.** The runtime is attached through the
  non-init, `compare=False`, `repr=False` `_binding` (`core/schema/binding.py`), fully
  decoupled from the dataclass/TypedDict surface the `assert_type` chain projects through.
  Swapping `_SegmentRuntime`'s numpy arrays for a fungeom graph is **invisible to mypy and
  to equality**; `tests/test_typed_deep_chain.py` stays green untouched.
- **No enums / named identity — holds.** fungeom `Roster`/`RosterMap`/`Bundle` are
  `Hashable`-keyed; authored strings remain the identity. (Note: fungeom's `Segment` is a
  *line-segment* primitive — a naming collision with body "segment", cosmetic.)
- **Query-method contracts kept.** `marker.positions() → (T,3)`, `patch.points()`,
  `segment.poses()` keep their eager numpy signatures via a thin bridge inside the bodies:
  resolve the signal once over the track `Sampling`, `np.stack` the `(T,)` values; map
  `Unresolvable` → **NaN** (observed markers, matching today) / → **`ValueError`** (modeled,
  matching today). No caller or test changes.

---

## 5. What stays parked numeric (the clean seam)

These have **no fungeom analog by design** and remain retarget-side, refactored to
*consume* fungeom `Scalar`/`Point3`/`Plane` values rather than recompute geometry:

- **Temporal derivatives** — velocities, normal/tangential/angular speed, quiet-activity
  (Savitzky-Golay, finite difference). *Biggest gap; fungeom has no `d/dt`.*
- **Contact masking over time** — hysteresis, `clean_mask_by_time`, discrete contact
  `ContactTrack` storage/resample (no `BoolSignal`; fungeom's contact `Bool` is per-instant).
- **Estimation & orchestration** — alignment cross-correlation, `SyncPlan` networkx graph
  validation/shortest-path (emit/consume `TimeMap`).
- **Robust/statistical kernels** — RANSAC plane refine, quiet/noise/chi-squared confidence,
  heightmap supports, support-state labels, contact plan.

The geometric spine they sit on (clearance `Scalar`, footprint `min`, plane fit, the
contact `Bool` at an instant, occlusion `Unresolvable`) **does** move to fungeom.

---

## 6. fungeom additions needed (the complete forward inventory)

**The complete forward inventory is `docs/fungeom-needs-for-substrate.md`** — every net-new
fungeom primitive/combinator the migration needs (~40 items across geometry, bundles, signals,
time, perf), tiered by what blocks which stage, with the parked-numeric boundary and the
architecture decisions (generic `map`/`lift`, affine `Transform2`, sanctioned ndarray readback,
the perf batch carrier). Headlines:

- **`Region2` rung** (+ point-to-region distance + `Face` clamped clearance) — blocks Stage 2;
  detail in `docs/region2-handoff.md`.
- **B2 set** — Plane↔2-D bridge, `Point2Bundle`, affine `Transform2`, `PlaneSignal`, the
  `Signal.lift` keystone — blocks Stage 2.
- **B3 set** — `BoolSignal`, finite-difference derivative, bundle-signal folds, `BoolBundle`,
  `argmin`/`nearest`→key — blocks Stage 3.
- **Perf carrier + `resolve_over`** — gate Stage 1 viability.

These land in fungeom sessions; the editable path flows each into retarget live.

---

## 7. Staged plan (each stage its own green PR; deep chain preserved throughout)

Strangler-fig: the public surface never moves; we swap implementations underneath and
delete dead machinery only once its replacement is proven.

- **Stage 1 — substrate foundation.** Markers/poses as fungeom resolvers behind the
  binding (the mypy-invisible swap); the `SegmentGeometry` view; query-method bridge
  (Unresolvable→NaN/ValueError). Absorbs the existing adapter's `pose_signal`/
  `marker_cloud_signal` into the core. *No authoring-API change yet; pure internal swap.*
  Full fidelity/viability needs the **perf carrier + `resolve_over`** (P-tier) and **T1
  transport** from `docs/fungeom-needs-for-substrate.md`; the non-perf wiring can land first.
- **Stage 2 — open patch algebra.** `Patch(geometry=callable)` returning `Patch.on(plane,
  region)`; fungeom `Plane` (oriented surface) + `Region2` (bounded region); retire the
  `<Aspect>Resolver` *and* `ExtentResolver`/`RectangularRegion` menus, keep `Patch.planar`
  sugar. **⛓ Cross-repo dependency:** requires the fungeom `Region2` rung
  (`docs/region2-handoff.md`) to land first. **⚠ Sign-off gate:** the `Patch[RegionT]`
  generic repurposing + `SegmentGeometry.__getitem__` typing land here.
- **Stage 3 — contacts geometric spine.** Clearance `Scalar`, footprint `min`, per-instant
  contact `Bool`, occlusion `Unresolvable`; numeric heuristics (quiet/noise/hysteresis/
  plan) refactored to consume fungeom values.
- **Stage 4 (optional) — time/sync** onto `TimeMap`/`Sampling`/`Coverage`/`Interpolation`;
  estimators stay parked, emit `TimeMap`.
- **Stage 5 — cleanup + `AGENTS.md`.** Delete dead eager machinery and the interim
  `<Aspect>Resolver` layer; flip the "interim resolver / don't pre-build the formal
  Resolver" notes.

**Gates & risks:** Stage 2's public-API change is the one hard human sign-off. The biggest
*technical* risk (perf at scale) is **retired** — the spike shows it's viable with
resolve-once caching. The remaining unknowns are typing ergonomics of the
`SegmentGeometry` view (Stage 2) and how much velocity/contact-mask numerics want to stay
retarget-side (Stage 3) — both contained, neither blocking.

---

*Stage 0 is non-destructive and complete. Nothing in `src/` changed. Stage 1 begins only
on your go; Stage 2 pauses for explicit sign-off on the `Patch` generic.*
