# `Region2` rung — handoff spec (retarget → fungeom)

**Status:** requirements + proposed surface, authored from the **retarget** session for a
**fungeom** session to implement. This is the reverse of `docs/fungeom-backend-adapter.md`:
that one was fungeom telling retarget what existed; this one is retarget telling fungeom
what it needs next. The internal design is fungeom's call — this fixes the *contract*
retarget will consume, not the implementation.

> Read order for the fungeom session: this file → fungeom `README.md` (the combinator table
> + the rung-3 `Roster`/`RosterMap` precedent) → fungeom `CHECKLIST.md` (the exact procedure
> for adding a primitive) → retarget `docs/fungeom-substrate-migration.md` (why retarget
> needs this). retarget repo: `~/GitHub/retargeting_from_scratch`.

---

## 0. Why

retarget is making fungeom its modeling substrate. A retarget **patch** is *an oriented
surface + a bounded region* (e.g. a shoe sole, a board deck). The oriented surface is
already expressible (`Plane` + `Plane.frame(origin, tangent) → Transform`). The **bounded
region is not** — fungeom has only unbounded analytic geometry (planes/lines/rays/segments)
plus bounded **1-D** sets for time (`Interval`/`Coverage`). The missing piece is the **2-D
spatial sibling of `Interval`/`Coverage`**: a decidable, composable, bounded 2-D region.

retarget wants patch regions to be *arbitrarily definable* in intuitive code, e.g.

```python
def sole(seg):
    plane  = seg.markers["plane_rear","plane_inner","plane_outer"].fit_plane().offset(-0.004)
    region = (Region2.hull(seg.markers["toe_*"].in_frame(plane))   # convex hull of markers
                     .offset(-0.005)                               # shrink 5 mm
                     .difference(Region2.disc(0.01, at=heel_uv)))  # punch a hole
    return Patch.on(plane, region)
```

That requires a `Region2` rung.

---

## 1. Where it sits

fungeom already ships the full 2-D analytic family the rung builds on: `point2`, `line2`,
`segment2`, `direction2`, `frame2`, `ray2`. `Region2` is the **bounded-area** member of that
family — area is to `point2`/`segment2` what `Interval`/`Coverage` is to `Instant`/`Duration`.
It should follow the same shape as every fungeom primitive: **one class that you both
construct from (classmethods) and compose with (fluent methods)**, an immutable lazily-
evaluated resolver, `decide() → Resolvable | Unresolvable`, partiality first-class.

A patch's 3-D geometry is then `Plane` (oriented surface) + `Region2` (in `Plane.frame`),
the region lifted to 3-D by the plane frame. Whether fungeom also wants a first-class 3-D
`Face`/`OrientedRegion3` (plane + region as one object) is an open question (§5); the
minimum is `Region2` + the existing `Plane.frame` lift.

---

## 2. Required surface (the contract retarget will call)

All decidable; degenerate inputs are `Unresolvable`, never exceptions.

**Constructors** (classmethods, in a `frame2` / local 2-D coords):
- `Region2.rectangle(width, height, center=origin) → Region2`
- `Region2.disc(radius, center=origin) → Region2`
- `Region2.polygon(points: Sequence[Point2]) → Region2`  — simple polygon; self-intersecting → `Unresolvable`
- `Region2.hull(points: Point2Bundle | Sequence[Point2]) → Region2`  — convex hull; < 3 distinct present points → `Unresolvable`

**Combinators** (fluent; closed under the algebra):
- `union(other) · intersection(other) · difference(other)`
- `offset(distance) → Region2`  — grow (≥0) / erode (<0); erosion past extinction → empty
- `transformed_by(t: Transform2) → Region2`  — **note (G5):** for cross-patch / patch-on-deck
  composition the transform is an **oblique plane→plane projection, which is affine (shear),
  not rigid**. `Transform2` must therefore carry shear/non-uniform scale (or add `Affine2`),
  and fungeom should provide `Plane.projection_to(other: Plane) → Affine2`. Without this,
  intersecting two patches projected onto a moving deck is unbuildable.

**Queries:**
- `contains(p: Point2) → Bool`
- `signed_distance(p: Point2) → Scalar`  — **(G2, load-bearing)** inside positive / outside
  negative (pick a convention). This is the **balance-board / ZMP stability margin** — this
  project is literally a balance board, so CoM/CoP-to-support-polygon distance must be a
  first-class fungeom value, not a retarget numeric. Empty region → `Unresolvable`.
- `nearest_boundary_point(p: Point2) → Point2`  — (G2) the closest point on the boundary.
- `boundary() → ` an ordered loop (`Sequence[Segment2]` or a `Loop2`)
- `sample(spec) → Point2Bundle`  — **retarget needs boundary/corner samples** for footprint
  clearance; deterministic ordering; interior sampling optional
- `corners()` / `vertices() → Point2Bundle`  (polygonal regions)
- `area() → Scalar` · `centroid() → Point2` · `bounds() → (extent in the frame)`

**Decidability requirements (load-bearing for retarget):**
- Empty region: `area() → 0`, but `centroid()`/`sample()`/`boundary()` → `Unresolvable`.
- Degenerate (zero-width, collinear polygon, coincident hull points) → `Unresolvable` where
  the quantity is ill-defined.
- **Partiality must propagate**: if the region is built from occluded markers (an
  `Unresolvable` `Point2`), the region — and any contact query on it — is `Unresolvable`,
  not silently empty. This is the whole point; it must compose with `Bundle.present` /
  signal occlusion the way the rest of fungeom does.

---

## 3. What retarget does with it (so the contract fits the use)

1. **Authoring** — the open algebra above is the patch-definition language (`hull`, boolean
   ops, `offset`). This is the headline "define patches any way imaginable."
2. **Contact reasoning** — retarget samples the region's **boundary/corner points**, lifts
   them to 3-D via `Plane.frame`, and takes `Plane.signed_distance(...)` → `ScalarBundle.min`
   for footprint clearance. So `boundary()`/`sample()`/`corners()` returning well-ordered
   `Point2`s is essential (a predicate-only region would not serve this).
3. **Decidable contact** — an `Unresolvable` region ⇒ `Unresolvable` clearance ⇒ undecidable
   contact (not a guessed `False`). Already how retarget wants occlusion to flow.

retarget does the 3-D lift itself via the existing `Plane.frame(origin, tangent) → Transform`
applied to the region's `Point2`s — *unless* fungeom prefers to provide the embedded 3-D
form (§5), which it now should:

**`Face` / `OrientedRegion3` (G6, recommended).** A `Plane` + a `Region2` as one 3-D object,
with the clearance that real contact needs: `closest_point(p: Point3) → Point3` **clamped into
the region** (exactly as `Segment.project` clamps where `Line.project` does not), and
`clearance(p: Point3) → Scalar`. Today's plan computes footprint clearance against the
*infinite* plane (`Plane.signed_distance`), which is wrong when the foot is *beside*, not
above, the patch. `Face` is the honest bounded-patch clearance object and the natural home for
the Plane↔2-D bridge (`Plane.to_local`/`embed`, G3). `Patch.on(plane, region)` becomes a `Face`.

---

## 4. House-style constraints (fungeom's own rules)

- Python 3.13, strict PEP 695 generics; immutable, lazily-evaluated, decidable resolver;
  `decide()`/`resolve()`, `Resolvable`/`Unresolvable`; one class that IS the resolver
  (classmethod constructors + fluent combinators), `Region2.Value` value type.
- Reuse the existing 2-D family (`point2`/`segment2`/`frame2`/`direction2`); don't duplicate.
- 100 % coverage gate; ruff + strict mypy; follow `CHECKLIST.md`'s add-a-primitive procedure.
- **Pure, honest geometry only** — convex hull, polygon clipping, point-in-polygon are in scope
  (exact, opinion-free). An op that bakes a **hidden** modeling commitment (a fit objective, an
  inlier threshold, an iteration tolerance) stays out; admission is fungeom's own call now
  ([`functional_api/docs/substrate-membership.md`](../../functional_api/docs/substrate-membership.md)),
  same spirit as the `TimeWarp` "numerics deliberately out of scope" line.

---

## 5. Open questions for the fungeom session

1. **Boundary representation** — polygonal loops only, or analytic arcs for `disc` (affects
   exact `area`/`contains` vs. sampled approximation). retarget can live with polygonal
   sampling but exact is nicer.
2. **Boolean ops scope** — general simple-polygon clipping (Vatti/Weiler–Atherton) up front,
   or start with convex + half-plane intersections and grow? retarget's near-term patches are
   convex (hull/rect/disc minus a hole), so convex-first is acceptable if it lands sooner.
3. **`offset` joins** — rounded vs. miter; what's `Unresolvable` (self-overlap on large
   erosion).
4. **3-D embedding** — does fungeom want a first-class `Face`/`OrientedRegion3` (`Plane` +
   `Region2`) so `Patch.on(plane, region)` is a fungeom object, or is that retarget's
   composition over `Region2` + `Plane.frame`? (Minimum viable = the latter.)
5. **Frame coupling** — is `Region2` intrinsically in a `frame2` (inherits grounding /
   partiality), or frame-agnostic local coords with the patch supplying the frame?
6. **`sample()` contract** — boundary-only / interior-grid / both; deterministic corner
   ordering (retarget relies on stable corner identity for footprint sampling).

---

*This rung is the one cross-repo blocking dependency for retarget's Stage 2 (open patch
algebra). retarget Stage 1 (the internal marker/pose substrate swap) does not need it and
can proceed in parallel. Editable path means a landed `Region2` flows into retarget live.*
