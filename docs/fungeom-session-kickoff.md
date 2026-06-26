# Kickoff prompt for a fungeom session

Paste the block below into a fresh Claude Code session launched in `~/GitHub/functional_api`
(the fungeom repo). It orients the session to the substrate work and points it at the handoff
docs in the retarget tree.

---

We're making **fungeom the modeling substrate for `retarget`** (the motion-retargeting repo at
`~/GitHub/retargeting_from_scratch`, which already depends on fungeom via an editable path — so
anything you ship here flows into it live). retarget ran a full forward analysis of every fungeom
primitive/combinator it needs and wrote handoff docs. **Read these first** (they live in retarget's
tree):

1. `~/GitHub/retargeting_from_scratch/docs/fungeom-needs-for-substrate.md` — the master inventory:
   ~40 net-new items tiered B2/B3/P/E/N, the **locked architecture decisions (§1)** and **locked
   scope answers (§5)** (build to them; don't re-litigate), the per-item specs with signatures +
   decidability (§3), and the **PARKED list (§4) — do NOT build those; they stay numeric in retarget.**
2. `~/GitHub/retargeting_from_scratch/docs/region2-handoff.md` — the deep spec for the biggest item,
   the `Region2` rung (+ point-to-region distance, a `Face` clamped-clearance object, affine projection).

Then follow **this repo's own AGENTS.md / README / CHECKLIST**. Every addition is a normal fungeom
rung: immutable lazily-evaluated **decidable resolver**, classmethod constructors + fluent combinators
(one class that IS the resolver), partiality first-class (`decide()` → `Resolvable`/`Unresolvable`),
100% coverage + ruff + strict mypy, with the **rung-3 (`Roster`/`RosterMap`) commit as the template**.

**Governing rule for what belongs here:** exact / closed-form / combinatorial → fungeom; statistical
/ iterative / smoothing → stays parked in retarget. §4's PARKED list is binding (no RANSAC, smoothing,
hysteresis, heightmaps, sync estimation, chi-squared, etc.).

**Start with the geometric foundation for retarget's open patch algebra — a coherent, self-contained
first rung that composes cleanly on the existing 2-D family (`point2`/`segment2`/`frame2`) and touches
no existing code:**
- **`Region2`** (G1) — bounded 2-D region/face: `rectangle`/`disc`/`polygon`/`hull`; `union`/
  `intersection`/`difference`/`offset`/`transformed_by`; `contains`/`boundary`/`sample`/`corners`/
  `area`/`centroid`/`bounds`. Convex + simple-polygon (non-convex/Minkowski deferred).
- the **`Plane`↔2-D bridge** (G3: `Plane.to_local(Point3)→Point2`, `Plane.embed(Point2)→Point3`) and
  **`Point2Bundle`** (G4) it builds on.
- the **region extensions**: `Region2.signed_distance(Point2)→Scalar` + `nearest_boundary_point` (G2 —
  this is the balance-board stability margin, load-bearing), and a **`Face`** = `Plane` + `Region2`
  with `closest_point(Point3)` clamped into the region and `clearance(Point3)→Scalar` (G6).

That chunk is the B2/B3 long pole. **A second, independent session** can take the **perf foundation**
that gates retarget's Stage 1 viability: a vectorized `RigidTransform` batch carrier so
`TransformBundleSignal` is backed by `(T,N,4,4)` arrays instead of ~40k Python `Transform` objects
(P1/P2), and a sanctioned `Signal.resolve_over(Sampling)→ndarray` (+ present mask) readback (P3).

**One call is yours to make** (it brushes your "numerics out of scope" line): **A4 — a
finite-difference derivative combinator** (`Point3Signal.velocity()/speed()`,
`TransformSignal.angular_velocity()` via closed-form SO(3) log). Our read: exact arithmetic on the
sample grid → in scope. If you judge it numerics, we keep velocities parked in retarget — clean either
way.

Work tier by tier: **B2 blocks retarget's open patch algebra, B3 blocks contacts, P gates Stage-1
perf; E/N are catalogued for build-on-demand.** Ship complete, tested rungs; the editable path carries
each into retarget immediately. If anything in the spec is ambiguous or you'd design it differently,
note it back — retarget owns the consuming side and can adjust.
