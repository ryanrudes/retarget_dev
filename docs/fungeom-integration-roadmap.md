# Roadmap — the rest of a *full* fungeom integration

**Audience:** a future retarget session (and whoever plans the work). **From:** the session that
migrated authoring + runtime + the contact spine onto fungeom. This is the honest "what's left",
tiered by the governing rule (**exact-combinatorial geometry/time → fungeom; statistical/iterative
→ stays numeric**), with value / cost / risk, the cross-repo fungeom asks, and a recommended order.

## Where we are (done, on `master`)

Fungeom is the substrate for the **geometry** of a scene:

- **Authoring** — patches are open `geometry=` callables returning a fungeom `Face`. The closed
  resolver/calibration surface is retired.
- **Patch runtime** — `Patch.points()/normals()/frames()/boundary_points()` resolve through
  `FaceSignal.of(face, pose).frame()/boundary()`. `lower_face` + the hand-rolled frame are gone.
- **Contact clearance** — patch supports use bounded `FaceSignal.clearance` (`_FaceSupport` in
  `contacts/detect.py`), partiality- and rotation-correct.

What is **still numpy** is everything *around* the patch: the marker/pose query layer, temporal
derivatives, resampling, sync, and the retargeting transfer. Most of that is exact geometry/time
that *could* be fungeom; some of it correctly should not be. Below is the split.

---

## The remaining work, in recommended order

### R1 — Runtime unification: markers + poses as fungeom signals (un-orphan the adapter)

**Today:** only patches go through fungeom. `Marker.positions()/velocities()` and
`Segment.translations()/rotations()/poses()/velocities()` read numpy straight off
`_SegmentRuntime`. The adapter (`retarget.fungeom.signals`) already has `marker_cloud_signal`
(→ `Point3BundleSignal`) and `pose_signal` (→ `TransformBundleSignal`) built **for exactly this**,
but they have **0 production use-sites** — the adapter is still mostly orphaned (only
`point_bundle_signal` and `relabel` are wired in).

**Do:** give `_SegmentRuntime` one cached fungeom graph per segment — markers as a
`Point3BundleSignal`, pose as a `TransformSignal` — and have the query methods `resolve_over` it
(returning the same `(T, …)` arrays). The patch `FaceSignal` then shares the segment's pose signal
instead of each patch rebuilding it.

**Why:** (1) **occlusion partiality end-to-end** — an occluded marker becomes a `present=False`
mask that flows markers → pose → patch → contact, instead of a silent NaN re-derived at each layer;
(2) one coherent decidable graph; (3) **un-orphans the adapter** (makes `marker_cloud_signal` /
`pose_signal` load-bearing), resolving the "two parallel fungeom surfaces" smell; (4) the patch
runtime stops rebuilding the pose `TransformSignal` per query.

**Cost/risk:** medium. Behavior-preserving (same arrays out) but it touches the hot query layer, so
**perf must match numpy** — see the per-instant lesson below. Keep the numpy fast-path if a signal
can't match it.

**fungeom needs:** a **vectorized `Point3BundleSignal` carrier from a dense `(T, N, 3)` array** (the
analog of `TransformSignal.from_matrices`). The adapter's `point_bundle_signal` builds a per-instant
`Point3` grid via `from_frames` — fine for K≈5 footprints, too slow for N markers × T frames.
Verify `marker_cloud_signal`/`pose_signal` `resolve_over` perf at realistic T before wiring.

**Recommendation — REASSESSED (don't do it).** On investigating the runtime (and after fungeom
delivered the carriers in v0.2.3/v0.3.0, so this is *not* a perf block): the marker/pose query layer
is almost entirely **pass-through of stored numpy** — `segment.translations()` is literally
`return runtime.translations`; observed `positions()` returns the raw frames; `rotations()/poses()`
are numeric *format* conversions; only modeled markers do a trivial sub-ms einsum. So wrapping these
as signals and resolving back is a **round-trip with overhead and no output change**. The partiality
benefit is illusory: the marker→pose link happens in the *loader* (pose estimation is statistical →
parked), so a runtime marker signal can't feed a pose signal — it just re-encodes the existing NaN.
And `from_matrices` is ~0 ms, so there's nothing to cache for the patches either. **R1/R2 are
plumbing, not value — skip them.** (This supersedes the "do this next" I originally wrote here; the
geometry layer that genuinely benefits from fungeom — patches, contacts — is already migrated.)

### R2 — Exact temporal derivatives via fungeom signals

**Today:** `finite_difference_velocity` backs `Marker/Patch/Segment.velocities()` (numpy).

**Do:** once R1 lands, swap those to `Point3Signal.velocity()/.speed()` and
`TransformSignal.angular_velocity()/.angular_speed()` (all exist in fungeom). **Keep
`local_polynomial_derivative` numeric** — it is a Savitzky-Golay-style *smoothed* derivative used by
the statistical quiet/motion detector, not an exact op.

**Cost/risk:** low, once R1 provides the signals. **fungeom needs:** none (delivered).
**Recommendation:** fold into R1's tail.

### R4 — Resampling + sync-warp on the signal graph (the deferred OQ4)

**Today:** `MocapTrack.resample_to` (linear-interp translations, discrete-sample rotations/contacts)
and `demo/alignment.py`'s `TimelineTransform` are numpy. The sync **estimation**
(`estimate_sync*`, cross-correlation/DTW over an `EnergySignal`) is statistical.

**Do:** resample/reparameterize the fungeom signal graph (`Sampling` / `resample` /
`reparameterize`) instead of the arrays; keep the **estimation** numeric. This was deliberately
deferred (adapter OQ4: "kept retarget's `TimelineTransform`; fungeom `TimeMap`/`Coverage` not
adopted for pass 1").

**Cost/risk:** significant — resampling is load-bearing across the demo layer; do it only after R1
(needs the signal graph). **fungeom needs:** confirm `Sampling`/`reparameterize` cover discrete
(nearest/previous) resampling + the contact-bool case. **Recommendation:** after R1/R2; medium value.

### R5 — The retargeting transfer (`GeometricTransfer`) — the frontier

**Today:** `transfer.py` has `relabel` (the *identity* transfer — re-key a bundle through a
`RosterMap`) plus a `GeometricTransfer` **Protocol seam with no implementation** (adapter OQ5,
parked). The actual "where does a target joint/morphology go given the source's contacts and
geometry" does not exist yet.

**Why it matters:** this is the repo's *raison d'être* — retargeting — and the substrate now makes
its **constraints** expressible in fungeom (contact points must meet target surfaces, footprints
align, clearances hold). The geometric constraints are exact (fungeom); the **solve** (IK /
optimization) is iterative (numeric). So a fungeom-native transfer = constraints decided in fungeom,
solver parked numeric.

**Cost/risk:** large — this is a *new capability and a design problem*, not a migration. Needs a
target model (skeleton/morphology), a correspondence story beyond `relabel`, and a solver. The
`Resolvable`/`Unresolvable` machinery is the right backbone (an unsatisfiable transfer is
*decidably* so). **Recommendation:** a separate design effort; the highest-value but highest-cost
item. Spec it on its own before any code.

### C1 — Adapter hygiene (decide after R1 / R5)

If R1 lands, `marker_cloud_signal`/`pose_signal` become load-bearing — good. The still-orphaned
`roster_map`/`identity_map`/`marker_at`/`joint_at`/`resolvability`/`GeometricTransfer` are scaffolding
for R5; **keep them only if R5 is on the near roadmap, else delete** (don't carry tested-but-unused
API). Also fold the `retarget.fungeom` re-exports of `segment_geometry`/`SegmentGeometry`/`patch_face`
(which now live in `core.geometry`) into one place to end the two-surfaces overlap.

---

## The line that STAYS numeric (do **not** "integrate" these)

Per the governing rule, fungeom carries the *values*; these kernels stay retarget-side:

- **Support fitting** — RANSAC + SVD robust plane fit, percentile heightmaps (`supports.py`,
  `utils/geometry.fit_patch_frame`). Robust/statistical. (A *non-robust* `Point3Bundle.fit_plane` is
  already used for patch authoring; that's fine.)
- **Contact scoring** — the χ² fusion, hysteresis, `clean_mask_by_time`, noise calibration
  (`_scoring.py`, `_quiet.py`, `noise.py`).
- **Sync estimation** — cross-correlation / DTW over the `EnergySignal`.
- **Smoothing / pose repair** — `fill_pose_gaps`, poly-smoothed derivatives (`local_polynomial_derivative`).

R3 (contact motion channels) is the gray zone: the normal/tangential/angular *projection* is exact
(fungeom-able) but it is welded to the smoothed derivative + scoring. Low value, high entanglement —
**leave it** unless R1/R2 make it nearly free.

---

## Lessons to carry into every round (earned the hard way)

1. **fungeom accessors ship per-instant first.** `frame`/`boundary`/`clearance`/`signed_distance`
   were each ~20–2651 ms at T=5000 before vectorization. **Measure `resolve_over` perf at T≈5000
   before wiring any accessor into a hot path**; if it's per-instant, hand off a vectorization ask
   (repro + acceptance: "within a small constant of the ~8 ms `TransformSignal` carrier"). The
   3-round loop (path A, then two clearance rounds) worked cleanly.
2. **The `transformed_by` gauge-bug class is real.** Transporting a Face/region over time can
   silently drop **in-plane rotation** (it bit `boundary` in 0.2.1 and `clearance`/`contains` in
   0.2.2). **Always test a new over-time geometry op under a *spinning* pose, not just translation.**
3. **`clearance` is unsigned** — re-sign for contact (`perp + sqrt(max(0, bounded² − perp²))`).
4. **`uv sync --refresh`** after every fungeom publish — PyPI's simple-index *and* aggregate JSON
   CDN-cache stale; `--refresh-package` was not enough.
5. **Cross-repo via docs.** One handover doc per round in `docs/`, with a runnable repro + an
   acceptance bar, has been the reliable channel; the fungeom session replies in the same file.

## Suggested sequence

**Revised after investigating the runtime (2026-06-29).** **Skip R1/R2** — they're pass-through
round-trips (see R1's reassessment); the data layer gains nothing from fungeom, and the geometry
layer that does is already migrated. So the *substrate* integration is effectively complete. What
remains that genuinely uses fungeom:

- **R5 (the retargeting transfer) — the real frontier.** It's a *new capability*, not a migration:
  express the transfer's geometric constraints (contacts meet target surfaces, footprints align,
  clearances hold) in fungeom — an unsatisfiable retarget is then *decidably* `Unresolvable` — with
  the solve (IK/optimization) parked numeric. This needs retarget-side product decisions first
  (target model, correspondence beyond `relabel`, solver), so it deserves its own design kickoff.
- **R4 (decidable resampling) — optional/marginal.** A real transformation (not pass-through), so
  not pure plumbing, but the resampled *values* are unchanged; the only gain is the decidable /
  partiality framing. Do it only if that framing proves useful, after R5's direction is set.
- **C1 (adapter):** since R1 won't consume it and R5 is the only plausible future consumer, either
  keep the `marker_cloud_signal`/`pose_signal`/correspondence/transfer scaffolding *for* R5, or trim
  the orphaned bits now to cut debt. Decide when R5 is scoped.

**Bottom line: the geometry substrate is done; the next real step is the R5 design, not more
runtime plumbing.**
