# fungeom membership reframe — reconcile retarget with the new substrate rule (fungeom → retarget)

> **Audience:** an agent working in the retarget repo (`~/GitHub/retargeting_from_scratch`).
> **From:** the markovlib / fungeom propagation. The "governing rule" retarget originated has been
> reframed and canonicalized **fungeom-side**.
> **Status:** a **reconciliation sweep — nothing moves.** Every kernel retarget parks stays
> retarget-side; this retires the slogan *as the rule* and re-justifies *why*. No code changes; docs +
> one memory note.
> **Canonical rule (read first):**
> [`functional_api/docs/substrate-membership.md`](../../functional_api/docs/substrate-membership.md).
> **Read order:** that note → this file → the retarget docs under §"The sweep."

## What changed, and why it reached retarget

The rule retarget authored — *exact / closed-form / combinatorial ⇒ fungeom; statistical / iterative /
smoothing ⇒ parked, consumer-side* — was a good **geometry** rule. But fungeom is a **general
decidability substrate** ("geometry is instance #1, time #2, anything honestly decidable is the goal"),
and the slogan sorts on **kind of math** — a correlate that stops tracking the right thing the moment
geometry isn't fungeom's only instance (an exact inference scan like forward–backward is pure and
honest, *not* "statistical-and-therefore-out"). fungeom has **retired the slogan as the membership
rule** and restated it in the substrate's own terms.

## The rule now (two bright lines)

- **Admit** an op iff it is **referentially transparent** — a pure function of its resolver graph,
  every seed / initial-guess / iteration-budget **reified as an explicit input** — **and honestly
  resolvable** — success, failure, *and approximation character* surfaced through `decide()`, never a
  silent `NaN`.
- **Park** it iff it bakes a **hidden modeling commitment** — a prior, a kernel bandwidth, a stopping
  tolerance whose right value is domain *taste*, not a property of the inputs.
- **Exactness is now a resolution *grade*, not a gate** (the parked depth-B graded-`Resolvability`
  RFC) — so "approximate" is no longer automatically "out."

The bright line keeps the old rule's **anti-accretion spirit** (don't absorb consumer modeling
opinions; stay lean) while fencing on the property that actually protects the substrate.

## Why nothing retarget parked actually moves (re-justification, not re-placement)

Every genuinely-numeric retarget kernel lands in the same place, for a sharper reason:

| kernel | old reason | new reason — same verdict (retarget-side) |
|---|---|---|
| DTW / ICP / sync estimation | "statistical" | hidden objective + iterative search to implicit convergence — *estimation*, not a decidable value |
| smoothing (Savitzky–Golay, RMS) | "smoothing" | hidden window / order = modeling *taste* |
| RANSAC fits | "statistical" | hidden RNG **and** hidden inlier threshold |
| contact hysteresis | "stateful" | two-threshold opinion (the single-threshold `BoolSignal` is the fungeom form) |
| IK / contact-implicit solves | "iterative" | residency (large numeric domain) + opinion-bearing objective |

## And it *promotes* the items retarget was nervous about

| item | old status | new status |
|---|---|---|
| closed-form SO(3)-log angular velocity, FD derivatives | "brushes the numerics line" (the A4 worry) | **clearly admit** — closed-form, deterministic, opinion-free (already shipped in fungeom) |
| SVD plane-fit / Kabsch (exact, no RANSAC) | nervously in-scope | **clearly admit** — the least-squares optimum carries no tunable opinion |

So exact/closed-form geometry stays fungeom's, the genuinely-numeric estimators stay retarget's, and
the borderline "feels-numeric-but-is-exact" calls resolve cleanly to **admit**. Only the *justification*
changes.

## The sweep — retire the slogan AS THE RULE; defer to the canonical note

Reframe each: retire the slogan as *the rule*, keep its spirit via the bright line, link
`functional_api/docs/substrate-membership.md`, and restate parked items by **hidden opinion** or
**residency** rather than "kind of math."

- ✅ `docs/fungeom-needs-for-substrate.md` §"The governing rule" — **already reconciled** by the fungeom
  session (defers to the canonical note + states the two bright lines). Verify it reads right; align the
  rest to it. (Its A#/G#/T# inventory stays as-is — per-item residency calls, already correct.)
- ☐ `docs/fungeom-session-kickoff.md` — §"Governing rule for what belongs here."
- ☐ `docs/fungeom-integration-roadmap.md` — the tiering preamble + the "Per the governing rule…" lines
  (lines ~5, ~126). Re-tier by *residency*, not category.
- ☐ `docs/region2-handoff.md` — "Anything statistical/iterative is out of scope" (~line 132).
- ☐ `docs/fungeom-runtime-handoff.md` — the non-goals "governing rule: …" line (~line 266).
- ☐ `AGENTS.md` — the "genuinely-numeric kernels … stay parked" line (~line 225); reframe to the bright
  line (these stay parked for *hidden opinion / residency*, the conclusion unchanged).

## What NOT to do

- **Don't move any kernel.** Placement is unchanged — this is re-justification only.
- **Don't restate fungeom's rule in full** anywhere — link the canonical note (single source of truth);
  retarget docs should state retarget's *needs* and *residency calls*, and defer admission to fungeom.
- Don't disturb the inventories / per-item residency notes — they're already correct.

## Record it

Add a retarget memory note (the reframe + this doc as the pointer) so a future retarget session applies
the bright line and doesn't re-assert the retired slogan. The full rationale (the four-cuts diagnosis,
*partiality ⊇ uncertainty*, the moved-fence argument) lives in the canonical fungeom note — link, don't
duplicate.
