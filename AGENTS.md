# AGENTS.md

This repository is a typed, extensible research toolkit for motion retargeting:
demonstration loading, synchronization, resampling, contact reasoning, and
future SMPL/GVHMR/video integrations.

The public model is **typed-first and enum-free**. Scene and demonstration
structure is authored with `TypedDict` schemas and compiled into dual-purpose
frozen dataclasses that are *both* the authoring objects and the bound runtime
query surface. There are no identifier enums (`MarkerId`, `SegmentId`,
`TrackId`, …). Authored string names are the identity; runtime keys are plain
string-based targets.

## Source of truth (for any agent)

`AGENTS.md` is the canonical, self-contained guide for this repo and is written
for **any** coding agent or human — not just one tool. `CLAUDE.md` is a symlink
to this file, so Claude Code reads exactly the same content. Cursor additionally
mirrors these conventions as always-applied rules under `.cursor/rules/*.mdc`;
those are a convenience for Cursor and are never the primary source. If anything
there ever disagrees with this file, **this file wins**, and you need not read
anything else to get started.

## Python baseline

Requires **Python 3.13+**. Use PEP 695 generics with defaults for public
generic types. The package ships a `py.typed` marker; keep it.

## Array layout and style

Arrays are time-major and validated at construction boundaries: `(T,)` for a
scalar/bool series, `(T, 3)` for one vector signal, `(T, N, 3)` for many,
`(T, 3, 3)` for rotations, `(T, 4)` for quaternions, `(T, N)` for per-entity
bool. In tests, assert with `np.testing.assert_array_equal` / `assert_allclose`.

Prefer `from __future__ import annotations`; frozen + slotted dataclasses for
immutable values (mutable lazy caches stay non-frozen); `Mapping` for read-only
inputs and `MappingProxyType` for stored views. Concrete `TypedDict` subclasses
project literal keys to declared types — that is what statically types the deep
chain. Iterating a *generic* `TypedDict` type var still yields broad values, so
cast internally where needed. Package with `uv` + hatchling; do not introduce
another packaging system.

## The headline guarantee: typed deep chain

Concrete `TypedDict` subclasses project literal keys to their declared field
types, so the full query chain is statically typed with no codegen:

```python
demo.tracks["mocap"]                                   # MocapTrack[MocapSubjects]
mocap.subjects["left_shoe"]                            # Subject[ShoeSegments]
subject.segments["shoe"]                               # Segment[ShoeMarkers, ShoePatches]
segment.markers["heel"].positions()                    # TimeVec3
segment.patches["sole"].points()                       # TimeVec3
```

Do not regress this. The mechanism is: `Demonstration[TracksT].tracks -> TracksT`
and `MocapTrack[SubjectsT].subjects -> SubjectsT`, where `TracksT`/`SubjectsT`
are the user's concrete `TypedDict`s. Do **not** reintroduce identifier enums,
`SegmentSpec[M, P]`, `SceneView`/`SegmentView`, handles, `TypedDemonstration`,
or codegen to "improve" typing.

## Autonomy and decision-making

You have real latitude here. Default to making progress over asking, keep the
change small and coherent, and leave the harness green. The intent is to give
you the freedom of a trusted contributor — without room for catastrophic
mistakes.

Decide on your own, then note the choice in your summary:

- naming, module/file layout within a layer, formatting, and docstrings;
- choosing among equivalent implementations;
- adding focused tests, fixtures, and internal helpers;
- refactors confined to the layer you are already editing;
- adding new concrete schemas, tracks, or query helpers that follow the patterns
  already established in this file.

Stop and confirm with a human first — these are expensive or hard to undo:

- changing the public API shape or the typed deep-chain mechanism;
- reintroducing anything under "Forbidden patterns", or reversing an
  architectural decision recorded in this file;
- broad cross-cutting refactors, mass renames, or deleting modules or tests;
- changing dependencies, packaging, the Python baseline, or lockfiles;
- weakening the harness to make it pass — skipping or `xfail`-ing tests,
  loosening mypy, or sprinkling `# type: ignore` to silence a real error (a
  targeted, commented ignore for a genuine language/stub limitation is fine);
- destructive git or filesystem actions (history rewrites, force pushes,
  deleting bags/data, other irreversible operations).

"Confirm first" means gather context and prepare the change, not freeze. When in
doubt, take the reversible path and say what you did.

## Required workflow

Before changing code:

1. Read this file — it is self-contained.
2. Identify the layer you are touching: core schema/dataclasses,
   targets/keys/state, IO/parsing, demonstration container, mocap track,
   resampling, alignment/sync, contact tracks, or examples/tests.
3. Make the smallest coherent change and add or update tests for it.
4. Run the narrow relevant tests, then the full harness below.

## Harness and definition of done

Run from the repository root; prefer `python3` and `uv run`. Python **3.13+** is
required — if `uv run` complains about the interpreter, `uv python pin 3.13`.

A change is done when all three pass:

```bash
python3 -m compileall -q src/retarget examples   # syntax / import sanity
uv run pytest -q                                  # full test suite
uv run mypy                                        # strict; must stay clean
```

`pytest` adds `src/` to the path via `pythonpath` in `pyproject.toml`, so no
`PYTHONPATH=src` is needed for tests. While iterating, run the narrowest
relevant test first, e.g.:

```bash
uv run pytest -q tests/test_demo_resampling.py
uv run pytest -q tests/test_demo_sync.py
uv run pytest -q tests/test_typed_deep_chain.py   # the assert_type deep chain
```

For ad-hoc snippets outside pytest, add `src/` yourself; for examples, use the
vendored layout:

```bash
PYTHONPATH=src python3 -c "from retarget.core import bind_scene"
cd examples/process_mocap_data && PYTHONPATH=../../src:. python3 new_api_example.py
```

`uv run mypy` is strict and currently green across all source files — keep it
that way, and keep the typed deep chain (`demo.tracks["mocap"].subjects[...]`)
clean. `ruff` is available (`uv run ruff check`) but is not part of the required
gate.

## Authoring schema (the only public scene model)

The package provides generic primitives:

- `TypedDict` bases: `Markers`, `Patches`, `Segments`, `Subjects`, and (demo
  layer) `Tracks`.
- frozen dataclasses: `Marker`, `Patch`, `Segment[MarkersT, PatchesT]`,
  `Subject[SegmentsT]`.

Project/user code declares concrete schemas and instances:

```python
class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker

class ShoePatches(Patches):
    sole: Patch
    toe_contact: Patch

class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]

class MocapSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]
```

`bind_scene(subjects)` path-binds the authored schema (so `*_target(...)` and
geometry inspection work) and returns the same `SubjectsT` type. Loading data
(`MocapTrack.from_unbagged(root, subjects)`) returns a `MocapTrack[SubjectsT]` whose
`.subjects` answers time-series queries.

`Marker`/`Patch`/`Segment`/`Subject` carry a private, non-init `_binding` that
links them to loaded data. It is `None` while authoring and excluded from
equality/repr, so the public constructors stay pure authoring
(`Marker(mocap_name=...)`). Mutable lazy caches (if any) stay non-frozen.

## Identity, targets, and runtime keys

Identity is the authored string name. Stable runtime keys are string-based
dataclasses:

```python
SegmentTarget(subject="left_shoe", segment="shoe")
MarkerTarget(subject="left_shoe", segment="shoe", marker="heel")
PatchTarget(subject="left_shoe", segment="shoe", patch="sole")
SegmentKey(subject="left_shoe", segment="shoe")   # SceneState pose key
```

Contact tracks key off `PatchTarget`. Scene pose state (`SceneState`) keys off
`SegmentKey`. Do not reintroduce `MarkerHandle`/`PatchHandle` or enum-typed
identity. `Segment.marker_target(...)`, `.patch_target(...)`,
`.segment_target()`, and `Marker.target` / `Patch.target` build these from the
binding.

## Demonstration containers

```python
class GroundEstimationTracks(Tracks):
    mocap: MocapTrack[GroundEstimationSubjects]
    contacts: ContactTrack

demo = Demonstration(GroundEstimationTracks(mocap=mocap, contacts=contacts))
demo.tracks["mocap"]      # statically typed
demo["mocap"]             # secondary, string-keyed Track access
demo.track_ids()          # ("mocap", "contacts")
clip = demo.slice_time(0.0, 1.0)   # DemonstrationView (tracks typed as Track)
```

`Demonstration[TracksT]` holds the typed `tracks` mapping and optional
`alignments`. `slice_time(...)` returns a `DemonstrationView` whose `.tracks`
values are sliced/view tracks typed through the common `Track` surface (honest,
broad). To keep the typed deep chain after slicing, slice the *track*
(`mocap.slice_time(...)` returns `MocapTrack[SubjectsT]`), not the demo.

Do not add `with_alignments`, `with_track`, `align`, or `sync` methods. Use the
free functions in `retarget.demo.sync`.

## Batch queries

Single-entity access is mapping-style; batch/`as_dict` are string methods on
`Segment`:

```python
segment.markers["heel"].positions()
segment.marker_positions("heel", "toe")              # (T, 2, 3)
segment.marker_positions("heel", "toe", as_dict=True)  # {"heel": (T,3), ...}
segment.patch_points("sole")                          # (T, 1, 3)
segment.patch_contacts("sole")                        # (T, 1) bool
```

No enum-typed query helpers. No underscore "private" query API as the primary
surface.

## Mocap resampling policy

`MocapTrack.resample_to(timestamps, output_timestamps=None, method=...)`:

- segment translations: linear interpolation (ignores `method`);
- segment rotations: discrete sampling via `method` (`NEAREST`/`PREVIOUS`);
- attached contacts: delegated to `ContactTrack.resample_to(..., method=...)`;
- raw marker frames: dropped on resampled output.

A single `method` governs every discrete quantity; there are no separate
`rotation_method`/`contact_method` parameters.

Do not synthesize raw marker observations. Do not linearly interpolate rotation
matrices without explicit SO(3) projection/tests.

## Contact tracks

`ContactTrack` stores boolean contact state and optional confidence arrays keyed
by `PatchTarget`. Resampling is discrete (`ResampleMethod.NEAREST`/`PREVIOUS`);
never interpolate contact booleans. `ContactTrack`/`ContactTrackView` share
query/resampling via `_ContactQueryMixin`.

## Alignment and sync

`EnergySignal` is a named scalar series; callers collapse vector data to scalar.
`TimelineTransform` uses `to_reference(...)` / `to_source(...)`; `then(other)`
applies `self` first. `TrackAlignment`, `SyncEdge`, `SyncPlan` are string-keyed.
`SyncPlan` is a connected graph rooted at `reference` and rejects empty/self/
duplicate/disconnected edges. `estimate_sync_and_resample_to_reference(...)` is
the one-shot free-function workflow (estimate, compose to root, slice, resample);
do not mutate `Demonstration`.

## Backend/manual loaders

Backend loaders (e.g. `examples/process_mocap_data/backend_specs/`) author the
*same* typed `Subjects`/`Tracks` schemas, but may derive geometry from real VSK
files via `calibrate_patch_transform(...)` and `read_marker_positions_from_vsk`.
They return typed `Demonstration[TracksT]`. They do not use enums or private
query helpers.

## Forbidden patterns

These were tried and deliberately removed. Do not reintroduce them or add new
variants:

- **identifier enums** — `MarkerId`, `SegmentId`, `PatchId`, `SubjectId`,
  `TrackId`, `NameId`, or enum-keyed containers like `Demonstration[TrackId]`.
  Identity is the authored string name. (Value enums like `RotationFormat`,
  `PoseFormat`, `MarkerRole`, `ResampleMethod`, `SemanticAxis` are fine.)
- **the spec/view/handle layer** — `SegmentSpec`, `SceneSpec`, `SubjectSpec`,
  `SceneView`, `SegmentView`, `MarkerHandle`, `PatchHandle`.
- **transitional demo names** — `TypedDemonstration`, `typed_tracks`,
  `get_track`, `_get_track`, `_generated_ids`, `GroundEstimationTrackId`,
  `._subject`, `._marker_positions`.
- **in-place demo mutation** — `with_alignments`, `with_track`, `align`, or
  `sync` methods on `Demonstration`; use the free functions in
  `retarget.demo.sync` instead.
- **typing crutches** — codegen, dependent-typing plugins, or extra mypy plugins
  added to "improve" typing. The deep chain types via plain concrete `TypedDict`
  projection; keep it that way.

## Tests

Every behavior change needs tests. Cover schema authoring/targets, observed vs
modeled marker positions, patch geometry, batch/`as_dict` queries, slicing
(including empty slices), mocap resampling, contact resampling, sync graph
validation + root composition + one-shot workflow, and the typed deep chain
(`tests/test_typed_deep_chain.py` uses `assert_type`).

Do not add fixtures or tests that reintroduce anything under "Forbidden
patterns" above.
