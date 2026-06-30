# AGENTS.md

This repository is a typed, extensible research toolkit for motion retargeting:
demonstration loading, synchronization, resampling, contact reasoning, and
future SMPL/GVHMR/video integrations.

The public model is **typed-first and enum-free**. Scene and demonstration
structure is authored as frozen `@dataclass` schemas (typed attribute fields)
that are *both* the authoring objects and the bound runtime query surface, and
are accessed **by attribute** (`markers.heel`, never `markers["heel"]`). Each
scene element is a typed symbol referenced directly. There are no identifier
enums (`MarkerId`, `SegmentId`, `TrackId`, …) and no string subscript keys on the
authoring/query surface. Authored attribute names are the identity; runtime keys
are plain string-based targets derived from those names.

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
inputs and `MappingProxyType` for stored views. Concrete `@dataclass` schemas
project **attribute access** to their declared field types — that is what
statically types the deep chain. The binding layer and runtime walkers iterate
schemas generically via the reflection helpers in `core/schema/base.py`
(`_schema_items`/`_schema_values`/`_schema_fields`/`_schema_get`/`_rebuild_schema`),
not dict iteration or `cast`. Package with `uv` + hatchling; do not introduce
another packaging system.

## The headline guarantee: typed deep chain

Concrete `@dataclass` schemas project **attribute access** to their declared
field types, so the full query chain is statically typed with no codegen:

```python
demo.tracks.mocap                                      # MocapTrack[MocapSubjects]
mocap.subjects.left_shoe                               # Subject[ShoeSegments]
subject.segments.shoe                                  # Segment[ShoeMarkers, ShoePatches]
segment.markers.heel.positions()                       # TimeVec3
segment.patches.sole.points()                          # TimeVec3
```

Do not regress this. The mechanism is: `Demonstration[TracksT].tracks -> TracksT`
and `MocapTrack[SubjectsT].subjects -> SubjectsT`, where `TracksT`/`SubjectsT`
are the user's concrete `@dataclass` schemas (each a frozen+slotted subclass of
the internal `_Schema` root). A misspelled field is a hard mypy + runtime error;
a stray subscript `schema["x"]` is too (schemas are deliberately not
subscriptable). Do **not** reintroduce `TypedDict` schema bases, string subscript
access, identifier enums, `SegmentSpec[M, P]`, `SceneView`/`SegmentView`, handles,
`TypedDemonstration`, or codegen to "improve" typing. The `assert_type` proof
lives in `tests/test_typed_deep_chain.py`; because it sits outside the gated mypy
scope, check it explicitly with `uv run mypy --strict tests/test_typed_deep_chain.py`.

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
that way. The typed deep chain (`demo.tracks.mocap.subjects.left_shoe…`) is
guarded by `tests/test_typed_deep_chain.py`, which is *outside* the gated mypy
scope (`packages=["retarget"]`), so verify it explicitly:
`uv run mypy --strict tests/test_typed_deep_chain.py`. `ruff` is available
(`uv run ruff check`); keep `src/retarget` clean and add no new violations.

## Authoring schema (the only public scene model)

The package provides generic primitives:

- empty `@dataclass(frozen=True, slots=True)` schema bases (subclasses of the
  internal `_Schema` root): `Markers`, `Patches`, `Segments`, `Subjects`, and
  (demo layer) `Tracks`.
- frozen dataclasses: `Marker`, `Patch`, `Segment[MarkersT, PatchesT]`,
  `Subject[SegmentsT]`.

Project/user code declares concrete schemas as **`@dataclass(frozen=True,
slots=True)`** subclasses with typed attribute fields (the decorator is
required — that is what gives the schema its `__init__` and makes its fields the
typed deep chain), and instances:

```python
@dataclass(frozen=True, slots=True)
class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker

@dataclass(frozen=True, slots=True)
class ShoePatches(Patches):
    sole: Patch
    toe_contact: Patch

@dataclass(frozen=True, slots=True)
class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]

@dataclass(frozen=True, slots=True)
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

## Patch geometry (the fungeom substrate)

A patch is an **oriented contact surface + a bounded footprint** — a fungeom `Face` (an oriented
`Plane` + a `Region2`). Author it as **data** over the segment's marker symbols, passed directly: a
`Marker` is a fungeom `SupportsPoint3`, coercing to its `marker.rest` free variable (a `Point3.free`
identified by the marker), so a misspelled marker is a `NameError`, not a silent key:

```python
from fungeom import Face, Point3Bundle, Region2

# heel, toe, plane_rear, … are the segment's Marker symbols, passed straight in (no .rest threading)
plane = Point3Bundle.of([plane_rear, plane_inner, plane_outer]).fit_plane()
footprint = Region2.hull(Point3Bundle.of([heel, toe]).in_frame(plane)).offset(0.005)
ShoePatches(sole=Patch(label="sole", geometry=Face.on(plane, footprint)))
```

`Patch.geometry` accepts a fungeom `Face` (the data form above) **or** a callable
`(SegmentGeometry) -> Face` (the legacy form, e.g. when the surface is fixed in the segment frame).
Inside a callable, `seg.markers[name] -> Point3` / `seg.markers[tuple] -> Point3Bundle` is the
fungeom `SegmentGeometry` *adapter* — the one place a `[...]` subscript remains, and it is **not**
the schema. At bind time the binding resolves the `Face` to a segment-local one: for the data form
via `face.bind(env)` (substituting each `marker.rest` free with its segment-frame rest position),
for a callable by evaluating it. `patch.points()/normals()/frames()/boundary_points()` then
transport that per-frame by the segment pose via a fungeom `FaceSignal`; `patch.face()` returns the
bound `Face`. **fungeom (`ryanrudes/fungeom`) is retarget's geometry substrate** — geometry lives
there, not in retarget (`retarget.core.geometry` holds the `SegmentGeometry` view + the `FaceSignal`
carrier). fungeom's free variables (`Point3.free` + `Face.bind(env)`, 0.4.0) make the data form
possible; its `SupportsPoint3` coercion (0.6.0) is what lets the markers be passed without `.rest`.
The genuinely-numeric kernels (DTW/ICP, sync estimation, smoothing) stay parked
retarget-side and *consume* fungeom values. Migration design:
`docs/fungeom-substrate-migration.md`; free-variable design: `docs/fungeom-free-variables-spec.md`.

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
identity. `Segment.segment_target()` and the symbols' own `Marker.target` /
`Patch.target` build these from the binding — e.g. `segment.markers.heel.target`
is the `MarkerTarget`. (There are no `marker_target`/`patch_target` lookup
methods; reference the symbol directly.)

## Demonstration containers

```python
@dataclass(frozen=True, slots=True)
class GroundEstimationTracks(Tracks):
    mocap: MocapTrack[GroundEstimationSubjects]
    contacts: ContactTrack

demo = Demonstration(GroundEstimationTracks(mocap=mocap, contacts=contacts))
demo.tracks.mocap         # statically typed (the deep chain)
demo["mocap"]             # secondary, string-keyed Track access (broad Track type)
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

Single-entity access is attribute access; the batch/`as_dict` helpers on
`Segment` take a **`Sequence` of bound symbols** (not string names), mirroring
`np.stack([...])`:

```python
segment.markers.heel.positions()
segment.marker_positions([segment.markers.heel, segment.markers.toe])              # (T, 2, 3)
segment.marker_positions([segment.markers.heel, segment.markers.toe], as_dict=True)  # {"heel": (T,3), …}
segment.patch_points([segment.patches.sole])                          # (T, 1, 3)
segment.patch_contacts([segment.patches.sole])                        # (T, 1) bool
```

`as_dict` keys come from each symbol's own authored name (`m.target.marker`). No
string-keyed or enum-typed query helpers. No underscore "private" query API as the
primary surface.

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
*same* typed `@dataclass` `Subjects`/`Tracks` schemas, deriving patch geometry as
`geometry=` data (fungeom `Face`s over `marker.rest`) from marker rest positions
read from real VSK files via `read_marker_positions_from_vsk` (the subject
`body_model`). They return typed `Demonstration[TracksT]`. They do not use enums,
string subscript access, or private query helpers.

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
  added to "improve" typing. The deep chain types via plain concrete `@dataclass`
  attribute projection; keep it that way.
- **`TypedDict` schema bases or string subscript access** — schemas are
  `@dataclass(frozen=True, slots=True)` subclasses of `_Schema`, accessed by
  attribute (`markers.heel`). Do not reintroduce `TypedDict` schema bases,
  `schema["x"]` subscript, dict-style `.items()`/`.keys()`/`.values()` on a schema
  (use the `_schema_*` reflection helpers), or a transitional subscript bridge on
  `_Schema`. Identifier-string batch helpers (`marker_positions("a","b")`) are
  likewise gone — batch helpers take a `Sequence` of symbols.
- **the retired closed patch surface** — the `Patch[RegionT]` generic,
  `ContactRegion`/`RectangularRegion`/`PolygonalRegion`, `calibrate_patch_transform` /
  `PatchCalibration` / `Patch.planar`, and the `<Aspect>Resolver` menu (`PlaneResolver`/
  `NormalResolver`/`TangentialResolver`/`OriginResolver`/`ExtentResolver` + `plane_from`/
  `axis_normal`/`side`/`along_axis`/`min_area_rectangle`/`bounding_box`/`fixed`). Patches are
  authored as `geometry=` **data** — a fungeom `Face` over `marker.rest` free variables (or a
  callable returning one); geometry is fungeom's job, not a retarget reimplementation.

## Tests

Every behavior change needs tests. Cover schema authoring/targets, observed vs
modeled marker positions, patch geometry, batch/`as_dict` queries, slicing
(including empty slices), mocap resampling, contact resampling, sync graph
validation + root composition + one-shot workflow, and the typed deep chain
(`tests/test_typed_deep_chain.py` uses `assert_type`).

Do not add fixtures or tests that reintroduce anything under "Forbidden
patterns" above.
