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

## Python baseline

Requires **Python 3.13+**. Use PEP 695 generics with defaults for public
generic types. The package ships a `py.typed` marker; keep it.

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

## Required workflow

Before changing code:

1. Read this file and all files in `.cursor/rules/`.
2. Identify the layer: core schema/dataclasses, targets/keys/state, IO/parsing,
   demonstration container, mocap track, resampling, alignment/sync, contact
   tracks, examples/tests.
3. Make the smallest coherent change and add/update tests for changed behavior.
4. Run the narrow relevant tests, then the full checks.

## Running commands

Run from the repository root. Prefer `python3` when invoking directly.

```bash
python3 -m compileall -q src/retarget examples
pytest -q                # or: uv run pytest -q
uv run mypy              # strict; the typed deep chain must stay clean
```

`pytest` adds `src/` via `pythonpath` in `pyproject.toml`.

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

`build_scene(subjects)` path-binds the authored schema (so `*_target(...)` and
geometry inspection work) and returns the same `SubjectsT` type. Loading data
(`load_mocap_track(root, subjects)`) returns a `MocapTrack[SubjectsT]` whose
`.subjects` answers time-series queries.

`Marker`/`Patch`/`Segment`/`Subject` carry a private, non-init `_binding` that
links them to loaded data. It is `None` while authoring and excluded from
equality/repr, so the public constructors stay pure authoring
(`Marker(vicon_name=...)`). Mutable lazy caches (if any) stay non-frozen.

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

demo = build_demonstration(GroundEstimationTracks(mocap=mocap, contacts=contacts))
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

`MocapTrack.resample_to(...)`:

- segment translations: linear interpolation;
- segment rotations: discrete nearest/previous sampling;
- attached contacts: delegated to `ContactTrack.resample_to(...)`;
- raw marker frames: dropped on resampled output.

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

## Tests

Every behavior change needs tests. Cover schema authoring/targets, observed vs
modeled marker positions, patch geometry, batch/`as_dict` queries, slicing
(including empty slices), mocap resampling, contact resampling, sync graph
validation + root composition + one-shot workflow, and the typed deep chain
(`tests/test_typed_deep_chain.py` uses `assert_type`).

Do not reintroduce enum/handle/spec/view fixtures or transitional names
(`TypedDemonstration`, `typed_tracks`, `get_track`, `GroundEstimationTrackId`,
`._subject`, `._marker_positions`, …) anywhere.
