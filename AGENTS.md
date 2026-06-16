# AGENTS.md

This repository is a typed, extensible research toolkit for motion retargeting, demonstration loading, synchronization, resampling, contact reasoning, and future SMPL/GVHMR/video integrations.

Coding agents must preserve the architecture. Do not flatten specs, state, views, tracks, and IO into one blob just because Python permits crimes against taste.

## Required workflow

Before making changes:

1. Read this file.
2. Read all files in `.cursor/rules/`.
3. Identify the layer being changed:
   - core specs, handles, targets, transforms, or state;
   - IO/parsing;
   - scene/subject/segment views;
   - demonstration containers/tracks/views;
   - resampling;
   - alignment/sync;
   - contact tracks/detection;
   - mocap tracks;
   - examples/tests.
4. Make the smallest coherent change.
5. Add or update tests for changed behavior.
6. Run the narrow relevant tests while iterating, then the full checks before calling the task done.

If checks cannot be run, say exactly what was not run and why.

## Running commands

Run from the repository root unless noted otherwise.

On macOS/local machines, prefer `python3` when invoking the interpreter directly.

### Default verification

```bash
python3 -m compileall -q src/retarget
python3 -m compileall -q examples
pytest -q
```

Equivalent with uv:

```bash
uv run pytest -q
```

### Demo/resampling/sync checks

```bash
pytest -q \
  tests/test_demo_resampling.py \
  tests/test_demo_contact.py \
  tests/test_demo_mocap.py \
  tests/test_demo.py \
  tests/test_demo_container.py \
  tests/test_demo_sync.py
```

### Optional lint/type checks

```bash
uv run ruff check .
uv run mypy src/retarget
```

Only claim a command passed if it actually ran.

## Non-negotiable architecture principles

### Static specs are reusable definitions

`SegmentSpec[M, P]` is subject-independent. It defines segment-local marker vocabulary, patch vocabulary, static marker geometry, axis convention, patch calibrations, and built patch specs.

Do not add `SubjectId` to `SegmentSpec`, `MarkerHandle`, or `PatchHandle`.

### Runtime identity is subject-scoped

Segment IDs are subject-local. Runtime state must disambiguate concrete segment instances with:

```python
SegmentKey(subject, segment)
```

Do not key `SceneState` by bare `SegmentId`.

### Views combine static specs with runtime state

`SceneView` holds both `spec: SceneSpec` and `state: SceneState`.

`SubjectView` holds both `subject_spec: SubjectSpec` and `state: SceneState`.

Segment lookup must preserve both paths:

```python
scene.subject(subject_id).segment(SEGMENT_SPEC)  # typed, preserves generics
scene.subject(subject_id).segment(segment_id)    # ergonomic runtime lookup
```

Do not remove either path.

### Concrete dataclasses are for human-friendly authoring

Concrete project specs should use typed fields with domain names and generic traversal methods. Avoid concrete fields named `segment` because they shadow `SubjectSpec.segment(...)`.

Keep the defensive base-method lookup in `SubjectView.segment(...)` unless the full hierarchy is proven safe:

```python
SubjectSpec.segment(self.subject_spec, segment)
```


## Current architecture direction: typed schema authoring

The next core-spec refactor should move toward a TypedDict-based schema authoring layer that compiles into normalized runtime specs. The goal is to make the user-authored scene hierarchy statically legible without making raw nested dictionaries the runtime model.

Use this split:

```text
TypedDict schema declarations
    -> build_scene(...) / compile step
    -> SceneSpec / SubjectSpec / SegmentSpec
    -> SceneView / SubjectView / SegmentView
    -> SegmentTarget / MarkerTarget / PatchTarget
```

The package should provide generic primitives such as `Marker`, `Patch`, `Markers`, `Patches`, `Segment[MarkersT, PatchesT]`, `Subject[SegmentsT]`, `Subjects`, and `build_scene(...)`.

User/project code should define only the concrete hierarchy, for example `ShoeMarkers`, `ShoePatches`, `ShoeSegments`, and `MocapSubjects`, then instantiate it and compile it with `build_scene(...)`.

Do not make raw `TypedDict` values the long-term runtime API. Runtime code should still use normalized specs/views and stable targets for validation, iteration, serialization, contact-track keys, mocap state keys, and dynamic data loaded from files.

## Demonstration containers

`Demonstration[K]` is a generic container mapping typed track IDs to `Track` instances. It may carry alignments. It should not become a workflow object.

Keep these APIs:

```python
demo[track_id]
demo.get_track(track_id)
demo.slice_time(start, stop)
```

`slice_time(...)` returns `DemonstrationView[K]` and preserves the root source.

Do **not** add `with_alignments`, `with_track`, `align`, or `sync` methods unless explicitly requested. Prefer free functions in `retarget.demo.sync` for workflows.

## Track IDs

Tracks must be identified with typed string-enum IDs, not raw strings in public APIs.

Use a base like:

```python
class TrackId(NameId):
    """Base class for user-defined demonstration track identifiers."""
```

Concrete demo track IDs should look like:

```python
class GroundEstimationTrackId(TrackId):
    MOCAP = "mocap"
    CONTACT = "contact"
```

## Track/resampling protocol

`Track.resample_to(...)` means:

```python
track.resample_to(
    timestamps,              # sample positions in this track's native time basis
    output_timestamps=None,  # optional labels for returned track
)
```

For alignment-aware resampling, non-reference tracks sample at transformed source-time coordinates and label the output with reference timestamps.

`DemonstrationView.resample_to(reference)` preserves the reference track as-is. Non-reference tracks must implement `resample_to(...)`.

Generic resampling helpers belong in `retarget.demo.resampling`, not miscellaneous `utils/` sludge.

## Contact tracks

`ContactTrack` stores boolean contact state and optional confidence arrays keyed by `PatchTarget[Any]`.

Contact resampling is discrete and uses `ResampleMethod.NEAREST` or `ResampleMethod.PREVIOUS`. Do not interpolate contact booleans.

`ContactTrack` and `ContactTrackView` share query/resampling behavior through the private `_ContactQueryMixin`; concrete classes define visible contact/confidence arrays.

## Mocap tracks

Current mocap resampling policy:

- translations: linear interpolation;
- rotations: discrete nearest/previous sampling;
- attached contacts: delegated to `ContactTrack.resample_to(...)`;
- raw marker frames: intentionally dropped on resampled output.

Do not synthesize raw marker frames during resampling.

Do not linearly interpolate rotation matrices unless the code explicitly projects/re-normalizes and tests SO(3) behavior. Prefer a future quaternion/slerp implementation if smooth rotations become necessary.

## Alignment and sync

`EnergySignal` is a named scalar time series. Callers decide how to collapse vector/spatial data into scalar energy.

`TimelineTransform` direction names are:

```python
to_reference(...)
to_source(...)
```

Do not reintroduce `source_to_reference` or `reference_to_source` aliases.

`SyncPlan` is a connected graph rooted at `reference`, not a star-only plan. It should reject self edges, duplicate directed edges, duplicate undirected edges, empty edge lists, and disconnected graphs.

`estimate_sync_and_resample_to_reference(...)` is the one-shot free-function workflow: estimate sync, compose alignments to the root reference, slice, and resample onto reference time. Keep it a free function and do not mutate `Demonstration`.

## Examples

Examples should teach the generic demonstration pattern:

- define a project-specific `TrackId` enum;
- write project-specific loader functions;
- return `Demonstration[ProjectTrackId]`;
- retrieve tracks with `demo[track_id]` or `demo.get_track(track_id)`.

Do not create project-specific demo container classes unless explicitly requested.

## Tests

Any behavior change needs tests. For resampling behavior, cover timestamp validation, nearest/previous behavior, view behavior, `output_timestamps` relabeling, and source-time sampling versus output-time labeling.

For sync behavior, cover graph validation, pairwise alignment estimation, root-reference composition, reverse-edge inversion, missing tracks, and the one-shot sync-and-resample workflow.
