# Agent plan: TypedDict schema authoring refactor

You are implementing the new scene schema authoring model. Do not redesign the demo layer, sync layer, contact resampling, or mocap resampling while doing this. Apparently restraint is now a feature.

## Goal

Provide a user-friendly, statically legible schema authoring surface based on `TypedDict`, while preserving normalized runtime specs/views/targets for actual computation.

The intended architecture is:

```text
TypedDict schema declarations
    -> build_scene(...) / compile step
    -> SceneSpec / SubjectSpec / SegmentSpec
    -> SceneView / SubjectView / SegmentView
    -> SegmentTarget / MarkerTarget / PatchTarget
```

## Package-provided API

Add or adapt package-provided primitives roughly equivalent to:

```python
class Markers(TypedDict): ...
class Patches(TypedDict): ...
class Segments(TypedDict): ...
class Subjects(TypedDict): ...

@dataclass(frozen=True, slots=True)
class Marker:
    vicon_name: str

@dataclass(frozen=True, slots=True)
class Patch:
    label: str
    frame: str | None = None

@dataclass(frozen=True, slots=True)
class Segment[MarkersT: Markers, PatchesT: Patches]:
    markers: MarkersT
    patches: PatchesT

@dataclass(frozen=True, slots=True)
class Subject[SegmentsT: Segments]:
    segments: SegmentsT
```

Use PEP 695 generics for new code. Keep names aligned with existing core concepts where possible.

## User/project-authored API

Examples should define concrete schemas like:

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
    right_shoe: Subject[ShoeSegments]
```

Then instantiate and compile:

```python
subjects = MocapSubjects(...)
scene = build_scene(subjects)
```

## Runtime model

Do not use raw nested `TypedDict` values as the runtime model for algorithms. Compile into normalized specs/views/targets. Runtime code should use methods like:

```python
scene.subject("left_shoe").segment("shoe").marker("heel")
scene.subject("left_shoe").segment("shoe").patch_target("sole")
```

Time-series data should use stable targets:

```python
contacts: Mapping[PatchTarget, BoolArray]
segment_poses: Mapping[SegmentTarget, SegmentPoseTrajectory]
```

## Tests to add

Add focused tests that cover:

- user-authored `TypedDict` schemas compile with `build_scene(...)`;
- subjects/segments/markers/patches are preserved by name;
- marker and patch target labels are stable;
- invalid runtime lookups raise useful `KeyError`/`ValueError`;
- left/right subjects can reuse the same `ShoeSegments` schema;
- dynamic runtime validation still rejects missing targets;
- examples use the new authoring syntax.

## Non-goals

Do not:

- remove runtime specs/views/targets;
- flatten everything into nested dictionaries;
- rewrite demo resampling/sync/contact behavior;
- introduce TypeVar-heavy legacy typing for new generic primitives;
- introduce project-specific schema code under generic package modules;
- make patches marker metadata.

## Verification

Run narrow tests first, then full checks:

```bash
python3 -m compileall -q src/retarget
python3 -m compileall -q examples
pytest -q
```
