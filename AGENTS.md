# AGENTS.md

This repository is building a typed, extensible research toolkit for motion retargeting, demonstration loading, synchronization, SMPL/GVHMR integration, and contact reasoning.

Coding agents must preserve the intended pipeline. Do not flatten the architecture just because Python allows it. Python also allows many crimes against taste.

## Required workflow

Before making changes:

1. Read this file.
2. Read all files in `.cursor/rules/`.
3. Identify which layer you are changing:
   - core primitives
   - mocap specs/state/views
   - IO
   - demonstration tracks
   - alignment/resampling
   - contact state/detection
   - SMPL/GVHMR
   - examples/tests
4. Make the smallest coherent change that advances the requested task.
5. Add or update tests for changed behavior.
6. Run the checks in [Running commands](#running-commands).

If you cannot run checks, explicitly say so and explain why.

## Running commands

Run everything from the **repository root** unless noted otherwise.

On this machine, prefer `python3` (not `python`) when invoking the interpreter directly.

### Default verification (after code changes)

```bash
python3 -m compileall -q src/retarget
python3 -m compileall -q examples
pytest -q
```

`pytest` picks up `src/` via `[tool.pytest.ini_options] pythonpath` in `pyproject.toml`. Do **not** rely on `PYTHONPATH=src` for tests unless you are debugging path issues.

Equivalent with uv:

```bash
uv run pytest -q
```

### Ad-hoc Python (`python3 -c`, one-off scripts)

The `retarget` package is under `src/` and is not necessarily installed into the active environment. Set `PYTHONPATH`:

```bash
PYTHONPATH=src python3 -c "from retarget.demo.mocap import MocapTrack; ..."
```

### Test helpers (`tests/conftest.py`)

`tests/conftest.py` is **not** part of the `retarget` package. Pytest loads it automatically from `tests/`.

Do **not** import it from the repo root:

```bash
# fails: ModuleNotFoundError: No module named 'conftest'
PYTHONPATH=src python3 -c "from conftest import make_mocap_track"
```

To reuse conftest helpers outside pytest, either run from `tests/` or add `tests` to the path:

```bash
cd tests && PYTHONPATH=../src python3 -c "from conftest import make_mocap_track; ..."
# or
PYTHONPATH=src:tests python3 -c "from conftest import make_mocap_track; ..."
```

Prefer `pytest -q` or `pytest -q tests/test_demo_mocap_track.py` over reimplementing setup in `python3 -c`.

### Examples (`examples/process_mocap_data/`)

Example modules (`demo_vocab`, `demo_specs`, `mocap_specs`, `mocap_vocab`) live beside the scripts. Run from that directory with both `src` and the example dir on the path:

```bash
cd examples/process_mocap_data
PYTHONPATH=../../src:. python3 run_demo_example.py
```

### Optional lint/type checks

```bash
uv run ruff check .
uv run mypy src/retarget
```

Only claim these passed if they actually ran.

## Non-negotiable architecture principles

### Static specs are reusable definitions

`SegmentSpec[M, P]` is subject-independent.

It defines:

- segment-local marker vocabulary
- patch vocabulary
- static marker geometry
- axis convention
- patch calibrations
- built patch specs

Do not add `SubjectId` to:

- `SegmentSpec`
- `MarkerHandle`
- `PatchHandle`

A segment spec should be reusable across subjects.

### Runtime identity is subject-scoped

Segment IDs are subject-local.

Runtime state must disambiguate concrete segment instances with:

```python
SegmentKey(subject, segment)
```

Do not key `SceneState` by bare `SegmentId`.

### Views combine static specs with runtime state

`SceneView` must hold both:

```python
spec: SceneSpec
state: SceneState
```

`SubjectView` must hold:

```python
subject_spec: SubjectSpec
state: SceneState
```

Segment lookup must preserve both paths:

```python
# Typed path, preserves marker/patch generics.
scene.subject(subject_id).segment(SEGMENT_SPEC)

# Ergonomic runtime path, returns SegmentView[Any, Any].
scene.subject(subject_id).segment(segment_id)
```

Do not remove either path.

### Concrete dataclasses are for human-friendly authoring

Use concrete dataclass fields for project-specific specs and demos. Use generic iterators for traversal.

Good:

```python
@dataclass(frozen=True, slots=True)
class LeftShoeSubjectSpec(SubjectSpec):
    left_shoe: SegmentSpec[LeftShoeMarkerId, LeftShoePatchId]

    def iter_segments(self) -> Iterable[SegmentSpec[Any, Any]]:
        yield self.left_shoe
```

Avoid concrete fields named `segment` because they shadow `SubjectSpec.segment(...)`.

Keep the defensive base-method lookup in `SubjectView.segment(...)` unless the full hierarchy is proven safe:

```python
SubjectSpec.segment(self.subject_spec, segment)
```

## Demonstration-layer purpose

The demonstration layer should make complete multimodal demonstrations easy to load, slice, align, resample, and query.

Target user-facing feel:

```python
demo = load_ground_estimation_demo(root)
clip = demo.slice_time(2.0, 5.0)

left_shoe = clip.mocap.subject(ViconSubjectId.LEFT_SHOE).segment(
    LeftShoeSegmentId.LEFT_SHOE
)

heel_obs = left_shoe.marker_positions(LeftShoeMarkerId.HEEL)
heel_model = left_shoe.marker_positions(LeftShoeMarkerId.HEEL, modeled=True)
sole_contacts = left_shoe.patch_contacts(LeftShoePatchId.SOLE)
```

Do not force users to manually coordinate marker frames, segment views, and free functions in high-level examples.

Low-level free functions are acceptable inside IO/math utilities. They are not the public demonstration API.

## Track IDs

Tracks must be identified with string-enum IDs, not raw strings.

Use a base like:

```python
class TrackId(NameId):
    """Base class for user-defined demonstration track identifiers."""
```

Concrete demo track IDs should look like:

```python
class GroundEstimationTrackId(TrackId):
    MOCAP = "mocap"
    VIDEO = "video"
    SMPL = "smpl"
    CONTACTS = "contacts"
```

Generic lookup may use enum IDs:

```python
demo.track(GroundEstimationTrackId.MOCAP)
```

Concrete demos should also expose idiomatic properties:

```python
demo.mocap
demo.video
demo.smpl
demo.contacts
```

Both forms are valuable. Keep both.

## Query methods belong on domain objects

User-facing queries should be methods on the objects users already have.

Prefer:

```python
left_shoe.marker_positions(marker)
left_shoe.marker_velocities(marker)
left_shoe.patch_contacts(patch)

person.root_positions()
person.joint_positions(joint)
```

Avoid making high-level users call:

```python
marker_position(marker_frame, segment=left_shoe, marker=marker)
```

That kind of helper may remain in `retarget.io`, but it should not drive the demonstration abstraction.

## Boolean options for binary choices

For observed vs modeled marker data, use:

```python
modeled: bool = False
```

Default marker queries should return observed data when available:

```python
left_shoe.marker_positions(marker)                 # observed
left_shoe.marker_positions(marker, modeled=True)   # modeled
```

Do not create an enum for a two-source choice unless the source space expands beyond binary.

For collection outputs, use:

```python
return_dict: bool = False
```

Default multiple-entity output is time-major AoS:

```python
left_shoe.marker_positions([m1, m2])
# shape: (T, N, 3)
```

Dictionary output preserves typed IDs:

```python
left_shoe.marker_positions([m1, m2], return_dict=True)
# Mapping[M, ndarray shape (T, 3)]
```

Accept single IDs and sequences naturally. Use `isinstance` on typed enum bases where appropriate.

## Internal array layout

Use time-major AoS internally:

- single vector signal: `(T, 3)`
- many markers/joints/patches: `(T, N, 3)`
- rotation matrices: `(T, 3, 3)`
- quaternions: `(T, 4)`
- contact state: `(T,)`
- many contact states: `(T, N)`

Time slicing is the dominant operation. Do not make feature-major SoA the internal default.

## Representation options

Use enums for genuinely multi-way representation choices, such as rotation and pose formats.

Example:

```python
class RotationFormat(NameId):
    MATRIX = "matrix"
    QUATERNION_XYZW = "quaternion_xyzw"
    QUATERNION_WXYZ = "quaternion_wxyz"
    ROTVEC = "rotvec"
```

Boolean parameters are not appropriate for multi-format choices.

## Alignment and resampling principles

The demo layer must support generic alignment between arbitrary tracks while providing human-friendly defaults for common cases like mocap/video/SMPL.

Alignment should operate on extracted time-series energy signals, not hard-coded modality pairs.

Required concepts:

- `EnergySignal`: timestamps plus scalar/vector signal values.
- `TimelineTransform`: maps source-track time to reference-track time.
- `TrackAlignment`: records source, reference, transform, score, and method metadata.
- `demo.align(...)`: generic public alignment API.
- common helpers for mocap-SMPL/video alignment.

Start with offset-only cross-correlation. Leave room for clock scale/drift later.

Resampling is track-specific:

- mocap translations: linear interpolation
- mocap rotations: SLERP or rotation-aware interpolation
- observed markers: linear or nearest with missing-data policy
- video: nearest frame by default
- SMPL root/joint rotations: rotation-aware interpolation
- contact labels: nearest or interval-aware logic

Do not impose one universal interpolation strategy across all modalities.

## Contact-state design principles

Contact must be planned as a first-class derived track, even before contact detection is implemented.

Contact state couples to patch definitions. Contact data should be keyed by `PatchTarget[P]`, not raw strings.

User-facing access should feel patch-local:

```python
left_shoe.patch_contacts(LeftShoePatchId.SOLE)
```

Internally, this should resolve a `PatchTarget` using the subject ID and patch handle.

Contact-track shapes:

- one patch: `(T,)`, bool or confidence array
- many patches: `(T, N)`
- `return_dict=True`: mapping from patch IDs or patch targets to arrays

The future contact detector should produce a contact track from mocap/patch time-series. Do not bolt contact onto marker arrays or stash it in ad-hoc metadata.

## SMPL/GVHMR principles

GVHMR output should become a typed SMPL/body track, not a loose NumPy blob.

Plan for a typed submodule parallel to the existing spec/state/view style:

```text
src/retarget/smpl/
    enums.py
    specs.py
    state.py
    track.py
    views.py
```

SMPL APIs should mirror mocap query style:

```python
person = clip.smpl.body(SmplBodyId.PERSON)

root = person.root_positions()
root_vel = person.root_velocities()

joints = person.joint_positions(SmplJointId.LEFT_ANKLE)

rots = person.joint_rotations(
    SmplJointId.LEFT_HIP,
    format=RotationFormat.QUATERNION_XYZW,
)
```

## Testing expectations

Every new layer must include tests for:

- type-preserving and ergonomic lookup paths
- slicing by time/index
- single ID vs sequence inputs
- default AoS arrays and `return_dict=True`
- observed vs modeled marker outputs when applicable
- representation conversions when added
- alignment transforms and simple known-offset recovery
- contact-track lookup behavior when contact skeleton is added

Do not rely on real video files in tests. Use fake paths, fake timestamps, and small arrays.

## Things not to do

- Do not replace typed IDs with raw strings.
- Do not flatten specs, states, and views into one object.
- Do not put subject identity into reusable segment specs.
- Do not make users call IO free functions in high-level demo examples.
- Do not use one universal interpolation method for all modalities.
- Do not implement contact as arbitrary metadata.
- Do not make SMPL/GVHMR a bag of arrays without typed IDs/specs/views.
- Do not rewrite working core abstractions while adding demo-layer features.

When in doubt, preserve the typed pipeline and add a thin ergonomic layer above it. Do not demolish the basement because you wanted a nicer porch.
