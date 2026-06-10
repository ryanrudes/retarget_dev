You are doing a correctness + internal-performance pass on the new demo/track/alignment/contact implementation.
Do not redesign the whole system.
Do not remove the typed mocap/spec/state/view architecture.
Do not replace typed IDs with raw strings.
Do not implement SMPL yet.
Do not implement full contact detection yet.
Do not implement real alignment-aware resampling yet.
Do not jump ahead to new features before fixing the correctness issues below.
There are two goals:
1. Finish the hardening pass that was not completed.
2. Replace demo-layer batch queries that loop over scalar view calls with efficient, readable, time-major AoS array internals.
Read these files first:
- AGENTS.md
- .cursor/rules/retarget-architecture.mdc
- .cursor/rules/demo-pipeline.mdc

Phase 1: Finish correctness hardening

1. Fix test packaging

Problem: PYTHONPATH=src:. pytest -q passes, but PYTHONPATH=src pytest -q fails because tests import helpers from another test module:

from tests.test_demo_mocap_track import _make_track

Do not import one test module from another.

Required fix

Create:

tests/conftest.py

Move shared helper enums/specs/helpers there, including things like:

_SubjectId
_SegmentId
_MarkerId
_PatchId
_SEGMENT_SPEC
_make_track

Expose them as fixtures or helper functions.

Then update all tests that currently import from tests.test_demo_mocap_track.

Required check

This must pass:

PYTHONPATH=src pytest -q

Not just:

PYTHONPATH=src:. pytest -q

The obvious test command should work. Shocking standard, apparently.

⸻

2. Fix normalize_entity_input(...)

Problem: the current helper likely does something like:

def normalize_entity_input(entity, entity_type):
    if isinstance(entity, entity_type):
        return (entity,), False
    return tuple(entity), True

This is unsafe because our IDs are StrEnums, which are also strings. Passing a raw string or wrong enum class can be interpreted as a sequence of characters.

Example disaster:

tuple(LeftShoeMarkerId.HEEL)
# ('h', 'e', 'e', 'l')

That must not happen.

Required fix

Replace the helper with strict validation:

from collections.abc import Sequence
from typing import TypeVar
from retarget.core import NameId
E = TypeVar("E")
def normalize_entity_input(
    entity: E | Sequence[E],
    entity_type: type[E],
) -> tuple[tuple[E, ...], bool]:
    if isinstance(entity, entity_type):
        return (entity,), False
    if isinstance(entity, str):
        raise TypeError(
            f"Expected {entity_type.__name__} or a sequence of "
            f"{entity_type.__name__}; got raw string {entity!r}"
        )
    if isinstance(entity, NameId):
        raise TypeError(
            f"Expected {entity_type.__name__}; got {type(entity).__name__}"
        )
    entities = tuple(entity)
    for item in entities:
        if not isinstance(item, entity_type):
            raise TypeError(
                f"Expected all items to be {entity_type.__name__}; "
                f"got {type(item).__name__}"
            )
    return entities, True

Adjust imports/typing to fit the actual module.

Required tests

Add tests that:

left_shoe.marker_positions("heel")

raises TypeError.

Also add a wrong-enum-class test:

class OtherMarkerId(MarkerId):
    HEEL = "heel"
with pytest.raises(TypeError):
    left_shoe.marker_positions(OtherMarkerId.HEEL)

Add similar tests for patch queries if the same helper handles patches.

⸻

3. Fix duplicated ground-estimation specs / enum mismatch

Problem: there is package-level example-specific code like:

src/retarget/demo/ground_estimation_specs.py

and the loader imports:

from retarget.demo.ground_estimation_specs import VICON_SCENE

But the example also has:

examples/process_mocap_data/mocap_vocab.py
examples/process_mocap_data/mocap_specs.py

This creates duplicate enum classes and specs. The demo may be built with one marker enum class and queried with another. Because these IDs are StrEnums, this can fail in subtle ways.

Preferred fix

Remove or stop using package-level ground-estimation-specific specs from src/retarget/demo.

The package should provide generic machinery only:

src/retarget/demo/
    alignment.py
    contact.py
    demo.py
    loaders.py
    mocap.py
    tracks.py

Example-specific vocab/specs/loaders should live in:

examples/process_mocap_data/
    mocap_vocab.py
    mocap_specs.py
    demo_specs.py
    run_demo_example.py

Implement a generic loader helper in src/retarget/demo/loaders.py:

from __future__ import annotations
from pathlib import Path
import numpy as np
from retarget.core import SceneSpec
from retarget.demo.mocap import MocapTrack
from retarget.io import (
    UnbaggedDirectory,
    iter_vicon_marker_frames,
    load_scene_state,
)
def load_mocap_track(
    root: Path | UnbaggedDirectory,
    scene: SceneSpec,
    *,
    tf_prefix: str = "vicon",
) -> MocapTrack:
    export = root if isinstance(root, UnbaggedDirectory) else UnbaggedDirectory(root)
    state = load_scene_state(export, scene, tf_prefix=tf_prefix)
    marker_frames = tuple(iter_vicon_marker_frames(export))
    timestamps = np.array(
        [frame.stamp_seconds for frame in marker_frames],
        dtype=np.float64,
    )
    if state.num_timesteps != len(marker_frames):
        raise ValueError(
            "SceneState timestep count does not match marker frame count: "
            f"{state.num_timesteps} != {len(marker_frames)}"
        )
    return MocapTrack(
        scene_spec=scene,
        state=state,
        timestamps=timestamps,
        marker_frames=marker_frames,
    )

Then define the ground-estimation-specific loader in the example directory, using the example’s own VICON_SCENE and enum classes.

For example:

# examples/process_mocap_data/demo_specs.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from mocap_specs import VICON_SCENE
from retarget.core import TrackId
from retarget.demo import load_mocap_track
from retarget.demo.mocap import MocapTrack
from retarget.demo.contact import ContactTrack
from retarget.demo.alignment import TrackAlignment
class GroundEstimationTrackId(TrackId):
    MOCAP = "mocap"
    VIDEO = "video"
    SMPL = "smpl"
    CONTACTS = "contacts"
@dataclass(frozen=True, slots=True)
class GroundEstimationDemo:
    mocap: MocapTrack
    tracks: Mapping[GroundEstimationTrackId, object]
    alignments: tuple[TrackAlignment[GroundEstimationTrackId], ...] = ()
    def __post_init__(self) -> None:
        object.__setattr__(self, "tracks", MappingProxyType(dict(self.tracks)))
    def track(self, track: GroundEstimationTrackId) -> object:
        if track is GroundEstimationTrackId.MOCAP:
            return self.mocap
        return self.tracks[track]
    @property
    def contacts(self) -> ContactTrack:
        try:
            value = self.tracks[GroundEstimationTrackId.CONTACTS]
        except KeyError as exc:
            raise KeyError("No contact track is attached to this demonstration") from exc
        if not isinstance(value, ContactTrack):
            raise TypeError("CONTACTS track is not a ContactTrack")
        return value
    def slice_time(self, start: float, stop: float) -> GroundEstimationDemoView:
        # implement using self.mocap.slice_time and slicing other tracks if present
        ...
    def with_contacts(self, contacts: ContactTrack) -> GroundEstimationDemo:
        mocap_with_contacts = self.mocap.with_contacts(contacts)
        updated = dict(self.tracks)
        updated[GroundEstimationTrackId.MOCAP] = mocap_with_contacts
        updated[GroundEstimationTrackId.CONTACTS] = contacts
        return GroundEstimationDemo(
            mocap=mocap_with_contacts,
            tracks=updated,
            alignments=self.alignments,
        )
def load_ground_estimation_demo(root: Path) -> GroundEstimationDemo:
    mocap = load_mocap_track(root, VICON_SCENE)
    return GroundEstimationDemo(
        mocap=mocap,
        tracks={GroundEstimationTrackId.MOCAP: mocap},
    )

Adapt this to the actual code. The key point is: do not duplicate ground-estimation vocab/spec classes between src and examples.

Minimum acceptable fix

If moving this is too large for this pass, make run_demo_example.py import the exact same enum classes used by the loader.

But preferred fix is to keep example-specific specs out of src/retarget.

⸻

4. Tighten ContactTrack validation

Problem: ContactTrack currently coerces non-bool contact arrays to bool. That is unsafe.

Required fixes

In ContactTrack.__post_init__:

1. Reject non-bool contact arrays.

if array.dtype != np.bool_:
    raise TypeError("contact arrays must have bool dtype")

Do not coerce.

2. Validate confidence keys are a subset of contact keys.

unknown = set(confidences) - set(contacts)
if unknown:
    raise ValueError(
        "confidences contains targets not present in contacts: "
        f"{unknown}"
    )

3. Validate confidence arrays:

if confidence.shape != (num_timesteps,):
    raise ValueError(...)
if not np.issubdtype(confidence.dtype, np.floating):
    raise TypeError("contact confidence arrays must have floating dtype")
if np.any(confidence < 0.0) or np.any(confidence > 1.0):
    raise ValueError("contact confidences must be in [0, 1]")

Required tests

Add tests for:

* non-bool contacts raise TypeError
* confidence target missing from contacts raises ValueError
* confidence wrong shape raises ValueError
* confidence non-float dtype raises TypeError
* confidence outside [0, 1] raises ValueError
* valid confidence works

⸻

5. Validate contact timestamps against mocap timestamps

Problem: MocapTrack accepts:

contacts: ContactTrack | None

but does not validate that contact timestamps match mocap timestamps.

Then patch_contacts(...) can slice contact data using mocap indices even if timelines differ. That is wrong.

Required fix

In MocapTrack.__post_init__:

if self.contacts is not None:
    if len(self.contacts.timestamps) != len(timestamps):
        raise ValueError(
            "ContactTrack timestamp count must match MocapTrack timestamp count"
        )
    if not np.allclose(self.contacts.timestamps, timestamps):
        raise ValueError(
            "ContactTrack timestamps must match MocapTrack timestamps"
        )

If future contact tracks can have independent timelines, that should be handled through explicit resampling. Do not silently assume indices line up.

Required tests

Add tests that:

* different contact length raises ValueError
* same length but different timestamps raises ValueError
* matching contact timestamps works

⸻

6. Fix with_contacts(...)

Problem: with_contacts(...) creates mocap_with_contacts, but does not update the generic MOCAP track entry.

Required fix

Wherever GroundEstimationDemo.with_contacts(...) lives, ensure:

updated = dict(self.tracks)
updated[GroundEstimationTrackId.MOCAP] = mocap_with_contacts
updated[GroundEstimationTrackId.CONTACTS] = contacts
return GroundEstimationDemo(
    mocap=mocap_with_contacts,
    tracks=updated,
    alignments=self.alignments,
)

Required test

demo_with_contacts = demo.with_contacts(contacts)
assert demo_with_contacts.mocap.contacts is contacts
assert demo_with_contacts.track(GroundEstimationTrackId.MOCAP).contacts is contacts
assert demo_with_contacts.track(GroundEstimationTrackId.CONTACTS) is contacts

Use casts if the generic track(...) return type is object.

⸻

7. Make resample_to(...) honest

Problem: DemonstrationView.resample_to(...) looks implemented but does not apply TrackAlignment transforms.

That is misleading.

Required fix

For now, make it raise:

raise NotImplementedError(
    "DemonstrationView.resample_to requires alignment-aware track resampling; "
    "this is not implemented yet."
)

Do the same for any concrete demo view’s resample_to(...).

Do not pretend resampling works until alignment-aware resampling is actually implemented.

Required test

Add a test asserting resample_to(...) raises NotImplementedError.

⸻

8. Harden alignment

8.1 Strict timestamps

EnergySignal timestamps must be strictly increasing.

if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
    raise ValueError("EnergySignal timestamps must be strictly increasing")

8.2 Robust dt calculation

Do not compute median dt from:

np.diff(np.concatenate([ref_times, src_times]))

Use:

dts = np.concatenate([
    np.diff(reference.timestamps),
    np.diff(source.timestamps),
])
dts = dts[dts > 0]
if len(dts) == 0:
    raise ValueError("Cannot estimate alignment without positive timestamp spacing")
median_dt = float(np.median(dts))

8.3 Constant signals

If either signal is constant, raise a clear error.

if std <= 1e-12:
    raise ValueError("Cannot estimate alignment from a constant energy signal")

8.4 Lock sign convention

Use this convention:

TimelineTransform maps source time to reference time:
    t_reference = scale * t_source + offset

Therefore:

reference peak at 1.0
source peak at 1.3
offset = -0.3

Document this on TimelineTransform.

8.5 Required tests

Replace sine-only sign tests with non-periodic Gaussian/impulse tests.

Test delayed source:

times = np.linspace(0.0, 2.0, 401)
reference = gaussian(times, center=1.0)
source = gaussian(times, center=1.3)
# Expected offset: -0.3

Test advanced source:

reference = gaussian(times, center=1.0)
source = gaussian(times, center=0.7)
# Expected offset: +0.3

Also test:

* duplicate timestamps raise ValueError
* constant reference signal raises ValueError
* constant source signal raises ValueError

⸻

9. Empty slice behavior

Problem: many query methods call np.stack([...]). Empty slices can crash.

Required behavior

For an empty mocap view, return correctly shaped empty arrays:

poses() -> ()
translations() -> np.empty((0, 3))
rotations(matrix) -> np.empty((0, 3, 3))
rotations(quaternion) -> np.empty((0, 4))
marker_positions(single) -> np.empty((0, 3))
marker_positions(multiple) -> np.empty((0, N, 3))
marker_positions(multiple, return_dict=True) -> {marker: np.empty((0, 3))}
marker_velocities(single) -> np.empty((0, 3))
marker_velocities(multiple) -> np.empty((0, N, 3))
patch_points(single) -> np.empty((0, 3))
patch_points(multiple) -> np.empty((0, N, 3))
patch_normals(single) -> np.empty((0, 3))
patch_normals(multiple) -> np.empty((0, N, 3))
patch_contacts(single) -> np.empty((0,), dtype=bool)
patch_contacts(multiple) -> np.empty((0, N), dtype=bool)

Required tests

Add empty-slice tests for:

* translations
* rotations
* marker_positions
* marker_velocities
* patch_points
* patch_normals
* patch_contacts when contact track exists

⸻

10. Remove unused matplotlib

Problem: pyproject.toml includes matplotlib, but it appears unused.

Required check

Run:

grep -R "matplotlib" -n src tests examples pyproject.toml

If only pyproject.toml mentions it, remove it from dependencies.

⸻

11. Add .gitignore and clean generated junk

The artifact contains generated junk:

__MACOSX/
.DS_Store
__pycache__/
*.pyc

Required fix

Add .gitignore:

.DS_Store
__MACOSX/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

If generated files are tracked, remove them from git.

Do not remove intentional source files. Do not delete vendored source. But do not commit nested .git directories or cache artifacts.

⸻

Phase 2: Efficient internal AoS representations

Only start this after Phase 1 passes.

The public API already returns AoS arrays, but current implementation builds them by looping over scalar view calls. Replace that with private full-track array caches and vectorized NumPy operations.

Do not remove scalar methods like:

SegmentView.pose_at(...)
MarkerView.position_world_at(...)
PatchView.contact_point_world_at(...)
PatchView.normal_world_at(...)

Those remain useful for scalar/debug access.

But demo-layer batch queries must not be implemented by repeatedly calling these scalar methods.

⸻

12. Add private MocapArrayCache

Create this in src/retarget/demo/mocap.py or private module src/retarget/demo/_mocap_arrays.py.

If the mocap module is getting large, prefer the private module. Try not to produce a 900-line swamp, a humble request from everyone with eyes.

from dataclasses import dataclass, field
import numpy as np
from retarget.core import SegmentKey
@dataclass(slots=True)
class MocapArrayCache:
    translations: dict[SegmentKey, np.ndarray] = field(default_factory=dict)
    rotations: dict[SegmentKey, np.ndarray] = field(default_factory=dict)
    observed_markers: dict[SegmentKey, np.ndarray] = field(default_factory=dict)

Add to MocapTrack:

_array_cache: MocapArrayCache = field(
    default_factory=MocapArrayCache,
    init=False,
    repr=False,
    compare=False,
)

MocapTrack can remain frozen. The private cache object may be mutable. Document that it is private lazy cache state.

⸻

13. Cache full-track segment translations and rotations

Add methods on MocapTrack:

def segment_translations(self, key: SegmentKey) -> np.ndarray:
    ...
def segment_rotations(self, key: SegmentKey) -> np.ndarray:
    ...

Return full-track arrays:

translations: (T, 3)
rotations:    (T, 3, 3)

Implementation sketch:

def segment_translations(self, key: SegmentKey) -> np.ndarray:
    cached = self._array_cache.translations.get(key)
    if cached is not None:
        return cached
    trajectory = self.state.pose_for_key(key)
    if len(trajectory.poses) == 0:
        arr = np.empty((0, 3), dtype=np.float64)
    else:
        arr = np.stack(
            [pose.translation for pose in trajectory.poses],
            axis=0,
        ).astype(np.float64, copy=False)
    self._array_cache.translations[key] = arr
    return arr

Rotations:

def segment_rotations(self, key: SegmentKey) -> np.ndarray:
    cached = self._array_cache.rotations.get(key)
    if cached is not None:
        return cached
    trajectory = self.state.pose_for_key(key)
    if len(trajectory.poses) == 0:
        arr = np.empty((0, 3, 3), dtype=np.float64)
    else:
        arr = np.stack(
            [pose.rotation for pose in trajectory.poses],
            axis=0,
        ).astype(np.float64, copy=False)
    self._array_cache.rotations[key] = arr
    return arr

Looping once to build the cache is fine. Rebuilding on every query is not.

⸻

14. Use cached arrays in MocapSegmentTrackView

Add helper:

def _source_track(self) -> MocapTrack:
    if isinstance(self.mocap, MocapTrack):
        return self.mocap
    return self.mocap.source
def _segment_key(self) -> SegmentKey:
    return SegmentKey(
        self.segment_view.subject_id,
        self.segment_view.segment_id,
    )
def _index_array(self) -> np.ndarray:
    return np.array(self.indices, dtype=np.intp)

Then implement:

def translations(self) -> TimeVec3:
    if not self.indices:
        return np.empty((0, 3), dtype=np.float64)
    source = self._source_track()
    full = source.segment_translations(self._segment_key())
    return full[self._index_array()]

Rotations:

def _rotation_matrices(self) -> TimeMat3:
    if not self.indices:
        return np.empty((0, 3, 3), dtype=np.float64)
    source = self._source_track()
    full = source.segment_rotations(self._segment_key())
    return full[self._index_array()]

Then rotations(format=...) should call array-format conversion helpers.

⸻

15. Add array-based rotation and pose format conversion

Do not force all non-default pose/rotation outputs through RigidTransform objects.

Add helpers, probably in retarget.demo.mocap or a small private utility module:

def rotation_matrices_to_format(
    rotations: np.ndarray,
    *,
    format: RotationFormat,
) -> np.ndarray:
    ...

Behavior:

RotationFormat.MATRIX -> (T, 3, 3)
RotationFormat.QUATERNION_XYZW -> (T, 4)
RotationFormat.QUATERNION_WXYZ -> (T, 4)
RotationFormat.ROTVEC -> (T, 3)

Implementation sketch:

from scipy.spatial.transform import Rotation
def rotation_matrices_to_format(
    rotations: np.ndarray,
    *,
    format: RotationFormat,
) -> np.ndarray:
    if format is RotationFormat.MATRIX:
        return rotations
    if rotations.shape[0] == 0:
        if format in {
            RotationFormat.QUATERNION_XYZW,
            RotationFormat.QUATERNION_WXYZ,
        }:
            return np.empty((0, 4), dtype=np.float64)
        if format is RotationFormat.ROTVEC:
            return np.empty((0, 3), dtype=np.float64)
    scipy_rotation = Rotation.from_matrix(rotations)
    if format is RotationFormat.QUATERNION_XYZW:
        return scipy_rotation.as_quat()
    if format is RotationFormat.QUATERNION_WXYZ:
        xyzw = scipy_rotation.as_quat()
        return xyzw[:, [3, 0, 1, 2]]
    if format is RotationFormat.ROTVEC:
        return scipy_rotation.as_rotvec()
    raise ValueError(f"Unsupported rotation format: {format}")

Pose conversion:

def pose_arrays_to_format(
    translations: np.ndarray,
    rotations: np.ndarray,
    *,
    format: PoseFormat,
) -> tuple[RigidTransform, ...] | np.ndarray:
    ...

Behavior:

RIGID_TRANSFORM -> tuple[RigidTransform, ...]
MATRIX_4X4 -> (T, 4, 4)
TRANSLATION_QUATERNION_XYZW -> (T, 7)
TRANSLATION_ROTATION_MATRIX -> either (T, 12) or a documented tuple; choose one and test it

Recommendation:
For TRANSLATION_ROTATION_MATRIX, return (T, 12) as [tx, ty, tz, R_flat...], unless current design already says otherwise. Document it.

Only build RigidTransform objects for PoseFormat.RIGID_TRANSFORM.

⸻

16. Vectorize modeled marker positions

Do not use:

[
    self.segment_view.marker(marker).position_world_at(index)
    for index in self.indices
]

Use cached arrays.

For one or many markers:

def _modeled_marker_positions_many(
    self,
    markers: Sequence[M],
) -> TimeEntityVec3:
    num_markers = len(markers)
    if not self.indices:
        return np.empty((0, num_markers, 3), dtype=np.float64)
    source = self._source_track()
    key = self._segment_key()
    R = source.segment_rotations(key)       # (T, 3, 3)
    t = source.segment_translations(key)    # (T, 3)
    P = np.stack(
        [self.segment_view.spec.marker_position(marker) for marker in markers],
        axis=0,
    ).astype(np.float64, copy=False)        # (N, 3)
    # R: (T, 3, 3), P: (N, 3), t: (T, 3)
    world_full = np.einsum("tij,nj->tni", R, P) + t[:, None, :]
    return world_full[self._index_array(), :, :]

Then:

if single:
    return many[:, 0, :]

For multiple marker dictionary output:

return {
    marker: many[:, column, :]
    for column, marker in enumerate(markers)
}

⸻

17. Preindex observed marker positions

Do not call marker_position(...) for every marker and every timestep in demo-layer queries.

Add to MocapTrack:

def observed_marker_positions_for_segment(
    self,
    subject: SubjectId,
    segment: SegmentSpec[M, Any],
) -> np.ndarray:
    ...

Return full-track observed tensor:

shape: (T, M, 3)

where:

M = segment.marker_type.size()

Initialize with np.nan.

Implementation sketch:

def observed_marker_positions_for_segment(
    self,
    subject: SubjectId,
    segment: SegmentSpec[M, Any],
) -> np.ndarray:
    key = SegmentKey(subject, segment.segment)
    cached = self._array_cache.observed_markers.get(key)
    if cached is not None:
        return cached
    if self.marker_frames is None:
        raise ValueError("MocapTrack has no observed marker frames")
    arr = np.full(
        (len(self.timestamps), segment.marker_type.size(), 3),
        np.nan,
        dtype=np.float64,
    )
    subject_label = subject.label
    segment_label = segment.segment.label
    for timestep, frame in enumerate(self.marker_frames):
        for obs in frame.markers:
            if obs.subject_name != subject_label:
                continue
            if obs.segment_name != segment_label:
                continue
            if obs.occluded:
                continue
            try:
                marker = segment.marker_type(obs.marker_name)
            except ValueError:
                continue
            arr[timestep, marker.index, :] = obs.position_world
    self._array_cache.observed_markers[key] = arr
    return arr

Then in MocapSegmentTrackView:

def _observed_marker_positions_many(
    self,
    markers: Sequence[M],
) -> TimeEntityVec3:
    num_markers = len(markers)
    if not self.indices:
        return np.empty((0, num_markers, 3), dtype=np.float64)
    source = self._source_track()
    full = source.observed_marker_positions_for_segment(
        self.segment_view.subject_id,
        self.segment_view.spec,
    )
    marker_indices = [marker.index for marker in markers]
    return full[self._index_array(), :, :][:, marker_indices, :]

This preserves missing observed values as np.nan.

⸻

18. Update marker_positions(...)

Use the vectorized helpers.

def marker_positions(
    self,
    marker: M | Sequence[M],
    *,
    modeled: bool = False,
    return_dict: bool = False,
):
    markers, is_many = normalize_entity_input(
        marker,
        self.segment_view.spec.marker_type,
    )
    if modeled:
        values = self._modeled_marker_positions_many(markers)
    else:
        values = self._observed_marker_positions_many(markers)
    if return_dict:
        return {
            marker_id: values[:, i, :]
            for i, marker_id in enumerate(markers)
        }
    if is_many:
        return values
    return values[:, 0, :]

If return_dict=True is passed for a single marker, either:

1. return {marker: values[:, 0, :]}, or
2. raise TypeError.

Pick one and test it. I prefer allowing it.

⸻

19. Update marker velocities

Implement velocities from vectorized position arrays:

def _differentiate(self, values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return np.empty_like(values)
    if values.shape[0] == 1:
        return np.zeros_like(values)
    return np.gradient(values, self.timestamps, axis=0)

But note: self.timestamps for a view should be the view timestamps, not full-track timestamps.

For a MocapSegmentTrackView, implement:

@property
def timestamps(self) -> np.ndarray:
    source = self._source_track()
    return source.timestamps[self._index_array()]

Then:

np.gradient(values, self.timestamps, axis=0)

For NaN observed positions, velocities will propagate NaNs. That is fine.

If values.shape[0] == 1, return zeros or NaNs? Choose and document. I recommend zeros for modeled data and np.nan for observed missing-heavy data is more nuanced, but for now np.gradient cannot operate on one point. Use zeros for one-sample velocity and document it.

⸻

20. Vectorize patch points and normals

Inspect current PatchView.contact_point_world_at(...) and PatchView.normal_world_at(...). Reproduce exactly.

Likely:

point_segment = patch_spec.transform_segment_patch.translation
normal_segment = patch_spec.transform_segment_patch.rotation @ np.array([0, 0, 1])

Do not guess. Inspect actual code.

Implement:

def _patch_points_segment_many(self, patches: Sequence[P]) -> np.ndarray:
    ...
def _patch_normals_segment_many(self, patches: Sequence[P]) -> np.ndarray:
    ...

Then:

def _patch_points_many(self, patches: Sequence[P]) -> TimeEntityVec3:
    num_patches = len(patches)
    if not self.indices:
        return np.empty((0, num_patches, 3), dtype=np.float64)
    source = self._source_track()
    key = self._segment_key()
    R = source.segment_rotations(key)       # (T, 3, 3)
    t = source.segment_translations(key)    # (T, 3)
    P = self._patch_points_segment_many(patches)  # (N, 3)
    # R: (T, 3, 3), P: (N, 3), t: (T, 3)
    world_full = np.einsum("tij,nj->tni", R, P) + t[:, None, :]
    return world_full[self._index_array(), :, :]

Normals:

def _patch_normals_many(self, patches: Sequence[P]) -> TimeEntityVec3:
    num_patches = len(patches)
    if not self.indices:
        return np.empty((0, num_patches, 3), dtype=np.float64)
    source = self._source_track()
    key = self._segment_key()
    R = source.segment_rotations(key)             # (T, 3, 3)
    N = self._patch_normals_segment_many(patches) # (N, 3)
    world_full = np.einsum("tij,nj->tni", R, N)
    return world_full[self._index_array(), :, :]

Then update patch_points(...) and patch_normals(...) to use these helpers and support return_dict=True.

⸻

21. Cache only full-track arrays

Do not cache sliced arrays.

Good:

MocapTrack._array_cache.translations[SegmentKey] = full_T_array

Bad:

MocapTrack._array_cache.translations[(SegmentKey, indices)] = sliced_array

Views should only slice full arrays:

full[self._index_array()]

Caching sliced views creates memory growth and stale semantics. Tiny optimization trapdoor, do not step on it.

⸻

22. Keep implementation readable

Do not turn this into an unreadable NumPy puzzle.

Use private helpers with clear names:

_source_track()
_segment_key()
_index_array()
_modeled_marker_positions_many()
_observed_marker_positions_many()
_patch_points_many()
_patch_normals_many()
rotation_matrices_to_format()
pose_arrays_to_format()

Add shape comments around einsum:

# R: (T, 3, 3), P: (N, 3), t: (T, 3)
world = np.einsum("tij,nj->tni", R, P) + t[:, None, :]

The goal is efficient internals, not a code-golf ritual for people who hate their coworkers.

⸻

23. Tests for efficient internals

Add behavioral tests. Do not test private cache implementation details unless necessary.

Required tests:

23.1 Modeled multi-marker consistency

multi = left_shoe.marker_positions([HEEL, TOE], modeled=True)
heel = left_shoe.marker_positions(HEEL, modeled=True)
toe = left_shoe.marker_positions(TOE, modeled=True)
np.testing.assert_allclose(multi[:, 0, :], heel)
np.testing.assert_allclose(multi[:, 1, :], toe)

23.2 Observed multi-marker consistency

Same as above for observed data, with equal_nan=True.

np.testing.assert_allclose(multi[:, 0, :], heel, equal_nan=True)

23.3 Patch vectorized output matches scalar view oracle

Use scalar PatchView methods only in tests as an oracle:

expected = np.stack(
    [
        segment_view.patch(SOLE).contact_point_world_at(i)
        for i in range(track_length)
    ],
    axis=0,
)
np.testing.assert_allclose(left_shoe.patch_points(SOLE), expected)

Same for normals.

23.4 Repeated calls stable

a = left_shoe.marker_positions([HEEL, TOE], modeled=True)
b = left_shoe.marker_positions([HEEL, TOE], modeled=True)
np.testing.assert_allclose(a, b)

23.5 Empty slice shapes

Covered above, but make sure they still pass after vectorization.

⸻

Final checks

At the end, run:

python -m compileall -q src/retarget
python -m compileall -q examples
PYTHONPATH=src pytest -q

If available:

ruff check .
mypy src/retarget

Do not claim ruff or mypy passed unless they actually ran.

Expected final state

After this pass:

* PYTHONPATH=src pytest -q passes.
* Wrong enum classes and raw strings fail fast.
* Example-specific vocab/specs are not duplicated between package and examples.
* Contact track validation is strict.
* Contact timestamps match mocap timestamps when attached.
* with_contacts(...) keeps .mocap and track(MOCAP) consistent.
* resample_to(...) is explicitly not implemented until alignment-aware resampling exists.
* Alignment has strict timestamp validation, robust dt calculation, constant-signal errors, and non-periodic sign tests.
* Empty slices return correctly shaped empty arrays.
* Unused matplotlib is removed.
* .gitignore exists and generated junk is removed.
* Demo-layer batch queries use cached full-track AoS arrays and vectorized operations, not repeated scalar view calls.

Do not start SMPL, video decoding, full contact detection, or real resampling until these are complete.