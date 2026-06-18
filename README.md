# Setup

Install the patched `ros2_unbag` from this repo (do not use the PyPI/mamba copy — see `vendor/ros2_unbag/PATCHES.md`):

```bash
source ~/mamba/envs/ros_env/setup.zsh
pip install -e ./vendor/ros2_unbag
```

Build the ROS 2 Vicon bridge:

```bash
cd rosws
colcon build --packages-select vicon_bridge
source install/setup.zsh
cd ..
```

Record a session as a ROS 2 bag (run from `bags/`, with the bridge running and Vicon streaming):

```bash
cd bags
ros2 bag record -a -o stepping_on_board
cd ..
```

`-a` records all topics. `/tf` carries **one transform per tracked Vicon subject in every message** (a 3-subject capture has 3 transforms per `/tf` message); `/vicon/markers` carries the labeled + unlabeled marker cloud.

Unbag the ROS bag in JSON format, using the `/tf` topic as the master. Every subject's transform on `/tf` is exported — each record is keyed by timestamp + `child_frame_id`, so a multi-subject `/tf` is **not** collapsed to a single subject. Each `/vicon/markers` row is resampled to the nearest `/tf` timestamp. The `keep_non_occluded_markers` processor then replaces that row's marker list with the nearest **non-occluded** reading for **each marker name** within 0.02 seconds of that `/tf` timestamp. Translations are converted from millimeters to meters.

At the end of export, `[WARNING]` lines summarize any `/tf` frames dropped by resampling and any per-marker omissions when `discard_eps` could not be met.

```bash
ros2 unbag bags/ground_estimation/ground_estimation_0.db3 \
  --output-dir bags/ground_estimation/unbagged \
  --export /tf:text/json@single_file \
  --export /vicon/markers:text/json@single_file \
  --use-processor ./processors/markers_preprocessors.py \
  --processing /vicon/markers:keep_non_occluded_markers:markers_field=markers,occluded_field=occluded,discard_eps=0.02 \
  --processing /vicon/markers:drop_empty_name_markers \
  --processing /vicon/markers:mm_to_m_translations \
  --resample /tf:nearest,0.02
```

## Typed-first model

The public API is typed and enum-free. You declare scene and demonstration
structure with `TypedDict` schemas, then instantiate frozen dataclasses. The
same objects are the authoring values *and* the bound runtime query surface, so
the full query chain is statically typed with no codegen:

```python
demo.tracks["mocap"]                                  # MocapTrack[MocapSubjects]
mocap.subjects["left_shoe"].segments["shoe"]          # Segment[ShoeMarkers, ShoePatches]
segment.markers["heel"].positions()                   # (T, 3) ndarray
segment.patches["sole"].points()                      # (T, 3) ndarray
```

## Scene authoring

```python
from retarget.core import (
    Marker, Markers, Patch, Patches, Segment, Segments, Subject, Subjects,
    SemanticAxis, bind_scene,
)

class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker
    mid: Marker

class ShoePatches(Patches):
    sole: Patch
    toe_contact: Patch

class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]

class ShoeSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]

subjects = ShoeSubjects(
    left_shoe=Subject(
        mocap_name="Left_Shoe_Improved",
        # Segment-frame marker rest positions (e.g. from a VSK), keyed by
        # mocap_name. Markers inherit these instead of repeating position_segment.
        body_model={
            "left_shoe_heel": (0.0, 0.0, 0.0),
            "left_shoe_toe": (0.20, 0.0, 0.0),
            "left_shoe_mid": (0.10, 0.05, 0.0),
        },
        segments=ShoeSegments(
            shoe=Segment(
                mocap_name="Left_Shoe_Improved",
                markers=ShoeMarkers(
                    heel=Marker(mocap_name="left_shoe_heel"),
                    toe=Marker(mocap_name="left_shoe_toe"),
                    mid=Marker(mocap_name="left_shoe_mid"),
                ),
                patches=ShoePatches(
                    # Fit at bind time from the markers' body_model positions.
                    sole=Patch.rectangle(
                        label="sole",
                        markers=("heel", "toe", "mid"),
                        width=0.10,
                        height=0.25,
                        outward_axis=SemanticAxis.UP,
                        forward_axis=SemanticAxis.FORWARD,
                    ),
                    toe_contact=Patch(label="toe_contact_display"),
                ),
            ),
        ),
    ),
)

scene = bind_scene(subjects)
shoe = scene["left_shoe"].segments["shoe"]
heel_target = shoe.marker_target("heel")
sole_target = shoe.patch_target("sole")
toe_target = shoe.patch_target("toe_contact")   # declaration-only patch is still targetable
# shoe.patches["toe_contact"].points() raises once loaded, because that patch has no geometry.
```

- `Markers`/`Patches`/`Segments`/`Subjects` are `TypedDict` bases declaring scene shape.
- `Marker`/`Patch`/`Segment`/`Subject` are frozen dataclasses for concrete data.
- Authored field names are the canonical identity; `Marker.mocap_name`,
`Segment.mocap_name`, and `Subject.mocap_name` are external/Vicon lookup metadata.
- `Patch(label=...)` declares a patch without geometry;
`Patch.rectangle(markers=...)` fits a rectangular patch frame from calibration
markers at bind time. For an already-known frame, pass `transform_segment_patch`
and a `RectangularRegion` to the base `Patch(...)` constructor.
- `Subject(body_model=...)` supplies segment-frame marker rest positions once per
subject so markers need not repeat `position_segment`.
- `bind_scene(...)` path-binds the schema so `*_target(...)` and geometry work,
and returns the same `SubjectsT` type.

## Stable runtime keys

Targets are plain string-based dataclasses used as stable keys for runtime data:

```python
SegmentTarget(subject="left_shoe", segment="shoe")
MarkerTarget(subject="left_shoe", segment="shoe", marker="heel")
PatchTarget(subject="left_shoe", segment="shoe", patch="sole")
```

Contact tracks are keyed by `PatchTarget`; scene pose state by `SegmentKey`.

## Loading and querying demonstrations

```python
from retarget.demo import MocapTrack, Tracks, Demonstration, MocapTrack.from_unbagged
from retarget.io import UnbaggedDirectory

class GroundEstimationTracks(Tracks):
    mocap: MocapTrack[ShoeSubjects]

mocap = MocapTrack.from_unbagged(UnbaggedDirectory("bags/.../unbagged"), subjects, rebase_time=True)
demo = Demonstration(GroundEstimationTracks(mocap=mocap))

mocap = demo.tracks["mocap"]
shoe = mocap.subjects["left_shoe"].segments["shoe"]
heel_positions = shoe.markers["heel"].positions()        # observed; NaN where unobserved
heel_modeled = shoe.markers["heel"].positions(modeled=True)
sole_points = shoe.patches["sole"].points()
translations = shoe.translations()

# Batch + dict queries (string-keyed):
shoe.marker_positions("heel", "toe")                     # (T, 2, 3)
shoe.marker_positions("heel", "toe", as_dict=True)       # {"heel": (T,3), "toe": (T,3)}

# Slice the track to keep the typed chain; slice the demo for whole-demo views.
clip = mocap.slice_time(0.0, 1.0)
```

## Example scripts

`examples/process_mocap_data/new_api_example.py` is the canonical typed example:
authoring, `bind_scene`, then the `demo.tracks["mocap"]` deep chain with a
bag-backed handoff when `bags/ground_estimation/unbagged/` exists.

`backend_specs/` holds backend/manual loader support that authors the same typed
schema but derives marker geometry and patch calibration from a real VSK file
(`calibrate_patch_transform`, `read_marker_positions_from_vsk`).
`run_demo_track_workflow.py` and `run_core_geometry_basics.py` load that real
data and query it through the public typed deep chain.