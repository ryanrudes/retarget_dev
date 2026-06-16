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

Unbag the ROS bag in JSON format, using the `/tf` topic as the master. Each `/vicon/markers` row is resampled to the nearest `/tf` timestamp. The `keep_non_occluded_markers` processor then replaces that row's marker list with the nearest **non-occluded** reading for **each marker name** within 0.02 seconds of that `/tf` timestamp. Translations are converted from millimeters to meters.

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

## Scene authoring

The preferred public path is the TypedDict authoring layer compiled by `build_scene(...)`.

- Use `Subjects`, `Segments`, `Markers`, and `Patches` to declare the scene shape.
- Use `Subject`, `Segment`, `Marker`, and `Patch` to author concrete scene data.
- Use `Marker.vicon_name` for external/Vicon lookup metadata.
- Use `Patch.label` and `Patch.frame` for display/metadata, not identity.
- Use `Patch(label=...)` to declare a patch without geometry.
- Use `Patch.rectangular(...)` to declare a calibrated patch with geometry.
- `segment.patch_target(...)` works for declared patches; `segment.patch(...)` and `segment.patch_spec(...)` require calibrated geometry.
- `build_scene(...)` compiles authored field names into runtime specs and private generated IDs.

Manual `SceneSpec` / `SubjectSpec` / `SegmentSpec` construction still exists, but it is a
low-level backend path for loader code and geometry examples.

## Segment lookup

Segment IDs are subject-local. At runtime, `SceneState` uses `SegmentKey(subject, segment)` to disambiguate concrete segment instances. For segment views, pass a `SegmentSpec` when you want marker/patch generic types preserved. Pass a `SegmentId` when you want ergonomic runtime lookup.

```python
# Typed path
scene.subject(subject_id).segment(LEFT_SHOE_SEGMENT)

# Runtime lookup path
scene.subject(subject_id).segment(LeftShoeSegmentId.LEFT_SHOE)
```

## Observed marker lookup

Observed marker lookup can use either a `SegmentView` or a `SegmentSpec`.

Preferred after resolving a view:

```python
marker_position(marker_frame, segment=left_shoe_view, marker=LeftShoeMarkerId.HEEL)
```

Alternative without a view:

```python
marker_position(
    marker_frame,
    subject=ViconSubjectId.LEFT_SHOE,
    segment=LEFT_SHOE_SEGMENT,
    marker=LeftShoeMarkerId.HEEL,
)
```

## Example scripts

`new_api_example.py` demonstrates the preferred TypedDict authoring path. `mocap_specs.py` and `run_core_geometry_basics.py` demonstrate low-level backend-oriented scene-spec construction and runtime queries. `run_demo_track_workflow.py` demonstrates the typed demonstration loader and track-query workflow.
