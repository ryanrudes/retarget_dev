# Processors

Custom [ros2_unbag](https://github.com/ika-rwth-aachen/ros2_unbag) processors for `/vicon/markers` export. Processors run on each message during unbag, after resampling and before writing JSON.

Load them with a single entry point:

```bash
--use-processor ./processors/markers_preprocessors.py
```

Then chain processors per topic with `--processing` (order matters):

| Processor | File | Purpose |
|-----------|------|---------|
| `keep_non_occluded_markers` | `filter_non_occluded_markers.py` | Rebuild each row from the nearest non-occluded bag reading per marker name within `discard_eps` |
| `drop_empty_name_markers` | `drop_empty_name_markers.py` | Remove markers with empty `marker_name`, `subject_name`, and `segment_name` |
| `mm_to_m_translations` | `mm_to_m_markers.py` | Convert marker translations from millimeters to meters |

`keep_non_occluded_markers` reads the bag path from `ROS2_UNBAG_BAG` (set by the patched `ros2_unbag` under `vendor/ros2_unbag`). At export end it writes drop statistics that `ros2_unbag` prints as warnings, including each marker's occlusion rate in the original bag.

See the project [README](../README.md) for the full unbag command.
