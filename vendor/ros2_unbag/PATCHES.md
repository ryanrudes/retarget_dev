# Local patches (retargeting_from_scratch)

Vendored copy of upstream [ika-rwth-aachen/ros2_unbag](https://github.com/ika-rwth-aachen/ros2_unbag) with the following changes on top of `main`. The nested `.git` directory was removed so this tree is part of the parent repo; `pyproject.toml` sets `fallback_version` for editable installs.

## `ros2_unbag/core/exporter.py`

- **macOS multiprocessing**: force `fork` start method so worker processes can pickle the `rosbag2_py` bag reader (default `spawn` fails on macOS).
- **Resampling `max_index`**: when a master topic is configured, use the master topic message count for `max_index` / progress instead of each topic's raw bag count (fixes unclosed single-file JSON when resampling exports fewer rows).
- **Single-file JSON finalize**: track single-file JSON outputs and append a closing `}` when the last row was never marked `is_last`. Now a redundant safety net — `routines/default.py` keeps the file closed after every write (see below) — but kept as belt-and-suspenders.
- **Resample drop warnings**: log dropped frames at `WARNING` instead of `INFO`.
- **Processor drop summaries**: after export, read `.keep_non_occluded_markers_stats.json` from the output directory (written by `processors/filter_non_occluded_markers.py`) and print a warning summary.

## `ros2_unbag/core/routines/default.py`

- **Multi-transform `/tf` keying**: `bag_reader` flattens each `TFMessage` into one `TransformStamped` per transform, all sharing that `/tf` message's stamp. Single-file JSON/YAML key records by `timestamp.isoformat()`, so several same-stamp transforms (e.g. a 3-subject `/tf`) collided and all but the last were silently lost — a multi-subject `/tf` unbagged down to a single subject. `_serialize_message_with_timestamp` now appends `child_frame_id` (present only on transforms) to the key: `"{iso}#{child_frame_id}"`. Records stay unique and chronologically sortable, and downstream readers parse the subject/segment from each record's `child_frame_id` value, not the key.
- **Crash-safe single-file JSON**: single-file JSON exports are written so the file is a complete, valid JSON object after *every* record (`_append_json_record`): the first record writes `{\n<record>\n}\n`; each later record seeks back over the trailing `}` and rewrites `,\n<record>\n}\n`. This replaces the `is_first`/`is_last` index check, which produced unclosed files (is_last never firing when resampling exports fewer rows than the raw bag count) and concatenated/dangling-comma files (a re-run starting at a non-zero index appended past an already-closed object instead of truncating). "First write" is now detected per worker process (`_json_started_paths`), and a single dedicated sequential worker owns each output, so the seek-back rewrite is in-order and safe. An export interrupted by a crash/Ctrl-C now still leaves loadable JSON. YAML/CSV paths are unchanged.

## `ros2_unbag/export.py`

- Set `ROS2_UNBAG_BAG` and `ROS2_UNBAG_OUTPUT_DIR` environment variables before export so custom processors can read the bag path and write drop stats to the output directory.

## Install

```bash
source ~/mamba/envs/ros_env/setup.zsh
pip install -e /path/to/retargeting_from_scratch/vendor/ros2_unbag
```
