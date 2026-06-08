# Local patches (retargeting_from_scratch)

Vendored copy of upstream [ika-rwth-aachen/ros2_unbag](https://github.com/ika-rwth-aachen/ros2_unbag) with the following changes on top of `main`. The nested `.git` directory was removed so this tree is part of the parent repo; `pyproject.toml` sets `fallback_version` for editable installs.

## `ros2_unbag/core/exporter.py`

- **macOS multiprocessing**: force `fork` start method so worker processes can pickle the `rosbag2_py` bag reader (default `spawn` fails on macOS).
- **Resampling `max_index`**: when a master topic is configured, use the master topic message count for `max_index` / progress instead of each topic's raw bag count (fixes unclosed single-file JSON when resampling exports fewer rows).
- **Single-file JSON finalize**: track single-file JSON outputs and append a closing `}` when the last row was never marked `is_last`.
- **Resample drop warnings**: log dropped frames at `WARNING` instead of `INFO`.
- **Processor drop summaries**: after export, read `.keep_non_occluded_markers_stats.json` from the output directory (written by `processors/filter_non_occluded_markers.py`) and print a warning summary.

## `ros2_unbag/export.py`

- Set `ROS2_UNBAG_BAG` and `ROS2_UNBAG_OUTPUT_DIR` environment variables before export so custom processors can read the bag path and write drop stats to the output directory.

## Install

```bash
source ~/mamba/envs/ros_env/setup.zsh
pip install -e /path/to/retargeting_from_scratch/vendor/ros2_unbag
```
