import json
import os
import sqlite3
from bisect import bisect_left
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ros2_unbag.core.processors.base import Processor

_DROP_STATS_FILENAME = ".keep_non_occluded_markers_stats.json"

_MARKER_CACHE: dict[str, list] | None = None
_OCCLUSION_PCT_BY_MARKER: dict[str, float] | None = None
_DROP_STATS = {
    "frames": 0,
    "frames_with_drops": 0,
    "frames_with_all_dropped": 0,
    "drops_by_marker": Counter(),
}


@dataclass(frozen=True)
class _MarkerSample:
    time_s: float
    frame_number: int
    marker: object


def _stamp_to_seconds(msg) -> float:
    stamp = msg.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


def _get_nested_attr(obj, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _set_nested_attr(obj, path: str, value):
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _is_unnamed_marker(
    marker,
    marker_name_field: str = "marker_name",
    subject_name_field: str = "subject_name",
    segment_name_field: str = "segment_name",
) -> bool:
    return (
        getattr(marker, marker_name_field) == ""
        and getattr(marker, subject_name_field) == ""
        and getattr(marker, segment_name_field) == ""
    )


def _load_marker_cache(
    bag_path: str, markers_topic: str, markers_field: str, occluded_field: str
) -> tuple[dict[str, list[_MarkerSample]], dict[str, float]]:
    conn = sqlite3.connect(bag_path)
    topics = {row[0]: (row[1], row[2]) for row in conn.execute("SELECT id, name, type FROM topics")}
    topic_id = next(tid for tid, (name, _) in topics.items() if name == markers_topic)
    msg_type = __import__(
        "rosidl_runtime_py.utilities", fromlist=["get_message"]
    ).get_message(topics[topic_id][1])
    deserialize_message = __import__(
        "rclpy.serialization", fromlist=["deserialize_message"]
    ).deserialize_message

    by_name: dict[str, list[_MarkerSample]] = {}
    occlusion_counts: dict[str, tuple[int, int]] = {}
    rows = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic_id,),
    )
    for timestamp_ns, data in rows:
        msg = deserialize_message(data, msg_type)
        time_s = timestamp_ns * 1e-9
        for marker in _get_nested_attr(msg, markers_field):
            if _is_unnamed_marker(marker):
                continue
            marker_name = marker.marker_name
            occluded_total, reading_total = occlusion_counts.setdefault(marker_name, (0, 0))
            reading_total += 1
            if getattr(marker, occluded_field):
                occluded_total += 1
                occlusion_counts[marker_name] = (occluded_total, reading_total)
                continue
            occlusion_counts[marker_name] = (occluded_total, reading_total)
            by_name.setdefault(marker_name, []).append(
                _MarkerSample(
                    time_s=time_s,
                    frame_number=msg.frame_number,
                    marker=marker,
                )
            )

    for samples in by_name.values():
        samples.sort(key=lambda sample: sample.time_s)
    occlusion_pct_by_marker = {
        marker_name: 100.0 * occluded / total if total else 0.0
        for marker_name, (occluded, total) in occlusion_counts.items()
    }
    return by_name, occlusion_pct_by_marker


def _get_marker_cache(
    markers_topic: str, markers_field: str, occluded_field: str
) -> dict[str, list[_MarkerSample]]:
    global _MARKER_CACHE, _OCCLUSION_PCT_BY_MARKER
    if _MARKER_CACHE is None:
        bag_path = os.environ.get("ROS2_UNBAG_BAG")
        if not bag_path:
            raise RuntimeError(
                "ROS2_UNBAG_BAG is not set. Use a patched ros2_unbag that exports the bag path."
            )
        _MARKER_CACHE, _OCCLUSION_PCT_BY_MARKER = _load_marker_cache(
            bag_path, markers_topic, markers_field, occluded_field
        )
    return _MARKER_CACHE


def _write_drop_stats() -> None:
    output_dir = os.environ.get("ROS2_UNBAG_OUTPUT_DIR")
    if not output_dir:
        return

    stats_path = Path(output_dir) / _DROP_STATS_FILENAME
    stats_path.write_text(
        json.dumps(
            {
                "frames": _DROP_STATS["frames"],
                "frames_with_drops": _DROP_STATS["frames_with_drops"],
                "frames_with_all_dropped": _DROP_STATS["frames_with_all_dropped"],
                "drops_by_marker": dict(_DROP_STATS["drops_by_marker"]),
                "occlusion_pct_by_marker": _OCCLUSION_PCT_BY_MARKER or {},
            },
            indent=2,
        )
    )


def _nearest_sample(
    samples: list[_MarkerSample], target_s: float, discard_eps_s: float
) -> _MarkerSample | None:
    if not samples:
        return None

    sample_times = [sample.time_s for sample in samples]
    idx = bisect_left(sample_times, target_s)

    best = None
    best_delta = None
    for candidate_idx in (idx - 1, idx):
        if 0 <= candidate_idx < len(samples):
            sample = samples[candidate_idx]
            delta = abs(sample.time_s - target_s)
            if delta <= discard_eps_s and (best_delta is None or delta < best_delta):
                best = sample
                best_delta = delta
    return best


@Processor(["vicon_bridge/msg/Markers"], ["keep_non_occluded_markers"])
def keep_non_occluded_markers(
    msg,
    markers_field: str = "markers",
    occluded_field: str = "occluded",
    markers_topic: str = "/vicon/markers",
    discard_eps: str = "0.02",
    warn_on_drop: str = "true",
):
    """
    For each /tf-aligned timestamp, rebuild markers from the nearest non-occluded
    reading of each marker name within discard_eps seconds.

    Args:
        markers_field:
            Attribute path containing the marker list. Usually "markers".

        occluded_field:
            Boolean attribute on each marker that indicates occlusion.

        markers_topic:
            Marker topic to preload from the bag. Usually "/vicon/markers".

        discard_eps:
            Maximum time delta in seconds between the /tf timestamp and a marker
            reading. Should match the --resample discard window.

        warn_on_drop:
            When true, print a summary warning at export end for markers omitted
            because discard_eps could not be met.
    """
    target_s = _stamp_to_seconds(msg)
    window_s = float(discard_eps)
    cache = _get_marker_cache(markers_topic, markers_field, occluded_field)
    track_drops = warn_on_drop.lower() not in {"0", "false", "no"}

    chosen_markers = []
    dropped_marker_names = []
    frame_number = None
    for marker_name in sorted(cache):
        sample = _nearest_sample(cache[marker_name], target_s, window_s)
        if sample is not None:
            chosen_markers.append(deepcopy(sample.marker))
            if frame_number is None:
                frame_number = sample.frame_number
        elif track_drops:
            dropped_marker_names.append(marker_name)

    if track_drops:
        _DROP_STATS["frames"] += 1
        if dropped_marker_names:
            _DROP_STATS["frames_with_drops"] += 1
            if not chosen_markers:
                _DROP_STATS["frames_with_all_dropped"] += 1
            for marker_name in dropped_marker_names:
                _DROP_STATS["drops_by_marker"][marker_name] += 1
        _write_drop_stats()

    _set_nested_attr(msg, markers_field, chosen_markers)
    if chosen_markers and hasattr(msg, "frame_number") and frame_number is not None:
        msg.frame_number = frame_number
    return msg
