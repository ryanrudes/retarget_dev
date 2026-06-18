# MIT License

# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import json
import os
from pathlib import Path

from rosidl_runtime_py import message_to_ordereddict, message_to_yaml

from ros2_unbag.core.routines.base import ExportRoutine, ExportMode, ExportMetadata
from ros2_unbag.core.utils.file_utils import get_time_from_msg


@ExportRoutine.set_catch_all(["text/json", "text/yaml", "table/csv"], mode=ExportMode.MULTI_FILE)
def export_generic_multi_file(msg, path: Path, fmt: str, metadata: ExportMetadata):
    """
    Generic export handler supporting JSON, YAML, and CSV formats. 
    Serialize the message, determine file extension, and save to the given path.

    Args:
        msg: ROS message instance to export.
        path: Output file path (without extension).
        fmt: Export format string ("text/yaml", "text/json", "table/csv").
        metadata: Export metadata including message index and max index.

    Returns:
        None
    """
    timestamp = get_time_from_msg(msg, return_datetime=True)

    if fmt == "text/json":
        payload = _serialize_message_with_timestamp(msg, "json", timestamp)
        file_ending = ".json"
    elif fmt == "text/yaml":
        payload = _serialize_message_with_timestamp(msg, "yaml", timestamp)
        file_ending = ".yaml"
    elif fmt == "table/csv":
        payload = _serialize_message_with_timestamp(msg, "csv", timestamp)
        file_ending = ".csv"

    # Save the serialized message to a file
    with open(path.with_suffix(file_ending), "w") as f:
        _write_line(f, payload, fmt, True, True)


@ExportRoutine.set_catch_all(["text/json", "text/yaml", "table/csv"], mode=ExportMode.SINGLE_FILE)
def export_generic_single_file(msg, path: Path, fmt: str, metadata: ExportMetadata):
    """
    Generic export handler supporting JSON, YAML, and CSV formats.
    Serialize the message, determine file extension, and append to the given path with file locking (precaution).

    Args:
        msg: ROS message instance to export.
        path: Output file path (without extension).
        fmt: Export format string ("text/yaml", "text/json", "table/csv").
        metadata: Export metadata including message index and max index.

    Returns:
        None
    """
    timestamp = get_time_from_msg(msg, return_datetime=True)

    if fmt == "text/json":
        payload = _serialize_message_with_timestamp(msg, "json", timestamp)
        file_ending = ".json"
    elif fmt == "text/yaml":
        payload = _serialize_message_with_timestamp(msg, "yaml", timestamp)
        file_ending = ".yaml"
    elif fmt == "table/csv":
        payload = _serialize_message_with_timestamp(msg, "csv", timestamp)
        file_ending = ".csv"

    out_path = path.with_suffix(file_ending)

    # JSON is written so the file is a complete, valid object after *every*
    # record. We do not rely on is_first/is_last (the resampling path computes a
    # max_index from the raw bag count, so is_last may never fire, and a re-run
    # may start at a non-zero index, leaving the file unclosed or concatenated).
    # A single dedicated sequential worker owns each single-file output, so
    # writes are in-order and a seek-back rewrite is safe.
    if fmt == "text/json":
        _append_json_record(out_path, payload)
        return

    # Determine if this is the first or last message for the file
    is_first = metadata.index == 0
    is_last = metadata.index == metadata.max_index

    # Save the serialized message to a file - if the filename is constant, messages will be appended
    with open(out_path, "a+") as f:
        if is_first:
            # clear the file if this is the first message
            f.seek(0)
            f.truncate()
        # Write payload line to the file
        _write_line(f, payload, fmt, is_first, is_last)


def _serialize_message_with_timestamp(msg, fmt, timestamp):
    """
    Serialize a ROS message to the specified format.

    Args:
        msg: ROS message instance to serialize.
        fmt: Export format string ("yaml", "json", "csv").
        timestamp: Timestamp to include in the serialized output.

    Returns:
        str: Serialized message as a string.
    """
    # Several messages can share one export timestamp -- notably a TFMessage,
    # which bag_reader flattens into one TransformStamped per transform, all
    # carrying the same /tf message stamp. Single-file JSON/YAML key by timestamp,
    # so without a disambiguator those same-stamp records collide and all but the
    # last are lost (e.g. a 3-subject /tf collapses to 1). child_frame_id is
    # present only on transforms, so append it to keep them distinct.
    child_frame_id = getattr(msg, "child_frame_id", None)
    timestamp_key = timestamp.isoformat()
    if child_frame_id:
        timestamp_key = f"{timestamp_key}#{child_frame_id}"

    if fmt == "json":
        message_dict = message_to_ordereddict(msg)
        serialized_line = json.dumps(message_dict, default=str)
        serialized_line_with_timestamp = f'"{timestamp_key}": {serialized_line}'
        return serialized_line_with_timestamp
    elif fmt == "yaml":
        yaml_content = message_to_yaml(msg)
        indented_yaml_content = "\n".join(f"  {line}" for line in yaml_content.splitlines())
        serialized_line_with_timestamp = f"{timestamp_key}:\n{indented_yaml_content}"
        return serialized_line_with_timestamp
    elif fmt == "csv":
        flat_data = _flatten(message_to_ordereddict(msg))
        header = ["timestamp", *flat_data.keys()]
        values = [str(timestamp), *flat_data.values()]
        return [header, values]


# Single-file JSON outputs this worker process has already begun. The first
# write to a path truncates any prior file; later writes append in place. This
# replaces the brittle metadata.index == 0 "is_first" check.
_json_started_paths: set = set()


def _append_json_record(out_path, record):
    """Append one ``"timestamp": {...}`` record to a single-file JSON export.

    The file is kept a complete, valid JSON object after every call: the first
    record writes ``{\\n<record>\\n}\\n``; each subsequent record seeks back over
    the trailing ``}``, writes ``,\\n<record>\\n}\\n``, and truncates. An export
    interrupted at any point (crash, Ctrl-C, wrong is_last) therefore still
    leaves a loadable object on disk.

    Args:
        out_path: Path of the .json output file.
        record: Serialized ``"timestamp": {...}`` record without braces/comma.

    Returns:
        None
    """
    out_path = Path(out_path)
    if out_path not in _json_started_paths:
        _json_started_paths.add(out_path)
        with open(out_path, "w") as f:
            f.write("{\n" + record + "\n}\n")
        return

    with open(out_path, "r+") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        # Find the closing brace that terminates the object, scanning the tail.
        read_size = min(end, 64)
        f.seek(end - read_size)
        tail = f.read(read_size)
        brace = tail.rfind("}")
        if brace == -1:
            # Not a closed object (empty/corrupt) — rewrite as a fresh object.
            f.seek(0)
            f.truncate()
            f.write("{\n" + record + "\n}\n")
            return
        # Overwrite from the closing brace onward with the new record + close.
        f.seek(end - read_size + brace)
        f.truncate()
        f.write(",\n" + record + "\n}\n")


def _write_line(file, line, filetype, is_first, is_last):
    """
    Write a serialized message line to the file.
    For JSON/YAML, write the string; for CSV, ensure header and write the row.

    Args:
        file: File object to write to.
        line: String for JSON/YAML, or [header, values] list for CSV.
        filetype: Export format string.
        is_first: Boolean indicating if this is the first message for the file.
        is_last: Boolean indicating if this is the last message for the file.

    Returns:
        None
    """

    # Simple writing for yaml
    if "text/yaml" in filetype:
        file.write(line)
        file.write("\n")

    # Writing for json - include parentheses before first line and after last line
    elif "text/json" in filetype:
        if is_first:
            file.write("{\n")
        file.write(line)
        if is_last:
            file.write("\n}\n")
        else:
            file.write(",\n")

    # Writing for csv - include header only for the first line
    if "table/csv" in filetype:
        if is_first:
            _add_csv_header(file, line[0])
        writer = csv.writer(file)
        writer.writerow(line[1])   

    file.flush()


def _add_csv_header(file, header):
    """
    Ensure the CSV file starts with the correct header.

    Args:
        file: File object to write to.
        header: List of column names for the CSV header.

    Returns:
        None
    """
    file.seek(0)
    file.truncate()
    writer = csv.writer(file)
    writer.writerow(header)


def _flatten(d, parent_key='', sep='.'):
    """
    Flatten a nested dict into a single-level dict with compound keys separated by sep.

    Args:
        d: Dictionary to flatten.
        parent_key: Prefix for keys (used in recursion).
        sep: Separator string for compound keys.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
