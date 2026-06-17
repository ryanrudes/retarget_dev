from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from retarget.core import (
    Marker,
    Markers,
    Patch,
    Patches,
    Segment,
    SegmentKey,
    Segments,
    Subject,
    Subjects,
)
from retarget.io import (
    UnbaggedDirectory,
    iter_vicon_marker_frames,
    load_scene_state,
    marker_positions_by_name,
    parse_tf_child_frame,
    rigid_transform_from_ros_json,
    stamp_to_seconds,
    tf_child_frame_id,
)

_EXTERNAL = "Left_Shoe_Improved"


def _write_unbagged(
    directory: Path,
    *,
    tf_messages: dict[str, object],
    marker_messages: dict[str, object],
) -> UnbaggedDirectory:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tf.json").write_text(json.dumps(tf_messages), encoding="utf-8")
    (directory / "vicon_markers.json").write_text(
        json.dumps(marker_messages), encoding="utf-8"
    )
    return UnbaggedDirectory(directory)


def _tf_message(
    *, stamp_sec: int, stamp_nsec: int, child_frame_id: str,
    translation: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "header": {"stamp": {"sec": stamp_sec, "nanosec": stamp_nsec}, "frame_id": "vicon/world"},
        "child_frame_id": child_frame_id,
        "transform": {
            "translation": {"x": translation[0], "y": translation[1], "z": translation[2]},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }


def _marker_message(
    *, stamp_sec: int, stamp_nsec: int, marker_name: str,
    translation: tuple[float, float, float], occluded: bool = False,
) -> dict[str, object]:
    return {
        "header": {"stamp": {"sec": stamp_sec, "nanosec": stamp_nsec}, "frame_id": ""},
        "frame_number": 1,
        "markers": [
            {
                "marker_name": marker_name,
                "subject_name": _EXTERNAL,
                "segment_name": _EXTERNAL,
                "translation": {"x": translation[0], "y": translation[1], "z": translation[2]},
                "occluded": occluded,
            }
        ],
    }


class _LoadMarkers(Markers):
    heel: Marker


class _LoadPatches(Patches):
    sole: Patch


class _LoadSegments(Segments):
    shoe: Segment[_LoadMarkers, _LoadPatches]


class _LoadSubjects(Subjects):
    left_shoe: Subject[_LoadSegments]


class _DuplicateSubjects(Subjects):
    left_shoe: Subject[_LoadSegments]
    right_shoe: Subject[_LoadSegments]


def _segment(vicon_name: str, marker_vicon: str) -> Segment[_LoadMarkers, _LoadPatches]:
    return Segment(
        vicon_name=vicon_name,
        markers=_LoadMarkers(heel=Marker(vicon_name=marker_vicon)),
        patches=_LoadPatches(sole=Patch(label="sole")),
    )


def _single_subject() -> _LoadSubjects:
    return _LoadSubjects(
        left_shoe=Subject(
            vicon_name=_EXTERNAL,
            segments=_LoadSegments(shoe=_segment(_EXTERNAL, "left_shoe_heel")),
        )
    )


def _duplicate_external_subjects() -> _DuplicateSubjects:
    return _DuplicateSubjects(
        left_shoe=Subject(
            vicon_name=_EXTERNAL,
            segments=_LoadSegments(shoe=_segment(_EXTERNAL, "left_shoe_heel")),
        ),
        right_shoe=Subject(
            vicon_name=_EXTERNAL,
            segments=_LoadSegments(shoe=_segment(_EXTERNAL, "right_shoe_heel")),
        ),
    )


def test_stamp_to_seconds() -> None:
    assert stamp_to_seconds({"sec": 2, "nanosec": 500_000_000}) == pytest.approx(2.5)


def test_parse_and_build_tf_child_frame() -> None:
    assert parse_tf_child_frame(f"vicon/{_EXTERNAL}/{_EXTERNAL}") == (_EXTERNAL, _EXTERNAL)
    assert tf_child_frame_id(_EXTERNAL, _EXTERNAL) == f"vicon/{_EXTERNAL}/{_EXTERNAL}"


def test_rigid_transform_from_ros_json() -> None:
    transform = rigid_transform_from_ros_json(
        {
            "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    )
    np.testing.assert_allclose(transform.translation, np.array([1.0, 2.0, 3.0]))


def test_load_scene_state_uses_external_tf_names(tmp_path: Path) -> None:
    child_frame = tf_child_frame_id(_EXTERNAL, _EXTERNAL)
    export = _write_unbagged(
        tmp_path,
        tf_messages={
            "2026-06-07T16:55:32.000000": _tf_message(
                stamp_sec=1, stamp_nsec=0, child_frame_id=child_frame, translation=(0.0, 0.0, 0.0)
            ),
            "2026-06-07T16:55:32.100000": _tf_message(
                stamp_sec=1, stamp_nsec=100_000_000, child_frame_id=child_frame, translation=(0.0, 0.0, 0.1)
            ),
        },
        marker_messages={},
    )

    state = load_scene_state(export, _single_subject())
    trajectory = state.pose_for_key(SegmentKey("left_shoe", "shoe"))
    assert trajectory.num_timesteps == 2
    np.testing.assert_allclose(trajectory.at(1).translation, np.array([0.0, 0.0, 0.1]))


def test_load_scene_state_rejects_duplicate_external_tf_targets(tmp_path: Path) -> None:
    export = _write_unbagged(tmp_path, tf_messages={}, marker_messages={})
    with pytest.raises(ValueError, match="Duplicate external subject/segment names"):
        load_scene_state(export, _duplicate_external_subjects())


def test_iter_vicon_marker_frames(tmp_path: Path) -> None:
    export = _write_unbagged(
        tmp_path,
        tf_messages={},
        marker_messages={
            "2026-06-07T16:55:32.000000": _marker_message(
                stamp_sec=1, stamp_nsec=0, marker_name="heel", translation=(0.9, 0.1, 0.0)
            ),
            "2026-06-07T16:55:32.100000": _marker_message(
                stamp_sec=1, stamp_nsec=100_000_000, marker_name="heel",
                translation=(0.9, 0.1, 0.1), occluded=True,
            ),
        },
    )

    frames = list(iter_vicon_marker_frames(export))
    assert len(frames) == 2
    assert frames[0].stamp_seconds == pytest.approx(1.0)
    positions = marker_positions_by_name(
        frames[0], subject_name=_EXTERNAL, segment_name=_EXTERNAL
    )
    np.testing.assert_allclose(positions["heel"], np.array([0.9, 0.1, 0.0]))
    assert "heel" not in marker_positions_by_name(frames[1])
