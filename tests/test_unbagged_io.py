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
    RigidTransform,
    SceneSpec,
    SegmentId,
    SegmentKey,
    SubjectId,
    Subject,
    Subjects,
    Segment,
    Segments,
    build_scene,
    SubjectSpec,
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


class _SubjectId(SubjectId):
    LEFT_SHOE = "Left_Shoe_Improved"


class _SegmentId(SegmentId):
    LEFT_SHOE = "Left_Shoe_Improved"


def _write_unbagged(
    directory: Path,
    *,
    tf_messages: dict[str, object],
    marker_messages: dict[str, object],
) -> UnbaggedDirectory:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tf.json").write_text(json.dumps(tf_messages), encoding="utf-8")
    (directory / "vicon_markers.json").write_text(
        json.dumps(marker_messages),
        encoding="utf-8",
    )
    return UnbaggedDirectory(directory)


def _tf_message(
    *,
    stamp_sec: int,
    stamp_nsec: int,
    child_frame_id: str,
    translation: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "header": {
            "stamp": {"sec": stamp_sec, "nanosec": stamp_nsec},
            "frame_id": "vicon/world",
        },
        "child_frame_id": child_frame_id,
        "transform": {
            "translation": {
                "x": translation[0],
                "y": translation[1],
                "z": translation[2],
            },
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }


def _marker_message(
    *,
    stamp_sec: int,
    stamp_nsec: int,
    marker_name: str,
    translation: tuple[float, float, float],
    occluded: bool = False,
) -> dict[str, object]:
    return {
        "header": {
            "stamp": {"sec": stamp_sec, "nanosec": stamp_nsec},
            "frame_id": "",
        },
        "frame_number": 1,
        "markers": [
            {
                "marker_name": marker_name,
                "subject_name": _SubjectId.LEFT_SHOE.label,
                "segment_name": _SegmentId.LEFT_SHOE.label,
                "translation": {
                    "x": translation[0],
                    "y": translation[1],
                    "z": translation[2],
                },
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


def _authored_single_subject_scene(
    *,
    subject_vicon_name: str = "Left_Shoe_Improved",
    segment_vicon_name: str = "Left_Shoe_Improved",
) -> SceneSpec:
    return build_scene(
        _LoadSubjects(
            left_shoe=Subject(
                vicon_name=subject_vicon_name,
                segments=_LoadSegments(
                    shoe=Segment(
                        vicon_name=segment_vicon_name,
                        markers=_LoadMarkers(
                            heel=Marker(vicon_name="left_shoe_heel"),
                        ),
                        patches=_LoadPatches(
                            sole=Patch(label="sole"),
                        ),
                    )
                ),
            )
        )
    )


def _authored_duplicate_external_name_scene() -> SceneSpec:
    return build_scene(
        _DuplicateSubjects(
            left_shoe=Subject(
                vicon_name="Left_Shoe_Improved",
                segments=_LoadSegments(
                    shoe=Segment(
                        vicon_name="Left_Shoe_Improved",
                        markers=_LoadMarkers(
                            heel=Marker(vicon_name="left_shoe_heel"),
                        ),
                        patches=_LoadPatches(
                            sole=Patch(label="sole"),
                        ),
                    )
                ),
            ),
            right_shoe=Subject(
                vicon_name="Left_Shoe_Improved",
                segments=_LoadSegments(
                    shoe=Segment(
                        vicon_name="Left_Shoe_Improved",
                        markers=_LoadMarkers(
                            heel=Marker(vicon_name="right_shoe_heel"),
                        ),
                        patches=_LoadPatches(
                            sole=Patch(label="sole"),
                        ),
                    )
                ),
            ),
        )
    )


class _LeftShoeSubjectSpec(SubjectSpec):
    def __init__(self) -> None:
        object.__setattr__(self, "subject", _SubjectId.LEFT_SHOE)

    def iter_segments(self):
        yield _LeftShoeSegmentSpec()


class _LeftShoeSegmentSpec:
    segment = _SegmentId.LEFT_SHOE


class _SceneSpec(SceneSpec):
    def iter_subjects(self):
        yield _LeftShoeSubjectSpec()


def test_stamp_to_seconds() -> None:
    assert stamp_to_seconds({"sec": 2, "nanosec": 500_000_000}) == pytest.approx(2.5)


def test_parse_tf_child_frame() -> None:
    assert parse_tf_child_frame(
        "vicon/Left_Shoe_Improved/Left_Shoe_Improved"
    ) == ("Left_Shoe_Improved", "Left_Shoe_Improved")
    assert tf_child_frame_id(_SubjectId.LEFT_SHOE, _SegmentId.LEFT_SHOE) == (
        "vicon/Left_Shoe_Improved/Left_Shoe_Improved"
    )


def test_rigid_transform_from_ros_json() -> None:
    transform = rigid_transform_from_ros_json(
        {
            "translation": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    )
    np.testing.assert_allclose(transform.translation, np.array([1.0, 2.0, 3.0]))


def test_load_scene_state_from_unbagged_directory(tmp_path: Path) -> None:
    child_frame = tf_child_frame_id(_SubjectId.LEFT_SHOE, _SegmentId.LEFT_SHOE)
    export = _write_unbagged(
        tmp_path,
        tf_messages={
            "2026-06-07T16:55:32.000000": _tf_message(
                stamp_sec=1,
                stamp_nsec=0,
                child_frame_id=child_frame,
                translation=(0.0, 0.0, 0.0),
            ),
            "2026-06-07T16:55:32.100000": _tf_message(
                stamp_sec=1,
                stamp_nsec=100_000_000,
                child_frame_id=child_frame,
                translation=(0.0, 0.0, 0.1),
            ),
        },
        marker_messages={},
    )

    state = load_scene_state(export, _SceneSpec())
    key = SegmentKey(_SubjectId.LEFT_SHOE, _SegmentId.LEFT_SHOE)
    trajectory = state.pose_for_key(key)

    assert trajectory.num_timesteps == 2
    np.testing.assert_allclose(
        trajectory.at(0).translation,
        np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        trajectory.at(1).translation,
        np.array([0.0, 0.0, 0.1]),
    )


def test_load_scene_state_uses_external_tf_names(tmp_path: Path) -> None:
    scene = _authored_single_subject_scene()
    export = _write_unbagged(
        tmp_path,
        tf_messages={
            "2026-06-07T16:55:32.000000": _tf_message(
                stamp_sec=1,
                stamp_nsec=0,
                child_frame_id="vicon/Left_Shoe_Improved/Left_Shoe_Improved",
                translation=(0.0, 0.0, 0.0),
            ),
            "2026-06-07T16:55:32.100000": _tf_message(
                stamp_sec=1,
                stamp_nsec=100_000_000,
                child_frame_id="vicon/Left_Shoe_Improved/Left_Shoe_Improved",
                translation=(0.0, 0.0, 0.1),
            ),
        },
        marker_messages={},
    )

    state = load_scene_state(export, scene)
    subject_id = scene.generated_ids.subjects.left_shoe
    segment_id = scene.generated_ids.segments[subject_id].shoe
    key = SegmentKey(subject_id, segment_id)
    trajectory = state.pose_for_key(key)

    assert trajectory.num_timesteps == 2
    np.testing.assert_allclose(
        trajectory.at(0).translation,
        np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        trajectory.at(1).translation,
        np.array([0.0, 0.0, 0.1]),
    )


def test_load_scene_state_rejects_duplicate_external_tf_targets(tmp_path: Path) -> None:
    scene = _authored_duplicate_external_name_scene()
    export = _write_unbagged(
        tmp_path,
        tf_messages={},
        marker_messages={},
    )

    with pytest.raises(ValueError, match="Duplicate external subject/segment names"):
        load_scene_state(export, scene)


def test_iter_vicon_marker_frames(tmp_path: Path) -> None:
    export = _write_unbagged(
        tmp_path,
        tf_messages={},
        marker_messages={
            "2026-06-07T16:55:32.000000": _marker_message(
                stamp_sec=1,
                stamp_nsec=0,
                marker_name="heel",
                translation=(0.9, 0.1, 0.0),
            ),
            "2026-06-07T16:55:32.100000": _marker_message(
                stamp_sec=1,
                stamp_nsec=100_000_000,
                marker_name="heel",
                translation=(0.9, 0.1, 0.1),
                occluded=True,
            ),
        },
    )

    frames = list(iter_vicon_marker_frames(export))
    assert len(frames) == 2
    assert frames[0].stamp_seconds == pytest.approx(1.0)
    positions = marker_positions_by_name(
        frames[0],
        subject_name=_SubjectId.LEFT_SHOE.label,
        segment_name=_SegmentId.LEFT_SHOE.label,
    )
    np.testing.assert_allclose(positions["heel"], np.array([0.9, 0.1, 0.0]))
    assert "heel" not in marker_positions_by_name(frames[1])
