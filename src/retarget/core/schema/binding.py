"""Path-binding and runtime attachment for authored scene schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from retarget.core.geometry import segment_geometry_at
from retarget.core.keys import SegmentKey
from retarget.core.schema.marker import Marker, _MarkerBinding
from retarget.core.schema.patch import Patch, _PatchBinding
from retarget.core.schema.segment import Segment, _SegmentBinding, _SegmentRuntime
from retarget.core.schema.subject import Subject, Subjects, _SubjectBinding
from retarget.core.targets import SegmentTarget
from retarget.core.types import Vec3


def bind_scene[SubjectsT: Subjects](subjects: SubjectsT) -> SubjectsT:
    """Path-bind an authored scene so targets/geometry are available.

    Returns a structure of the same ``SubjectsT`` type whose subjects/segments/
    markers/patches know their compiled names (enabling ``*_target(...)`` and
    geometry) but are not yet bound to time-series data.
    """
    _validate_subjects(subjects)
    return _bind_subjects(subjects, runtimes=None)


def bind_subjects_runtime[SubjectsT: Subjects](
    subjects: SubjectsT,
    runtimes: Mapping[SegmentKey, _SegmentRuntime],
) -> SubjectsT:
    """Bind an authored scene to per-segment runtime data (used by MocapTrack)."""
    return _bind_subjects(subjects, runtimes=runtimes)


def _bind_subjects[SubjectsT: Subjects](
    subjects: SubjectsT,
    *,
    runtimes: Mapping[SegmentKey, _SegmentRuntime] | None,
) -> SubjectsT:
    bound = {
        name: _bind_subject(subject, name=name, runtimes=runtimes)
        for name, subject in cast(Mapping[str, Subject[Any]], subjects).items()
    }
    return cast(SubjectsT, bound)


def _bind_subject(
    subject: Subject[Any],
    *,
    name: str,
    runtimes: Mapping[SegmentKey, _SegmentRuntime] | None,
) -> Subject[Any]:
    bound_segments = {
        segment_name: _bind_segment(
            segment,
            subject=name,
            segment_name=segment_name,
            body_model=subject.body_model,
            runtime=(None if runtimes is None else runtimes.get(SegmentKey(name, segment_name))),
        )
        for segment_name, segment in subject.segments.items()
    }
    bound = Subject(
        segments=cast(Any, bound_segments),
        mocap_name=subject.mocap_name,
        body_model=subject.body_model,
    )
    object.__setattr__(bound, "_binding", _SubjectBinding(subject=name))
    return bound


def _bind_segment(
    segment: Segment[Any, Any],
    *,
    subject: str,
    segment_name: str,
    body_model: Mapping[str, Vec3] | None,
    runtime: _SegmentRuntime | None,
) -> Segment[Any, Any]:
    bound_markers = {
        marker_name: _bind_marker(
            marker,
            subject=subject,
            segment=segment_name,
            marker_name=marker_name,
            body_model=body_model,
            runtime=runtime,
        )
        for marker_name, marker in segment.markers.items()
    }
    marker_positions_segment = {
        marker_name: marker.position_segment
        for marker_name, marker in bound_markers.items()
        if marker.position_segment is not None
    }
    bound_patches = {
        patch_name: _bind_patch(
            patch,
            subject=subject,
            segment=segment_name,
            patch_name=patch_name,
            marker_positions_segment=marker_positions_segment,
            runtime=runtime,
        )
        for patch_name, patch in segment.patches.items()
    }
    bound = Segment(
        markers=cast(Any, bound_markers),
        patches=cast(Any, bound_patches),
        mocap_name=segment.mocap_name,
    )
    object.__setattr__(
        bound,
        "_binding",
        _SegmentBinding(subject=subject, segment=segment_name, runtime=runtime),
    )
    return bound


def _bind_marker(
    marker: Marker,
    *,
    subject: str,
    segment: str,
    marker_name: str,
    body_model: Mapping[str, Vec3] | None,
    runtime: _SegmentRuntime | None,
) -> Marker:
    position_segment = marker.position_segment
    if position_segment is None and body_model is not None:
        position_segment = body_model.get(marker.mocap_name)
    bound = Marker(
        mocap_name=marker.mocap_name,
        position_segment=position_segment,
    )
    object.__setattr__(
        bound,
        "_binding",
        _MarkerBinding(subject=subject, segment=segment, marker=marker_name, runtime=runtime),
    )
    return bound


def _bind_patch(
    patch: Patch,
    *,
    subject: str,
    segment: str,
    patch_name: str,
    marker_positions_segment: Mapping[str, Vec3],
    runtime: _SegmentRuntime | None,
) -> Patch:
    bound_face = None
    if patch.geometry is not None:
        # open patch algebra: evaluate the geometry callable over the segment's bind-time
        # geometry; the resulting Face is the patch geometry, transported per-frame at query time.
        seg_geometry = segment_geometry_at(SegmentTarget(subject=subject, segment=segment), marker_positions_segment)
        bound_face = patch.geometry(seg_geometry)
    bound = Patch(label=patch.label, frame=patch.frame, geometry=patch.geometry)
    object.__setattr__(
        bound,
        "_binding",
        _PatchBinding(
            subject=subject,
            segment=segment,
            patch=patch_name,
            runtime=runtime,
            face=bound_face,
        ),
    )
    return bound


def _validate_subjects(subjects: Mapping[str, Any]) -> None:
    if not subjects:
        raise ValueError("A scene must declare at least one subject")
    for subject_name, subject in subjects.items():
        if not subject_name:
            raise ValueError("Subject names must be non-empty")
        segments = cast(Mapping[str, Segment[Any, Any]], subject.segments)
        for segment_name, segment in segments.items():
            _validate_segment(subject_name, segment_name, segment)


def _validate_segment(
    subject_name: str,
    segment_name: str,
    segment: Segment[Any, Any],
) -> None:
    seen_vicon: dict[str, str] = {}
    for marker_name, marker in segment.markers.items():
        previous = seen_vicon.get(marker.mocap_name)
        if previous is not None:
            raise ValueError(
                "Duplicate Marker.mocap_name within "
                f"{subject_name!r}/{segment_name!r}: {marker.mocap_name!r} used by "
                f"{previous!r} and {marker_name!r}"
            )
        seen_vicon[marker.mocap_name] = marker_name
