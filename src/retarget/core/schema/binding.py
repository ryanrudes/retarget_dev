"""Path-binding and runtime attachment for authored scene schemas."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
from fungeom import Face, Point3

from retarget.core.geometry import segment_geometry_at
from retarget.core.keys import SegmentKey
from retarget.core.schema.base import (
    _rebuild_schema,
    _schema_fields,
    _schema_items,
    _schema_values,
)
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
    return _rebuild_schema(
        subjects,
        lambda name, subject: _bind_subject(subject, name=name, runtimes=runtimes),
    )


def _bind_subject(
    subject: Subject[Any],
    *,
    name: str,
    runtimes: Mapping[SegmentKey, _SegmentRuntime] | None,
) -> Subject[Any]:
    bound_segments = _rebuild_schema(
        subject.segments,
        lambda segment_name, segment: _bind_segment(
            segment,
            subject=name,
            segment_name=segment_name,
            body_model=subject.body_model,
            runtime=(None if runtimes is None else runtimes.get(SegmentKey(name, segment_name))),
        ),
    )
    bound = Subject(
        segments=bound_segments,
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
    bound_markers = _rebuild_schema(
        segment.markers,
        lambda marker_name, marker: _bind_marker(
            marker,
            subject=subject,
            segment=segment_name,
            marker_name=marker_name,
            body_model=body_model,
            runtime=runtime,
        ),
    )
    marker_positions_segment = {
        marker_name: marker.position_segment
        for marker_name, marker in _schema_items(bound_markers)
        if marker.position_segment is not None
    }
    # Free-variable environment for data-authored patches: each authored Marker (the identity its
    # ``.rest`` free variable carries) -> its segment-frame rest Point3. Keyed by the *authored*
    # markers (what the Face references); valued from the bound positions (filled from body_model).
    # Authored and bound markers share field order, so zip pairs each authored identity with its
    # bound position.
    marker_env: dict[Hashable, Point3] = {}
    for authored_marker, bound_marker in zip(
        _schema_values(segment.markers), _schema_values(bound_markers), strict=True
    ):
        position = bound_marker.position_segment
        if position is not None:
            coords = np.asarray(position, dtype=np.float64).reshape(3)
            marker_env[authored_marker] = Point3.at(float(coords[0]), float(coords[1]), float(coords[2]))
    bound_patches = _rebuild_schema(
        segment.patches,
        lambda patch_name, patch: _bind_patch(
            patch,
            subject=subject,
            segment=segment_name,
            patch_name=patch_name,
            marker_positions_segment=marker_positions_segment,
            marker_env=marker_env,
            runtime=runtime,
        ),
    )
    bound = Segment(
        markers=bound_markers,
        patches=bound_patches,
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
    marker_env: Mapping[Hashable, Point3],
    runtime: _SegmentRuntime | None,
) -> Patch:
    bound_face = None
    geometry = patch.geometry
    if isinstance(geometry, Face):
        # open patch algebra (data form): a Face authored over free-variable markers (``m.rest``);
        # bind substitutes each free with its segment-frame rest position -> the segment-local Face.
        bound_face = geometry.bind(marker_env)
    elif geometry is not None:
        # open patch algebra (callable form): evaluate over the segment's bind-time geometry; the
        # resulting Face is the patch geometry, transported per-frame at query time.
        seg_geometry = segment_geometry_at(SegmentTarget(subject=subject, segment=segment), marker_positions_segment)
        bound_face = geometry(seg_geometry)
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


def _validate_subjects(subjects: Subjects) -> None:
    if not _schema_fields(subjects):
        raise ValueError("A scene must declare at least one subject")
    for subject_name, subject in _schema_items(subjects):
        for segment_name, segment in _schema_items(subject.segments):
            _validate_segment(subject_name, segment_name, segment)


def _validate_segment(
    subject_name: str,
    segment_name: str,
    segment: Segment[Any, Any],
) -> None:
    seen_vicon: dict[str, str] = {}
    for marker_name, marker in _schema_items(segment.markers):
        previous = seen_vicon.get(marker.mocap_name)
        if previous is not None:
            raise ValueError(
                "Duplicate Marker.mocap_name within "
                f"{subject_name!r}/{segment_name!r}: {marker.mocap_name!r} used by "
                f"{previous!r} and {marker_name!r}"
            )
        seen_vicon[marker.mocap_name] = marker_name
