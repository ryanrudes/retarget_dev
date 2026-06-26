"""Typed scene authoring schema and the bound runtime query surface.

This package defines a single set of dual-purpose types:

* ``Markers`` / ``Patches`` / ``Segments`` / ``Subjects`` are ``TypedDict`` bases
  the user subclasses to declare the *shape* of a scene. Because each subclass is
  a concrete ``TypedDict``, literal-key access projects to the declared field
  type, which is what gives the deep query chain perfect static typing.
* ``Marker`` / ``Patch`` / ``Segment`` / ``Subject`` are frozen dataclasses the
  user instantiates to author concrete scene data. The same instances double as
  the runtime query surface: once a :class:`~retarget.demo.mocap.MocapTrack`
  binds them to loaded data, ``marker.positions()``/``segment.translations()``/
  ``patch.points()`` answer time-series queries.

Authoring objects carry a private, non-init ``_binding`` that links them to
loaded track data. It is ``None`` while authoring and ignored by equality/repr,
so the public constructors stay pure authoring (``Marker(mocap_name=...)``).
"""

from retarget.core.schema.marker import Marker, Markers
from retarget.core.schema.patch import Patch, Patches
from retarget.core.schema.segment import Segment, Segments, _SegmentRuntime
from retarget.core.schema.binding import bind_scene, bind_subjects_runtime
from retarget.core.schema.subject import Subject, Subjects

__all__ = [
    "Markers",
    "Patches",
    "Segments",
    "Subjects",
    "Marker",
    "Patch",
    "Segment",
    "Subject",
    "bind_scene",
    "bind_subjects_runtime",
    "_SegmentRuntime",
]
