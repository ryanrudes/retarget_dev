"""THROWAWAY Phase 2 spike: does an attribute-SYMBOL schema preserve the typed deep chain that the
current TypedDict-subscription schema gives, with NO codegen? And does it close the symbol loop
(patches + queries reference markers.heel as typed symbols, no strings anywhere)?

Run: uv run --extra dev mypy --strict <thisfile>   (must be clean for the static guarantee to hold)
     uv run python <thisfile>                       (runtime construction must work)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, assert_type


@dataclass(frozen=True)
class Marker:
    name: str
    bound: bool = False

    @property
    def rest(self) -> str:  # stands in for the fungeom free Point3 identified by this marker
        return f"free({self.name})"


@dataclass(frozen=True)
class Patch:
    label: str
    geometry: Any = None  # a fungeom Face built over markers[...].rest -- typed symbols, no strings


# ---- generic container primitives (parameterized over the user's *class* schemas) -----------------
@dataclass(frozen=True)
class Segment[MarkersT, PatchesT]:
    markers: MarkersT
    patches: PatchesT


@dataclass(frozen=True)
class Subject[SegmentsT]:
    segments: SegmentsT


@dataclass(frozen=True)
class MocapTrack[SubjectsT]:
    subjects: SubjectsT


@dataclass(frozen=True)
class Demonstration[TracksT]:
    tracks: TracksT


# ---- the USER's concrete schema: plain dataclasses with typed attributes (the "symbols") ----------
@dataclass(frozen=True)
class ShoeMarkers:
    heel: Marker
    toe: Marker


@dataclass(frozen=True)
class ShoePatches:
    sole: Patch


@dataclass(frozen=True)
class ShoeSegments:
    shoe: Segment[ShoeMarkers, ShoePatches]


@dataclass(frozen=True)
class MocapSubjects:
    left_shoe: Subject[ShoeSegments]


@dataclass(frozen=True)
class GroundTracks:
    mocap: MocapTrack[MocapSubjects]


# ---- authoring: markers ARE the symbols; the patch references them directly, no string, no callable
heel = Marker("heel")
toe = Marker("toe")
sole = Patch(label="sole", geometry=(heel.rest, toe.rest))  # `markers.heel.rest` at author time
shoe_segment = Segment(markers=ShoeMarkers(heel=heel, toe=toe), patches=ShoePatches(sole=sole))
demo = Demonstration(GroundTracks(mocap=MocapTrack(MocapSubjects(left_shoe=Subject(ShoeSegments(shoe=shoe_segment))))))


# ====== Q1/Q2: the typed deep chain via ATTRIBUTE access (the make-or-break) =======================
assert_type(demo.tracks, GroundTracks)
assert_type(demo.tracks.mocap, MocapTrack[MocapSubjects])
assert_type(demo.tracks.mocap.subjects, MocapSubjects)
assert_type(demo.tracks.mocap.subjects.left_shoe, Subject[ShoeSegments])
assert_type(demo.tracks.mocap.subjects.left_shoe.segments.shoe, Segment[ShoeMarkers, ShoePatches])
assert_type(demo.tracks.mocap.subjects.left_shoe.segments.shoe.markers.heel, Marker)
assert_type(demo.tracks.mocap.subjects.left_shoe.segments.shoe.patches.sole, Patch)
# the symbol-loop closure: a query reads a typed symbol, not a string key
assert_type(demo.tracks.mocap.subjects.left_shoe.segments.shoe.markers.heel.rest, str)


# ====== Q3: can binding reconstruct a bound schema GENERICALLY (the runtime cost)? =================
def bind_markers[T](authored: T) -> T:
    """Rebuild any frozen marker-schema dataclass with each Marker field marked bound -- the generic
    equivalent of today's `{name: bound_marker}` dict construction, but for a class."""
    bound_fields = {}
    for f in fields(authored):  # type: ignore[arg-type]  # authored is a dataclass instance
        value = getattr(authored, f.name)
        bound_fields[f.name] = Marker(value.name, bound=True) if isinstance(value, Marker) else value
    return type(authored)(**bound_fields)


bound = bind_markers(ShoeMarkers(heel=heel, toe=toe))
assert_type(bound, ShoeMarkers)  # the generic rebuild preserves the concrete type
assert bound.heel.bound and bound.heel.name == "heel"


if __name__ == "__main__":
    print("runtime deep chain:", demo.tracks.mocap.subjects.left_shoe.segments.shoe.markers.heel)
    print("generic rebuild kept type + bound the markers:", bound)
    print("PHASE 2 SPIKE RAN.")
