# Identity and targets

The scene model is hierarchical:

```text
Scene
  Subject
    Segment
      Marker
      Patch
```

A marker or patch is not a global name — it belongs to a segment, which belongs
to a subject. This project represents that with **authored string names** for
identity and **string-based targets** for stable scene-level references. There
are no identifier enums and no handle layer.

## Authoring is the identity

Scene structure is authored with `TypedDict` schemas plus frozen dataclasses:

```python
class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker

class ShoePatches(Patches):
    sole: Patch

class ShoeSegments(Segments):
    shoe: Segment[ShoeMarkers, ShoePatches]

class MocapSubjects(Subjects):
    left_shoe: Subject[ShoeSegments]
    right_shoe: Subject[ShoeSegments]
```

The authored field names (`"left_shoe"`, `"shoe"`, `"heel"`, `"sole"`) are the
canonical identity. `Marker.mocap_name`, `Segment.mocap_name`, and
`Subject.mocap_name` are external (Vicon) lookup metadata; `Patch.label` /
`Patch.frame` are display metadata, not identity.

Structurally identical subjects (e.g. both shoes) reuse one schema.

## Dual-purpose dataclasses

`Marker` / `Patch` / `Segment` / `Subject` are *both* the authoring values and
the bound runtime query surface. `bind_scene(subjects)` path-binds them
(targets + geometry available); loading data
(`MocapTrack.from_unbagged(root, subjects)`) binds them to time series. The link
is a private, non-init `_binding` excluded from equality/repr, so the public
constructors stay pure authoring.

## Traversal and queries

```python
scene = bind_scene(subjects)
shoe = scene["left_shoe"].segments["shoe"]
shoe.marker_target("heel")        # MarkerTarget("left_shoe", "shoe", "heel")
shoe.patch_target("sole")         # PatchTarget("left_shoe", "shoe", "sole")

mocap = demo.tracks["mocap"]
shoe = mocap.subjects["left_shoe"].segments["shoe"]
shoe.markers["heel"].positions()
shoe.patches["sole"].points()
shoe.translations()
```

A declared (geometry-less) patch is still targetable via `patch_target(...)`;
geometry access (`patches[...].points()`/`.normals()`) requires a calibrated
patch.

## Runtime targets

When references must be stored or indexed (dict keys, contact-track keys,
serialized metadata), use string-based targets:

```text
SegmentTarget = subject + segment
MarkerTarget  = subject + segment + marker
PatchTarget   = subject + segment + patch
SegmentKey    = subject + segment   (SceneState pose key)
```

```python
PatchTarget(subject="left_shoe", segment="shoe", patch="sole")
```

Contact tracks key off `PatchTarget`; scene pose state off `SegmentKey`.

## Why this types perfectly

Concrete `TypedDict` subclasses project literal keys to their declared field
types. Because `Demonstration[TracksT].tracks` is `TracksT` and
`MocapTrack[SubjectsT].subjects` is `SubjectsT`, the whole chain is statically
typed without dependent typing, overloads, or codegen:

```python
demo.tracks["mocap"]                          # MocapTrack[MocapSubjects]
mocap.subjects["left_shoe"].segments["shoe"]  # Segment[ShoeMarkers, ShoePatches]
segment.markers["heel"].positions()           # TimeVec3
```

The earlier enum/handle/spec model and its "dependent typing is impossible"
caveat no longer apply: literal-key `TypedDict` projection gives the deep chain
its static types directly.
