# Handles and Targets

This project uses typed enumerations as symbolic identifiers for subjects,
segments, markers, patches, and tracks. These IDs are safer than raw strings or
integers because they let type checkers catch many accidental mixups before
runtime.

The hard part is that the physical model is hierarchical:

```text
Scene
  Subject
    Segment
      Marker
      Patch
```

A marker or patch is not just a global name. It belongs to a segment, and that
segment belongs to a subject. This means we need to represent both **local
vocabulary** and **scene-level identity** without making the public API a maze
of tuple keys, handles, and tiny bureaucratic artifacts.

## The core tension

There are two tempting extremes.

The first is to define a separate enum for every level of every subject:

```python
class MySubjects(SubjectId):
    SUBJECT_A = "subjectA"
    SUBJECT_B = "subjectB"

class SubjectASegments(SegmentId):
    SEGMENT_1 = "segment1"
    SEGMENT_2 = "segment2"

class SubjectBSegments(SegmentId):
    SEGMENT_1 = "segment1"
    SEGMENT_3 = "segment3"
```

This can model each subject very precisely, but Python cannot automatically
express the dependent type relationship:

```text
if subject == SUBJECT_A, then valid segments are SubjectASegments
if subject == SUBJECT_B, then valid segments are SubjectBSegments
```

That kind of value-dependent relationship is called **dependent typing**.
Python's type system does not support it directly. We can approximate it with
`Literal` overloads or generated accessors, but decorators and runtime
registries cannot make a normal Python type checker infer this relationship.
Lovely little limitation, very on brand.

The second extreme is to use one shared enum for all subjects, segments,
markers, and patches:

```python
class MySubjects(SubjectId):
    SUBJECT_A = "subjectA"
    SUBJECT_B = "subjectB"

class MySegments(SegmentId):
    SEGMENT_1 = "segment1"
    SEGMENT_2 = "segment2"
    SEGMENT_3 = "segment3"
```

This is easier to type, but it loses information. A type checker can no longer
distinguish a robot segment ID from a shoe segment ID if both live in the same
broad enum family.

## The practical model

The repository uses a middle path:

1. **IDs** name local vocabulary items.
2. **Structured specs** describe containment.
3. **Runtime targets** identify concrete scene-level things when references
   need to be stored or indexed outside normal traversal.

In other words:

```text
ID      = local symbolic vocabulary
Spec    = structured containment and validation
Target  = concrete scene-level reference
```

This gives us strong practical typing without pretending Python is a proof
assistant wearing a hoodie.

## TypedDict authoring layer

New scene definitions should normally start with `Subjects`, `Segments`,
`Markers`, and `Patches`, then compile into runtime specs with `build_scene(...)`.
That authoring layer is the public path.

Manual `SceneSpec`, `SubjectSpec`, and `SegmentSpec` construction still exists
for backend-oriented loader and geometry code, but it is a low-level path, not
the preferred public API.

`build_scene(...)` may synthesize private runtime ID classes from the authored
field names. Those generated IDs are implementation details of the compiled
scene, not part of the public authoring API.

## IDs

IDs are typed symbolic names:

```python
class ViconSubjectId(SubjectId):
    LEFT_SHOE = "left_shoe"
    RIGHT_SHOE = "right_shoe"

class ShoeSegmentId(SegmentId):
    SHOE = "shoe"

class ShoeMarkerId(MarkerId):
    HEEL = "heel"
    TOE = "toe"

class ShoePatchId(PatchId):
    SOLE = "sole"
```

The preferred pattern is:

- one `SubjectId` enum per scene or project;
- one `SegmentId` enum per object family when possible;
- one `MarkerId` enum per segment family;
- one `PatchId` enum per segment family.

For example, a left shoe and right shoe should usually share `ShoeSegmentId`,
`ShoeMarkerId`, and `ShoePatchId`. They are different subjects, not different
vocabularies, unless their segment/marker/patch schemas really differ.

This lets the type checker catch broad vocabulary mistakes, such as passing a
hand marker enum into a shoe segment spec.

Authored field names are the internal canonical IDs. `Marker.vicon_name` is
external lookup metadata, and `Patch.label` / `Patch.frame` are metadata, not
identity.

## Structured specs

Specs carry the actual containment structure:

```text
SceneSpec
  SubjectSpec
    SegmentSpec
      MarkerSetSpec
      PatchCalibrationSpec
```

Conceptually, a scene contains subjects, a subject contains segments, and a
segment contains markers and patches. Public traversal should follow that
structure:

```python
scene_view.subject(ViconSubjectId.LEFT_SHOE).segment(ShoeSegmentId.SHOE)
```

or, at the demo layer:

```python
mocap.subject(ViconSubjectId.LEFT_SHOE).segment(ShoeSegmentId.SHOE)
```

This is the preferred user-facing style. Users should not have to assemble
low-level keys just to navigate the scene.

Patch declarations and calibrated patch geometry are separate concerns. A
declared patch can produce a `PatchTarget`, while geometry-heavy access such as
`segment.patch(...)` or `segment.patch_spec(...)` requires calibrated patch
geometry.

Patch APIs follow this split:

- `SegmentSpec.patch(...)` returns a `PatchHandle` for any declared patch.
- `SegmentSpec.patch_spec(...)` requires calibrated patch geometry.
- `SegmentView.patch(...)` returns a geometry-backed `PatchView` and requires
  calibrated patch geometry.
- `SegmentView.patch_target(...)` works for any declared patch.

Exact membership is validated against the `SceneSpec` at runtime. For example,
the type checker may know that `ShoeSegmentId.SHOE` is from the right segment
vocabulary, but the `SceneSpec` is what confirms that a particular subject
actually contains that segment.

## Runtime targets

Sometimes traversal is not enough. We need stable references that can be used
as dictionary keys, contact-track keys, serialized metadata, or alignment
inputs. That is what runtime targets are for.

A target identifies a concrete thing in a scene:

```text
Segment target = subject + segment
Marker target  = subject + segment + marker
Patch target   = subject + segment + patch
```

Current code still contains some legacy names such as `SegmentKey`,
`MarkerHandle`, and `PatchHandle`. The important conceptual split is:

- a **handle** is a local/spec-relative reference;
- a **target** is a concrete scene-level reference;
- a **key** should only be used when we specifically mean dictionary-key
  mechanics, not domain identity.

In practice, `PatchTarget` is used because contact tracks need concrete patch
identities:

```python
PatchTarget(
    subject=ViconSubjectId.LEFT_SHOE,
    handle=PatchHandle(
        segment=ShoeSegmentId.SHOE,
        patch=ShoePatchId.SOLE,
    ),
)
```

That handle-based shape is the stable public runtime model. Do not flatten it
as part of the authoring layer; a flattened form would be a separate runtime
migration.

## Why decorators are not enough

A registry decorator can be useful:

```python
@register(to=MySubjects.SUBJECT_A)
class SubjectASegments(SegmentId):
    SEGMENT_1 = "segment1"
    SEGMENT_2 = "segment2"
```

Such a decorator could attach metadata like:

```python
SubjectASegments.SEGMENT_1.value == "segment1"
SubjectASegments.SEGMENT_1.label == "subjectA:segment1"
SubjectASegments.SEGMENT_1.parent == MySubjects.SUBJECT_A
```

The same idea could apply to markers registered under segments. This is useful
for runtime validation, canonical labels, serialization, debugging, and
introspection.

It does not solve static dependent typing. Normal Python type checkers do not
infer enum-parent relationships from runtime decorator metadata. The decorator
can tell the program what belongs where at runtime; it cannot make Pyright or
mypy prove that a specific subject value implies a specific child enum type.

## What would give near-perfect typing

The closest Python can get is generated or hand-written typed accessors using
`Literal` overloads:

```python
@overload
def subject(
    self,
    subject: Literal[ViconSubjectId.LEFT_SHOE],
) -> SubjectSpec[ViconSubjectId, ShoeSegmentId, ShoeMarkerId, ShoePatchId]: ...

@overload
def subject(
    self,
    subject: Literal[ViconSubjectId.RIGHT_HAND],
) -> SubjectSpec[ViconSubjectId, HandSegmentId, HandMarkerId, HandPatchId]: ...
```

This lets the type checker understand that different subject values return
differently typed subject specs. It is powerful, but hand-writing these
overloads gets tedious quickly. If the project ever needs this level of static
precision, code generation is probably the least-bad option.

## Working rule

The working rule for this project is:

```text
Use structured traversal for normal user code.
Use runtime targets when references must be stored or indexed.
Use generics to prevent wrong vocabulary-family mixing.
Use SceneSpec validation to catch exact membership errors.
Do not chase full dependent typing in handwritten Python.
```

This gives most of the safety we want while keeping the API usable. The last
five percent of type perfection is possible only with overloads, generated
accessors, or a different language. The codebase has enough problems without
becoming a type-theory escape room.
