# Handles

Some references are segment-scoped, but since we implement all vocabularies as enums, an enum member alone may not be globally meaningful. For example, there may be two subjects each having a segment with a marker named "heel". So, we use handles to refer to segment-scope objects, namely markers (`MarkerHandle`) and patches (`PatchHandle`).

A handle is a typed symbolic reference to some object. It is not the object itself, i.e. a handle does not store geometry, observed data, or world-frame state. It only identifies which marker/patch/etc is being referred to.

Targets are the scene-level identities built from those local handles. Handles are local/spec-relative; targets are what contact tracks, runtime lookups, and serialization use when they need a concrete scene reference.

Handles are not forced to have any shared class structure, but they are always named such that they end with "Handle".

For the sake of consistent naming, "handles" are segment-scoped symbolic references, while "keys" are concrete runtime lookup identities. See `keys.md`.
