# fungeom handover — free-variable leaves + `resolve(env)`

**The keystone for "the scene is one resolvable graph."** retarget wants to author a contact
patch as **fungeom data** — a `Face` whose plane/footprint points are *markers that have no position
until bind time* — instead of today's `Callable[[SegmentGeometry], Face]`. That needs exactly one
fungeom capability: **free-variable leaves** (a `Point3` with no value yet, carrying an identity)
that compose through the normal algebra and are filled in by **`resolve(env)`** at bind time.

## Proven first (so this isn't speculative)

`docs/fungeom-free-variables-spike.py` stands in for this capability with a ~25-line `Deferred`
proxy, rewrites retarget's real `_sole` patch as data over typed `Marker` symbols, and resolves it.
Result, through retarget's actual pipeline (`face_signal → frame()/boundary()`):

```
[1] resolved Face is IDENTICAL to the callable Face   (frame() match: True   boundary() match: True)
[2] unbound marker  -> resolve fails naming the missing var   (would be Unresolvable)
[3] misspelled symbol -> NameError  (statically a mypy [name-defined] error)
```

The spike calls the **real** fungeom ops at resolve time, so the evidence that `fit_plane`/`in_frame`/
`hull`/`Face.on` produce the correct result over substituted free points is solid. What it does *not*
prove is fungeom's *native* free-leaf ergonomics — that's what this asks you to build.

## Why retarget needs it (the problem being solved)

Today a patch is an imperative callable, and inside it markers are stringly-typed:

```python
def sole(seg) -> Face:                                   # imperative island in a declarative schema
    plane = seg.markers["plane_rear", "plane_inner", "plane_outer"].fit_plane()   # typo = silent
    return Face.on(plane, Region2.hull(seg.markers["heel", "toe"].in_frame(plane)).offset(0.005))
Patch(geometry=sole)
```

Both warts have one root cause: you can't express a **late-bound, typed reference** to a marker, so
retarget uses a *function* (to defer) and a *string* (to name). Give fungeom free variables and the
patch becomes pure typed data, passable as an object:

```python
plane = bundle(plane_rear, plane_inner, plane_outer).fit_plane()      # plane_rear… are typed symbols
sole  = Face.on(plane, Region2.hull(bundle(heel, toe).in_frame(plane)).offset(0.005))
Patch(face=sole)                                                       # data; typo = NameError
```

A free marker is just a fungeom leaf that is `Unresolvable` until bound — which is *exactly* fungeom's
partiality model. So this isn't a bolt-on; it's "the unknown is a first-class leaf."

## The capability

### 1. Free-variable leaves
`Point3.free(identity)` → a `Point3` resolver, `Unresolvable` on its own, tagged with an opaque
**hashable `identity`**. It stays typed as `Point3`; only its resolvability differs. (retarget needs
`Point3` frees now; see Open Questions for generalizing to `Vec3`/`Scalar`/`Transform`.)

### 2. They compose through the algebra
A graph built over free leaves is itself a valid, typed, `Unresolvable` resolver. The exact op-set
retarget composes over free markers (from its two real patch authors, `examples/_shared/scene.py`):

| produces | ops that must accept free leaves |
|---|---|
| `Point3Bundle` | constructed from (free) `Point3`s; `.fit_plane()`, `.in_frame(plane)` |
| `Plane`        | `.facing(point)`, `.flipped()`, `.offset(scalar)` |
| `Region2`      | `Region2.hull(points_in_plane)`, `Region2.rectangle(w, h)`, `.offset(scalar)` |
| `Face`         | `Face.on(plane, region)` |

### 3. `resolve(env)` / `decide(env)`
`graph.resolve(env)` where `env: Mapping[identity, Point3 | value]` substitutes each free leaf, then
resolves. `graph.decide(env)` returns `Resolvable(value)` if every reachable free is bound, else
`Unresolvable` **naming the still-free identities**. With no env / missing frees → `Unresolvable`,
extended with a `free variable <identity> unbound` reason so partiality stays self-describing.

### Identity is object identity, not a string
That is the whole point — it's what removes the stringly-typed keys. retarget will pass the `Marker`
object (or a token the `Marker` owns) as `identity`, and key `env` by it. An opaque `Hashable`
identity is enough; fungeom needn't know it's a marker.

## How retarget consumes it (informative — not your concern, but grounds the API)

- `Marker` exposes `Point3.free(self)` — its segment-frame rest position as a free var.
- `Patch.geometry` widens to accept a `Face` (built over free marker leaves) **or** the existing
  callable, with the `Face` form becoming primary.
- Bind builds `env = {marker: Point3.at(*marker.position_segment) for markers}` — data retarget
  already gathers in `core.geometry.segment_geometry` — then `face.resolve(env)` → the segment-local
  `Face`, stored and transported by the segment pose **exactly as today**. The `FaceSignal` transport
  and every patch query are unchanged (the spike confirms identical outputs).

## Acceptance criteria (lift straight from the spike)

1. A `Face` over free leaves, `resolve`d in an env of their positions, is identical to the same
   `Face` built from concrete points — identical `plane()`, `region()`, `boundary()`.
2. `resolve`/`decide` with a missing free → `Unresolvable` naming that free.
3. Every op in the table accepts free leaves and resolves correctly (`fit_plane` over 3 free points
   resolving to the plane through their bound positions is the canonical case).

## Open questions for the fungeom session

- **Eager vs lazy.** The spike substituted-then-called (eager). Native fungeom should let you build
  `…fit_plane()` over free points as a *lazy graph* and `resolve(env)` later — which fits the
  "lazy graph of resolvers" architecture (a free var is a leaf that's `Unresolvable` without env).
  Confirm lazy-graph-with-free-leaves is the intended model.
- **Identity type.** Opaque `Hashable` token vs a `Var`/`Free` wrapper object. retarget prefers to
  pass the `Marker` object directly as an opaque identity.
- **Generality.** `free()` on every primitive, or just `Point3` now? Only `Point3` is needed today;
  a free `Scalar` (a calibration offset) is the most plausible next.
- **API surface.** `graph.resolve(env)` vs `graph.bind(env).resolve()` vs an env threaded through
  `decide()`. retarget only needs *a* way to resolve a graph against an identity→value map.

## Scope boundary

This is **only** the geometry-as-data keystone (fungeom side). The retarget-side question of whether
the *whole schema* moves from `TypedDict` subscription (`markers["heel"]`) to typed attribute-symbols
(`markers.heel`) is a **separate, later retarget decision** and is **not** required here: patches-as-data
already works with markers referenced as local symbol objects under today's schema (the spike does
exactly that). Land this, and retarget can retire the patch callable and the in-callable string keys
independently of any larger schema change.
