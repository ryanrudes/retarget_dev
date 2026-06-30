# fungeom handover: coerce `.rest`-bearing symbols to `Point3` at point boundaries

**Status:** spec for a fungeom session (like `docs/fungeom-free-variables-spec.md`, which produced 0.4.0).
This is retarget polish item (1) of the two that survived the scene-model design review (see the
`scene-model-validated` memory). It is a genuine, *cross-cutting* fungeom capability — there is no single
point-acceptance chokepoint in fungeom today — so it belongs in a fungeom session with the 100%-coverage
harness, not a retarget-side hack.

> **✅ SHIPPED by the fungeom session (2026-06-29).** Implemented exactly as proposed — the
> structural protocol **`SupportsPoint3`** with **`__fungeom_point3__(self) -> Point3`**, plus an
> internal `_as_point3` / `_as_point3s` applied at **every** public point boundary across `Point3`,
> `Line`, `Ray`, `Plane`, `Segment`, `Face`, and `Point3Bundle`. `SupportsPoint3` is exported from
> `fungeom`. A coverage-backed **guard test** fails if any public `Point3` parameter is ever left
> un-widened. Gate green: ruff, mypy --strict (485 files), pytest **100% coverage**. Two notes for
> the retarget follow-up below:
> - **`PlaneSignal.facing` does not exist** — only the static `Plane.facing` (now widened). The
>   signal facades accept *no* static `Point3` (their constructors take raw float coords / arrays),
>   so nothing there needed widening.
> - **Bonus boundaries** the guard surfaced and that were also widened: `Point3Bundle.distances_to`
>   / `closest_point_to` / `nearest_to` — so `cloud.nearest_to(marker)` works directly. The
>   `Point3Bundle.map_scalar` / `map_point` / `map_vec3` callbacks were **deliberately left bare**:
>   their `Callable[[Point3], …]` *receives* a real member, it is not a point the caller supplies.
>
> retarget: implement `Marker.__fungeom_point3__` (return `self.rest`) and proceed with §"retarget
> follow-up" — nothing in the proposed protocol changed.

## Goal

Retarget authors patch geometry as fungeom **data** over marker symbols. Today every point must be threaded
through `.rest` (the marker's segment-frame rest position, a `Point3.free`):

```python
Point3Bundle.of([m.plane_rear.rest, m.plane_inner.rest, m.plane_outer.rest]).fit_plane().facing(m.toe_grid_1.rest)
Region2.hull(Point3Bundle.of([m.heel.rest, m.toe.rest, ...]).in_frame(plane))
```

We want the marker symbols accepted **directly**, so the `.rest` noise disappears:

```python
Point3Bundle.of([m.plane_rear, m.plane_inner, m.plane_outer]).fit_plane().facing(m.toe_grid_1)
Region2.hull(Point3Bundle.of([m.heel, m.toe, ...]).in_frame(plane))
```

In short: **anywhere fungeom accepts a `Point3`, it should also accept an object that knows how to produce
one.** ("`.rest`-bearing markers" = objects implementing the protocol below.)

## Mechanism: a structural protocol + one coercion helper

fungeom must stay ignorant of retarget. Define a fungeom protocol and a single internal coercion:

```python
@runtime_checkable
class SupportsPoint3(Protocol):
    def __fungeom_point3__(self) -> Point3: ...

def _as_point3(x: Point3 | SupportsPoint3) -> Point3:
    return x if isinstance(x, Point3) else x.__fungeom_point3__()
```

Apply `_as_point3` at every public boundary that accepts a `Point3` (single or a `Sequence`), and widen those
parameter annotations to `Point3 | SupportsPoint3`. retarget opts in with one method on `Marker`:

```python
def __fungeom_point3__(self) -> Point3:
    return self.rest   # == Point3.free(self)
```

Prefer the `isinstance(x, Point3)`-else-call-the-dunder form over `isinstance(x, SupportsPoint3)` per call.

## Boundaries to widen (the point-accepting public surface)

From retarget's actual usage plus the general surface — **audit every `: Point3` / `: Sequence[Point3]`
parameter** and route it through `_as_point3` uniformly. Known sites that retarget exercises:

- `Point3Bundle.of(points: Sequence[Point3 | SupportsPoint3], keys=None)` — the primary one (plane fit + hull).
- `Plane` / `PlaneSignal.facing(point)` — single point.
- `Region2.hull(...)` — retarget passes a `Point3Bundle`, but accept loose points too if the API allows.
- Anything else taking a `Point3` (`midpoint`, `Segment.through`, `Line.through`, …).

Recommendation: centralize so a *future* point API can't forget to coerce — one `_as_point3` / `_as_point3s`
used everywhere, and consider a test that asserts no public `Point3` parameter rejects a `SupportsPoint3`.

## Typing + the 100% gate

- mypy strict: the widened `Point3 | SupportsPoint3` unions must resolve; `_as_point3` narrows them.
- Coverage: hit each widened boundary with **both** a real `Point3` and a `SupportsPoint3` stub.
- Keep it `Point3`-only for now (markers → points). Do **not** generalize to `Vec2`/`Vec3`/`Point2`/
  `Direction3` — there is no retarget need, and it widens the blast radius.

## Open decision for the fungeom session

**Resolved (2026-06-29):** kept the proposed `__fungeom_point3__` / `SupportsPoint3` — it matches fungeom's
`os.fspath`-style coercion idiom (a structural dunder + a private `_as_point3` helper), reads cleanly at the
boundaries, and namespaces safely. The coercing-classmethod alternative was not taken.

The dunder name `__fungeom_point3__` is a proposal; pick whatever matches fungeom's facade idiom (a coercing
classmethod `Point3.of(x: Point3 | SupportsPoint3)` is a reasonable alternative). retarget only needs *some*
stable protocol to implement — name it however fungeom prefers and note it back here.

## retarget follow-up (once fungeom ships it)

> **✅ DONE (retarget, on fungeom 0.6.0).** Bumped `fungeom>=0.6.0` (lock + sync); added
> `Marker.__fungeom_point3__` (returns `.rest`); dropped `.rest` across `examples/_shared/scene.py`
> (no backend specs authored geometry over `.rest`); added the markers-direct ≡ `.rest` equivalence
> test in `tests/test_patch_geometry_data.py`; refreshed the `AGENTS.md` snippet + the `Marker.rest`
> / `Patch` docstrings. Gate green: pytest, mypy --strict (57 src + scene/tests/deep-chain), ruff.

1. Add `Marker.__fungeom_point3__` (returns `self.rest`) in `src/retarget/core/schema/marker.py`.
2. Drop `.rest` in `examples/_shared/scene.py` (and any backend specs): `Point3Bundle.of([m.heel, m.toe, ...])`,
   `.facing(m.toe_grid_1)`.
3. Add a retarget test: `Point3Bundle.of([heel, toe])` (markers) binds **identically** to `[heel.rest, toe.rest]`.
4. Update the patch-geometry prose in `AGENTS.md` (drop `.rest` from the canonical snippet).
