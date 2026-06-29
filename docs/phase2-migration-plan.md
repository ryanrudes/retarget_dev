All facts confirmed. Baseline: pytest green, mypy strict green (56 files, scope `packages=["retarget"]`), ruff clean on `src/retarget` but **7 pre-existing ruff errors in tests/examples** (4× E402 in `examples/cli.py`, 3× F401 in `tests/test_demo_contact.py`/`test_smoke.py`) that are unrelated to this work. The synthesis core mechanism (dataclass-rooted `_Schema`) is verified zero-ignore, guardrail-preserving, and deep-chain-correct under the repo's own mypy. Here is the synthesized plan.

---

# Migration: TypedDict schemas → attribute‑symbol dataclass schemas

The three designs agree on ~90%. The decisive split is the **schema base shape**, and I verified the winner empirically against this repo's mypy (`>=2.1.0`, strict, `warn_unused_ignores`): a **dataclass‑rooted `_Schema`** makes every generic reflection helper and the `type(x)(**fields)` rebuild type‑check with **zero `# type: ignore` and zero `cast`** (the existing spike needed one ignore *only* because its typevar was unbounded), while the empty bases used as PEP 695 bounds keep the swap/foreign‑arg guardrail (both still `[type-var]`‑error). I adopt that core (the "purity" root), staged behind the "staging" bridge mixin for green file‑by‑file commits, hardened with the "risk" findings (the deep chain is *outside* the mypy gate; the deep‑chain test already fails strict mypy at line 60; ruff/coverage realities).

---

## 1. NEW CORE MECHANISM

### 1.1 Schema bases — a dataclass root + five empty dataclass bases

New module `src/retarget/core/schema/base.py` holds the structural root and the reflection kernel. The five bases (`Markers`, `Patches`, `Segments`, `Subjects`, `Tracks`) each become an **empty `@dataclass(frozen=True, slots=True)` subclass of `_Schema`** (today they are `TypedDict`). Keeping the five distinct names is mandatory — they are the **bounds** that block slot‑swaps.

```python
# src/retarget/core/schema/base.py
from __future__ import annotations
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from typing import Any

@dataclass(frozen=True, slots=True)
class _Schema:
    """Structural root of every authored schema container. Being a dataclass it carries
    __dataclass_fields__, so `fields(x: SchemaT)` type-checks with NO ignore for any SchemaT: _Schema."""
    # ── TRANSITIONAL bridge (Stage 1 adds it by inheritance; Stage 3 DELETES these 6 methods) ──
    def __getitem__(self, name: str) -> Any: return getattr(self, name)
    def __iter__(self) -> Iterator[str]: return iter(_schema_fields(self))
    def __contains__(self, name: object) -> bool: return isinstance(name, str) and name in _schema_fields(self)
    def items(self) -> Iterator[tuple[str, Any]]: return _schema_items(self)
    def values(self) -> Iterator[Any]: return _schema_values(self)
    def keys(self) -> tuple[str, ...]: return _schema_fields(self)
    # ── end bridge ──

def _schema_items(schema: _Schema) -> Iterator[tuple[str, Any]]:
    for f in fields(schema): yield f.name, getattr(schema, f.name)        # no ignore
def _schema_values(schema: _Schema) -> Iterator[Any]:
    for f in fields(schema): yield getattr(schema, f.name)
def _schema_fields(schema: _Schema) -> tuple[str, ...]:
    return tuple(f.name for f in fields(schema))
def _schema_get(schema: _Schema, name: str) -> Any:                       # dynamic string lookups (signals.py, IO)
    if name not in _schema_fields(schema): raise KeyError(name)
    return getattr(schema, name)
def _rebuild_schema[SchemaT: _Schema](schema: SchemaT, bind: Callable[[str, Any], Any]) -> SchemaT:
    return type(schema)(**{name: bind(name, value) for name, value in _schema_items(schema)})
```

```python
# marker.py / patch.py / segment.py / subject.py / demo.py
@dataclass(frozen=True, slots=True)
class Markers(_Schema): ...        # …identically Patches, Segments, Subjects, Tracks
```

Concrete authoring (tests/examples) gains **one decorator line**, body and base unchanged:

```python
@dataclass(frozen=True, slots=True)        # the only addition
class ShoeMarkers(Markers):
    heel: Marker
    toe: Marker
```

`fields(x: SchemaT)` type‑checks with no ignore precisely because `_Schema` is itself a dataclass (its bound supplies `__dataclass_fields__`, satisfying `fields()`'s `DataclassInstance` overload) — this is the property the plain‑class variants in the other two designs lacked, forcing a `# type: ignore[arg-type]`/`cast(Any, …)`.

### 1.2 Generic containers — signatures unchanged

`Marker`, `Patch`, `Segment[MarkersT: Markers, PatchesT: Patches]`, `Subject[SegmentsT: Segments]`, `MocapTrack[SubjectsT: Subjects = Subjects]`, `Demonstration[TracksT: Tracks = Tracks]` keep their PEP 695 params, defaults, `_binding` init‑false fields, and frozen+slots verbatim. The bounds now resolve to dataclass bases — transparent to every signature. `segment.markers` still projects `MarkersT`; `.heel` projects `Marker`. No `__getitem__`/`__iter__` is ever added to these containers (the bridge lives on `_Schema`, the child‑vocabulary bags, not the containers).

### 1.3 Attribute projection keeps the typed deep chain (no codegen)

Identical mechanism to today, one substitution at the leaf: *literal‑key subscript on a concrete `TypedDict`* → *attribute access on a concrete frozen dataclass*. Both are native, codegen‑free, plugin‑free type projections. Five hops, each `TypeVar‑typed attribute → concrete field type`:

```
demo.tracks(:TracksT=GroundTracks).mocap(:MocapTrack[MySubjects]).subjects(:MySubjects)
  .left_shoe(:Subject[ShoeSegments]).segments.shoe(:Segment[ShoeMarkers,ShoePatches])
  .markers.heel(:Marker).rest(:Point3)        # symbol loop closes
```

`Demonstration[TracksT].tracks -> TracksT` and `MocapTrack[SubjectsT].subjects -> SubjectsT` are unchanged in shape. Verified: all `assert_type` pass under this repo's strict mypy; swapped `Segment[ShoePatches, ShoeMarkers]` and foreign `Segment[int, …]` still `[type-var]`‑error.

### 1.4 Binding rebuilds dataclass schemas generically

The five dict‑comprehension forks in `binding.py` (`{name: …}` + `cast(SubjectsT/Any, …)`) collapse to `_rebuild_schema`; the four wrapper reconstructions (`Subject(...)`, `Segment(...)`, `Marker(...)`, `Patch(...)` + their `_binding` `object.__setattr__`) are **unchanged** (already explicit dataclass construction). The **six `cast`s drop**; no ignore is added. The `marker_env` (`binding.py:104‑109`) is **already keyed by the authored `Marker` identity**, so `face.bind(env)` semantics are intact; only its traversal switches to `_schema_items`.

```python
# _bind_subjects (was binding.py:45-49)
return _rebuild_schema(subjects, lambda name, s: _bind_subject(s, name=name, runtimes=runtimes))
# _bind_subject — segments container (was 58-67); cast(Any, …) at :69 drops
bound_segments = _rebuild_schema(subject.segments, lambda seg_name, seg: _bind_segment(
    seg, subject=name, segment_name=seg_name, body_model=subject.body_model,
    runtime=(None if runtimes is None else runtimes.get(SegmentKey(name, seg_name)))))
# _bind_segment — markers/patches (was 85-126); both cast(Any, …) at :123-124 drop
bound_markers = _rebuild_schema(segment.markers, lambda m, mk: _bind_marker(mk, subject=subject,
    segment=segment_name, marker_name=m, body_model=body_model, runtime=runtime))
# marker_positions_segment + marker_env: iterate _schema_items(segment.markers)/_schema_items(bound_markers)
bound_patches = _rebuild_schema(segment.patches, lambda p, pt: _bind_patch(pt, subject=subject,
    segment=segment_name, patch_name=p, marker_positions_segment=marker_positions_segment,
    marker_env=marker_env, runtime=runtime))
```

Validators: `_validate_subjects(subjects: Subjects)` (was `Mapping[str, Any]`), `if not subjects` → `if not _schema_fields(subjects)`, `.items()` → `_schema_items`; the empty‑name guard (`:199`) drops. `_validate_segment` `.items()` → `_schema_items`. `MocapTrack._build_segment_runtimes` keeps its `dict[SegmentKey, _SegmentRuntime]` and only switches its two `.items()` walks to `_schema_items`.

### 1.5 Demonstration API + batch query helpers

**Demonstration must stop replacing `tracks` with a `MappingProxyType`** (that erases the dataclass and breaks `demo.tracks.mocap`). `__post_init__` becomes **validate‑only** (the frozen dataclass is already the freeze); the secondary string surface is derived on demand:

```python
def __post_init__(self) -> None:
    for name, value in _schema_items(self.tracks):
        if not isinstance(value, Track):
            raise TypeError(f"Track {name!r} must be a Track; got {type(value).__name__}")
def _track_map(self) -> Mapping[str, Track]:
    return _track_mapping(self.tracks)          # _track_mapping lives in demo.py (needs Track), built on _schema_items
```

`_freeze_tracks` is deleted. `demo["mocap"]`, `in`, `iter`, `track_ids()`, `slice_time(...)`, `DemonstrationView`, `resample_to(reference: str)`, and the whole `sync.py` track‑name surface **stay string‑keyed** (secondary access + sync‑graph identity), now reading through `_track_map()`. `DemonstrationView` is **unchanged** (its `tracks: Mapping[str, Track]` is already type‑erased; its `resample_to` `.items()` is an honest dict, not a schema).

**Batch helpers** (`segment.py:179‑242`) flip varargs `*: str` → `*: Marker` / `*: Patch`; the symbols are already bound query handles, so `_marker`/`_patch`/`_missing_*_message` are deleted. Only the resolution step changes; empty‑case shaping and `as_dict` packaging are unchanged, and `as_dict` keys stay the authored name via the symbol's own target:

```python
def marker_positions(self, *markers: Marker, modeled: bool = False, as_dict: bool = False):
    if as_dict:
        return {m.target.marker: m.positions(modeled=modeled) for m in markers}   # authored name = runtime identity
    return _stack_like_today([m.positions(modeled=modeled) for m in markers])     # reuse existing (T,N,3) path
```

`marker_target`/`patch_target` take a symbol and delegate to the already‑attribute‑driven `marker.target`/`patch.target`; `_pose_jump_mask` uses `_schema_values(self.patches)`; `Subject.segment_external_name` takes a `Segment` symbol (or is dropped — §4).

### 1.6 What stays string‑based (the runtime‑identity floor, unchanged)

`SegmentKey`, `SegmentTarget`/`MarkerTarget`/`PatchTarget` (+ `.segment_key`), `SceneState`/`SegmentPoseTrajectory` (keyed by `SegmentKey`), `ContactTrack`/`SupportStateTrack` (keyed by `PatchTarget`), `_SegmentRuntime` (keyed by `mocap_name`/patch‑name/support‑name), the four `_*Binding`s, `Marker.mocap_name`/`Subject.body_model`, all **track names** across `demo`/`sync`/`alignment`, and the fungeom adapter keys — all unchanged, now **sourced from `f.name`** instead of dict keys. `Marker.target`/`Patch.target` already are the attribute‑driven template. **Out of scope (do NOT migrate):** `SegmentGeometry.markers["a","b","c"]` in `core/geometry.py` (the fungeom `MarkerGeometry` adapter inside patch callables — a different object returning `Point3`/`Point3Bundle`) and its `test_fungeom_*` `assert_type` sites.

### 1.7 The transitional bridge (why staging stays green)

A dataclass is not subscriptable, so the base flip would otherwise break ~150 access sites at once. The six bridge methods on `_Schema` (§1.1) keep every un‑migrated `schema["x"]`/`for x in schema`/`.items()`/`.values()`/`.keys()`/`in` working at runtime (typed `Any`, harmless — tests/examples are not mypy‑gated). Stage 3 deletes them, turning any leftover subscript into a hard mypy+runtime error that proves the unification is complete.

---

## 2. FILE‑BY‑FILE CHANGE PLAN (core‑first, gated)

**GATE** (all must pass before each commit):

```bash
python3 -m compileall -q src/retarget examples
uv run --extra dev pytest -q
uv run --extra dev mypy                                       # strict; scope = packages=["retarget"]
uv run --extra dev ruff check src/retarget                   # green today — keep green
uv run --extra dev mypy --strict tests/test_typed_deep_chain.py   # the deep-chain guarantee (NOT in gated scope); from Stage 1 on
```

Notes: `--extra dev` is required (plain `uv sync` prunes pytest/ruff/mypy). The mypy gate **cannot** see the deep chain (`packages=["retarget"]` excludes tests/examples — verified), so the explicit `mypy --strict` on the deep‑chain test is the real guarantee and is added from Stage 1. Ruff has 7 **pre‑existing** failures in tests/examples (4× E402 `examples/cli.py`, 3× F401 `tests/test_demo_contact.py`+`test_smoke.py`) unrelated to this work — keep `src/retarget` green and do not *add* violations to touched files (see §4 for whether to clean them).

### Stage 0 — primitives + prerequisites (pure additions; gate green)
- **New** `src/retarget/core/schema/base.py`: `_Schema` (with bridge), `_schema_items/_values/_fields/_get`, `_rebuild_schema`. Re‑export the helpers from `core/schema/__init__.py` for cross‑module import.
- **New** `tests/test_schema_base.py`: unit‑test `_rebuild_schema` (empty + non‑empty; concrete type preserved), `_schema_items` ordering, `_schema_get` KeyError branch (locks ~100% coverage on the new module).
- **Fix R2** (prerequisite for the explicit deep‑chain check): `tests/test_typed_deep_chain.py:60‑61` `position_segment=np.zeros(3)` / `np.array([1.0,0.0,0.0])` → `cast(Vec3, np.zeros(3))` / `cast(Vec3, np.array([1.0,0.0,0.0]))` (repo idiom, e.g. `fungeom/readback.py:50`).
- Gate.

### Stage 1 — the coordinated core flip (one bridge‑protected commit; gate green)
This is irreducible: base‑flip ⇒ all 62 concrete schemas need `@dataclass` (else construction has no `__init__`) ⇒ binding must yield dataclasses (else bound `track.subjects.left_shoe` fails) ⇒ `Demonstration` must stop wrapping tracks ⇒ the `assert_type` deep‑chain sites must convert. Everything *else* rides the bridge.
1. Bases: `marker.py:19`, `patch.py:39`, `segment.py:28`, `subject.py:13`, `demo.py:26`: `TypedDict` → `@dataclass(frozen=True, slots=True) class X(_Schema)`; drop now‑unused `TypedDict` imports. Reword `core/schema/__init__.py:1‑18` ("concrete `@dataclass` … attribute access projects").
2. `binding.py`: 5 forks → `_rebuild_schema`, 2 validators → `_schema_items`, drop 6 casts + empty‑name guard, `import` the helpers (+ `from dataclasses import fields` is no longer needed here).
3. `demo.py`: `Demonstration.__post_init__` validate‑only; `_track_map` via new `_track_mapping`; delete `_freeze_tracks`.
4. Add `@dataclass(frozen=True, slots=True)` to **all 62** concrete schemas — 12 test files (incl. the 4 in‑function decls in `test_smpl_track.py` and the empty `pass` ones) + 8 example schema decls (`examples/_shared/schema/*`).
5. Convert only the `assert_type`/subscript sites mypy‑strict now mistypes: `tests/test_typed_deep_chain.py` (8 sites → attribute), `tests/test_demo_container.py:36‑39,53‑55`.
- Gate (now including the explicit deep‑chain `mypy --strict`).

### Stage 2 — mechanical fan‑out (many tiny commits; bridge supports the unconverted)
Order: src walkers → batch helpers+callers → tests → examples. Gate after each cohort.
- **src walkers** → `_schema_items`/`_schema_values`/`_schema_fields`/`_schema_get`: `core/geometry.py:122`; `demo/mocap.py:407‑410,460,486,512` (+ docstring `:8‑10`); `demo/smpl.py:203‑204`; `demo/pose_repair.py:62‑63`; `io/unbagged.py:127‑130`; `contacts/_scope.py:28‑40`; `demo/sync.py:253` (`_tracks_of` → `_track_mapping`); `fungeom/signals.py` (`_names`→`_schema_fields`; `_subject`/`_segment` subscripts→`_schema_get`; string params stay — boundary decision).
- **Batch helpers** `segment.py:179‑261` → symbol varargs; delete `_marker`/`_patch`/`_missing_*`; update the one src caller `fungeom/signals.py:208` to resolve via `_schema_get(...).positions(...)`. (Optional foreign‑symbol guard preserves the error‑path test — §4.)
- **Tests** (21 files; exhaustive site list in the provided MAP): `schema[name]` → `schema.name`; batch calls pass `segment.markers.heel`/`patches.sole`; `conftest.demo_segment` (`track.subjects[SUBJECT].segments[SEGMENT]`) → attributes; `marker_target("…")`/`patch_target("…")` → `markers.heel.target`/`patches.sole.target`. Leave Category‑B `seg.markers["heel","toe"]` in `test_fungeom_*` as‑is.
- **Examples** (13 files): `cli.py`, `_shared/{run,video,inspect_geometry,inspect_geometry_3d}.py`, 8 `left_shoe_*/experiment.py` → attribute. The 3 `.keys()`+variable‑key loops (`inspect_geometry*.py`, `video.py`) → `_schema_fields`/`_schema_items`. `scene.py` builders need **no structural change** (already kwargs‑per‑field + data‑form `m.rest`).

### Stage 3 — remove the bridge (locks the unification; gate green)
- Delete the 6 bridge methods from `_Schema`. Bases become bare empty dataclasses.
- Grep gate: `rg '\.(markers|patches|segments|subjects|tracks)\[' src tests examples` is empty except Category‑B `SegmentGeometry`. Optional test: `segment.markers["x"]` now raises `TypeError`.
- Gate.

### Stage 4 — docs (fold into Stage 3 if small)
- `AGENTS.md` (headline + "Authoring schema"/"Batch queries"/"Demonstration containers"), `demo.py:4`/`mocap.py:8‑10` docstrings, `docs/scene.md`: subscript→attribute.

---

## 3. RISKS + MITIGATIONS

- **Deep chain is outside the mypy gate (highest).** `packages=["retarget"]` — tests/examples are not type‑checked, so the gate alone cannot catch a deep‑chain regression. **Mitigation:** the explicit `mypy --strict tests/test_typed_deep_chain.py` is in the GATE from Stage 1; promoting it permanently is §4.
- **Deep‑chain test already fails strict mypy (R2).** `test_typed_deep_chain.py:60‑61` (`np.zeros(3)` vs `Vec3`) — *verified*. **Mitigation:** fixed in Stage 0 (`cast(Vec3, …)`); otherwise the explicit check can never go green.
- **`assert_type` must ride attributes, not the bridge.** Bridge `__getitem__ -> Any` makes `assert_type(demo.tracks["mocap"], MocapTrack[…])` fail. **Mitigation:** convert all `assert_type` sites in Stage 1; the attribute chain is *verified* to pass.
- **`type(x)(**fields)` against a bound typevar / `fields()` needing an ignore.** *Verified clean*: the dataclass‑rooted `_Schema` bound gives precise `SchemaT` return and ignore‑free `fields()`; net the migration **removes 6 casts and adds zero ignores** (satisfies `warn_unused_ignores`).
- **Guardrail silently weakened.** *Verified*: swapped/foreign type args still `[type-var]`‑error — parity with the TypedDict bounds.
- **`Demonstration` no longer `MappingProxyType`‑wrapping `tracks`.** Most behavior‑changing edit. **Mitigation:** audited — only `_track_map`/`slice_time`/sync used it as a mapping, all rerouted through `_track_mapping`; add a test that `demo.tracks` is the authored dataclass and `demo["mocap"] is demo.tracks.mocap`.
- **Walkers lose `cast(Mapping[str, X])` element precision (`_schema_items -> Any`).** Strict mypy stays green (declared `Any`), but a wrong `.attr` in a loop body won't be caught. **Mitigation:** annotate the loop var where the guardrail matters (`subject: Subject[Any]`); net checking ≈ today's casts.
- **Bridge transiently erodes subscript typing / masks key typos.** **Mitigation:** affects only sites pending conversion; removed in Stage 3; the *guarantee* (assert_type) is on attributes from Stage 1.
- **Coverage (~100% norm; no `--cov-fail-under` — verified).** New `base.py` lines are hit by binding/walkers on the first bound track; `_schema_get` KeyError, `_rebuild_schema` empty/non‑empty, and the `Demonstration` TypeError branch get explicit tests (Stage 0 + keep the existing non‑Track test). Deleting `_marker`/`_patch`/`_missing_*` removes their KeyError branches — ensure the replacement symbol/guard path is exercised.
- **Frozen+slots multi‑level inheritance + hashability shift.** *Verified* construct/rebuild at runtime; schema bags become hashable `eq=True` dataclasses but are never used as keys/compared (`marker_env` keys the already‑frozen `Marker` leaf by `mocap_name`, unique per segment) — no break.
- **ruff gate is already red on tests/examples (7 pre‑existing).** **Mitigation:** scope the enforced ruff gate to `src/retarget` (green; AGENTS.md says ruff isn't a required gate anyway) and "no new violations in touched files"; see §4.
- **Stage 1 is large + atomic.** Unavoidable (base‑flip + 62 decorators must co‑land), but every edit is mechanical and the bridge keeps non‑`assert_type` sites green, so it is reviewable as one logical change.

---

## 4. OPEN QUESTIONS (need the human's call)

1. **Public‑API removals vs symbol‑delegating shims.** Default (reversible) is to keep `marker_target`/`patch_target` accepting a **symbol** and delegating to `marker.target`/`patch.target`, and to keep `Subject.segment_external_name` (symbol arg). Removing them entirely (they're superseded by `markers.heel.target`) is cleaner but a public‑API deletion → confirm. Which?
2. **Batch‑helper signature change is authorized but is an API break.** Flipping `marker_positions(*markers: str)` → `*markers: Marker` (and the five siblings) is the explicit goal, but changes the public shape. Confirm, and decide whether to keep a **foreign‑symbol membership guard** (`m._binding.segment == self._binding.segment`) to preserve the old error path + its coverage.
3. **Promote `tests/test_typed_deep_chain.py` into the mypy gate permanently?** That closes the R1 hole but is a harness/config change (`pyproject` `packages`/`files`) and would pull the whole `tests/` dir into strict checking (surfacing more pre‑existing strict errors beyond R2). Keep it as a manual/CI explicit command, or wire it in?
4. **Ruff gate scope.** Enforce ruff only on `src/retarget` (matches reality + AGENTS.md), or also fix the 7 pre‑existing tests/examples failures (3 auto‑fixable F401s; the 4 `examples/cli.py` E402 need `# noqa` for the intentional sys.path‑before‑import) as a Stage 0 cleanup?
5. **Bridge: ship it, or land atomically?** The bridge enables many small green commits (recommended) but is throwaway code that briefly types subscript as `Any`. Acceptable, or prefer one atomic Stage 1+2 commit with no bridge?
6. **Stage C (full purity, separable).** Retire the callable patch form + `SegmentGeometry`/`MarkerGeometry` string subscript (Category B), keeping only the data form (`Face.on(...)` over `m.rest`, already the example path)? This removes the last string‑subscript surface but is a larger, independent change (touches `core/geometry.py`, `test_fungeom_*`, callable‑form fixtures) — defer to a follow‑up, or fold in?

Verification artifacts: `/tmp/synth_verify.py` (synthesis core — strict‑clean, zero ignore, runtime‑OK), `/tmp/synth_guard.py` (guardrail errors as intended), in‑repo proof `docs/phase2-container-fork-spike.py`.
