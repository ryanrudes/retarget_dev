# Design spec — a vendored `smpl` library (the URDF-style counterpart for bodies)

**Goal:** a clean, typed, standalone `smpl` library (own repo `ryanrudes/smpl`, vendored like
`urdf`) that loads SMPL-family body models and runs forward kinematics, so `SmplTrack` becomes
"give it a model + params" easy and variant-aware. retarget *consumes* it through the contract that
already exists on the retarget side (`BodyModel` Protocol + `SmplTrack.from_body`, shipped); the lib
*implements* that contract. torch is fine to use.

## Mirror of `urdf` (the ideas you already used)

| `urdf` | `smpl` |
|---|---|
| parse URDF XML → in-memory model | load SMPL-family `.npz/.pkl` → in-memory model |
| typed link/joint/skeleton codegen | typed per-variant **param** + **body** types |
| `robots/registry` + per-robot packages | model **registry** + per-variant packages |
| `kinematics/kinematic_chain` (FK) | **LBS forward kinematics** (params → joints/vertices) |
| `dynamics/inertia` | (optional) per-vertex mass / contact surfaces |
| own pyproject, tests, docs, strict gate | same (hatchling, py.typed, mypy strict, 100% cov) |

## The one structural insight (why "all variants" is cheap)

SMPL, SMPL-X, SMPL+H, MANO, FLAME, STAR, SMAL **share one math**: a template mesh + shape
blendshapes (β) + pose blendshapes (θ) + a joint regressor + **linear blend skinning** over a
kinematic tree. They differ only in *sizes* and *extra params*. So — exactly like `urdf` parses any
URDF — **one generic loader handles any SMPL-family file**, and a variant is just (a) the data and
(b) a typed param dataclass.

| variant | joints J | adds | params beyond β/global/transl |
|---|---|---|---|
| SMPL | 24 | — | `body_pose` (23×3) |
| SMPL+H | 52 | hands | + `left/right_hand_pose` |
| SMPL-X | 55 | hands + face | + hand poses, `jaw/eye_pose`, `expression` |
| MANO | 16 | hand only | `hand_pose` |
| FLAME | — | head/face | `jaw_pose`, `expression` |

## The contract retarget already depends on (the lib implements this)

```python
@runtime_checkable
class BodyModel[ParamsT](Protocol):           # in retarget.demo.smpl, shipped
    @property
    def joint_names(self) -> tuple[str, ...]: ...
    def forward_joints(self, params: ParamsT) -> np.ndarray: ...   # (T, J, 3) world joints, numpy

SmplTrack.from_body(model, params, timestamps)   # runs FK -> track; shipped + tested with a stub
```

The lib’s job is to provide `BodyModel[SmplParams]`, `BodyModel[SmplxParams]`, … . **It may use torch
internally but must return a numpy `(T, J, 3)`** (`.detach().cpu().numpy()`), so retarget stays
torch-free in its own code and the track is variant-agnostic downstream.

## Proposed lib layout

```
src/smpl/
  core/        BodyModel base, the LBS forward-kinematics kernel, kinematic tree, blendshapes
  params/      typed param dataclasses per variant (SmplParams, SmplxParams, ManoParams, …)
  models/      per-variant model classes (Smpl, Smplx, …) -- thin: load arrays + variant param packing
  registry/    register a model by variant + a model-file path (you bring the .npz)
  io/          load the .npz/.pkl into the structured arrays (template, shapedirs, posedirs, J_regressor, weights, kintree)
```

### Backend decision (torch is fine)

- **Wrap `smplx` (recommended to start).** The official lib is correct, differentiable, and already
  covers every variant. The vendored `smpl` lib becomes a **clean typed facade**: typed params +
  registry + the `BodyModel` contract + numpy-out, with `smplx` doing the FK. Fastest to a correct,
  all-variants result, and differentiability is free for the eventual R5 solver.
- **Torch-native FK (the `urdf`-from-scratch ethos).** ~a few hundred lines of LBS; full control, no
  `smplx` dependency, but you re-verify correctness per variant.
- **Recommendation:** keep the facade + contract identical and start by wrapping `smplx`; the backend
  can swap to torch-native later **without touching retarget** (the `BodyModel` contract is the seam).

## Licensing — model files are NOT redistributable

SMPL/SMPL-X model `.npz/.pkl` are registration-gated at the MPI sites. So the lib **ships code only**;
users supply their own model files (exactly like `urdf` needs you to bring the `.urdf`). The
`registry` takes a path; tests use tiny synthetic/fixture arrays (template + a couple of joints), not
the real models — so CI stays redistributable and offline.

## How it lands in retarget (already wired + the next step)

- **Now (shipped):** `SmplTrack.from_body(model, params, timestamps)` → joints over time → temporal
  sync via `smpl_joint_energy` (no sync-side changes).
- **Next (schema integration):** the lib's **mesh/vertices** let us place foot-contact `patches`
  (`geometry=`→`Face`) and map joints→`markers`, so **contact detection reuses** for SMPL feet — the
  point where SMPL becomes "just another subject" in the typed/fungeom substrate.
- **Later (R5):** a differentiable (`smplx`-backed) body model makes the retargeting transfer's
  solver gradient-friendly.

## Build order for the lib

1. `io` loader + `core` LBS FK (or the `smplx` wrap) + `SmplParams`/`Smpl` → implement
   `BodyModel[SmplParams]`. Validate `from_body` against retarget with a real SMPL sequence.
2. Add `SmplxParams`/`Smplx` (the common case for video pipelines).
3. Registry + the remaining variants (SMPL+H/MANO/FLAME) as drop-ins.
4. Mesh/vertex output (for the retarget schema-integration step).
