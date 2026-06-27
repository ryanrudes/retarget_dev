# fungeom handover (round 2) — vectorize `FaceSignal.clearance` + the plane accessors

**Audience:** the **fungeom** repo (`~/GitHub/functional_api`). **From:** the retarget session.
**Context:** the 0.2.1 fix vectorized `FaceSignal.frame()`/`.boundary()` (thank you — path A landed).
The next retarget step — **bounded contact clearance** (a footprint vs another segment's Face-backed
patch, respecting the footprint edge) — needs `FaceSignal.clearance` and `plane().signed_distance`,
and both are still **per-instant**. One is a hard blocker.

This is the same shape as the last round: the API is correct, the *resolve_over* just isn't
vectorized over the sampling yet.

## Measurements (T=5000, K=5 footprint samples, single patch, 90°-Z sweep)

| call | time | status |
|---|---:|---|
| `FaceSignal.frame().resolve_over` | ~8 ms | ✅ fixed in 0.2.1 |
| `FaceSignal.boundary().resolve_over` | ~7 ms | ✅ fixed in 0.2.1 |
| `FaceSignal.plane().signed_distance(bundle).resolve_over` | **142 ms** | 🐢 per-instant |
| `FaceSignal.clearance(bundle).resolve_over` | **2651 ms** | 🐢🐢 per-instant (the blocker) |
| (`plane().normal()` / `origin()` from last round) | 20 / 51 ms | 🐢 per-instant (lower priority) |

`clearance` at 2.6 s is the killer — retarget calls it per tested-patch per patch-support, and the
numpy plane-dot-product it replaces is sub-millisecond. 142 ms for `signed_distance` is also well
past the ~8 ms carrier.

Repro:

```python
import time, numpy as np
from fungeom import Face, Point3, Point3Bundle, Region2, FaceSignal, TransformSignal, Sampling
pts = Point3Bundle.from_map({k: Point3.at(*v) for k, v in
      {"a":(-1,-1,0),"b":(1,-1,0),"c":(1,1,0),"d":(-1,1,0)}.items()})
face = Face.on(pts.fit_plane().facing(Point3.at(0,0,1.0)), Region2.hull(pts.in_frame(pts.fit_plane())))
T=5000; times=np.arange(T)*0.01; ang=np.linspace(0,np.pi/2,T); c,s=np.cos(ang),np.sin(ang)
M=np.zeros((T,4,4)); M[:,0,0]=c;M[:,0,1]=-s;M[:,1,0]=s;M[:,1,1]=c;M[:,2,2]=1;M[:,3,3]=1;M[:,0,3]=np.linspace(0,3,T)
fs=FaceSignal.of(face, TransformSignal.from_matrices(times,M)); samp=Sampling.at_times(times)
# (use retarget.fungeom.signals.point_bundle_signal, or any Point3BundleSignal of shape (T,K,3))
# t0=time.perf_counter(); fs.clearance(bundle).resolve_over(samp); print((time.perf_counter()-t0)*1000, "ms")
```

## What retarget does with it (so the fix targets the right thing)

fungeom's `clearance` is **unsigned** (distance to the bounded face); retarget needs a **signed**
contact clearance (− = penetration, penalized by the χ² scorer). retarget re-signs it branch-free
with the perpendicular distance, which is why it needs *both* accessors fast:

```python
perp      = face_sig.plane().signed_distance(footprint_bundle)   # signed ±, (T,K)
bounded   = face_sig.clearance(footprint_bundle)                 # unsigned ≥0, (T,K)  -- the bounded part
signed    = perp + sqrt(max(0, bounded**2 - perp**2))            # = perp inside footprint; + lateral gap off-edge
clearance = nanmin(signed, axis=1)                               # closest footprint sample; NaN where occluded
```

Inside the footprint `bounded == |perp|` so `signed == perp` (today's infinite-plane behavior);
off the edge it adds the lateral overhang as a positive gap (the accuracy win). Partiality already
works: an occluded sample resolves to NaN + a `present=False` mask, which becomes an honest contact
gap. So **correctness is fine — this is purely the per-instant resolve perf.**

## Ask + acceptance

Vectorize the `resolve_over` of `FaceSignal.clearance` and `PlaneSignal.signed_distance` (and, while
there, `plane().normal()`/`origin()`) — apply the materialized `(T,4,4)` pose stack to the static
geometry in one batched numpy op, the way `frame()`/`boundary()` now do. **Acceptance:** at T=5000,
K=5 each is within a small constant of the ~8 ms carrier (clearance especially: 2651 ms → ~tens of
ms), exact-matching the per-instant values. `clearance` is the priority.

## retarget status

Bounded clearance is **designed and staged** (formula above) but **not landed** — pending this
vectorization. The detector stays on the fast, correct **infinite-plane** clearance for patch
supports until then (a foot off the support edge still reads as contact; that's the accuracy gap the
bounded version closes). Everything else (authoring + the path-A runtime) is fungeom-native and on
`master`. Ping when a 0.2.x lands and I'll wire bounded clearance in.
