"""fungeom backend adapter — retarget's compiled scene ↔ fungeom resolvers.

A thin, typed translation layer over `fungeom`, the decidable
geometry/temporal/collections backend (sibling repo `ryanrudes/fungeom`). It
moves retarget's authored names and time-major arrays across the seam and reads
results back; **no geometry logic lives here** — that is fungeom's job.

- `marker_cloud_signal` / `pose_signal` — bound `MocapTrack` → `Point3BundleSignal`
  (markers, occlusion-aware) and `TransformBundleSignal` (segment poses).
- `point_bundle_signal` / `transform_bundle_signal` — the array-level core.
- `roster_map` / `identity_map` — authored `{source: target}` → `RosterMap`.
- `relabel` — the identity transfer; `GeometricTransfer` marks the parked
  numeric seam.
- `marker_at` / `joint_at` / `point_array` / `pose_matrix` / `resolvability` —
  read-back, surfacing `Unresolvable` reasons rather than swallowing them.

`Resolvable` / `Unresolvable` / `UnresolvableError` are re-exported so callers
need not import `fungeom` directly to branch on partiality.
"""

from __future__ import annotations

from fungeom import Resolvable, Unresolvable, UnresolvableError

from retarget.fungeom.correspondence import identity_map, roster_map
from retarget.fungeom.readback import (
    Mat4,
    joint_at,
    marker_at,
    point_array,
    pose_matrix,
    resolvability,
)
from retarget.fungeom.signals import (
    marker_cloud_signal,
    point_bundle_signal,
    pose_signal,
    transform_bundle_signal,
)
from retarget.fungeom.transfer import GeometricTransfer, relabel

__all__ = [
    # signals
    "marker_cloud_signal",
    "pose_signal",
    "point_bundle_signal",
    "transform_bundle_signal",
    # correspondence
    "roster_map",
    "identity_map",
    # transfer
    "relabel",
    "GeometricTransfer",
    # read-back
    "marker_at",
    "joint_at",
    "point_array",
    "pose_matrix",
    "resolvability",
    "Mat4",
    # re-exported fungeom decidability surface
    "Resolvable",
    "Unresolvable",
    "UnresolvableError",
]
