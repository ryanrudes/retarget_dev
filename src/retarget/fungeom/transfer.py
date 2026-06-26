"""Transfer across the identity seam — and the parked numeric seam beyond it.

``relabel`` is the **identity transfer**: re-key a bundle source→target through a
``RosterMap``. It carries values across unchanged and drops out-of-domain keys;
it does *not* move geometry. That is exactly fungeom's ``Bundle.relabel`` — this
module just lets callers pass a plain ``{source: target}`` mapping.

The **numeric geometric transfer** (where a target joint actually goes given
source motion) is parked, like retarget's other numeric kernels. It is *not*
implemented here; :class:`GeometricTransfer` only marks the seam and pins the
input the handoff spec already fixes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload

from fungeom import Point3Bundle, RosterMap, TransformBundle

from retarget.fungeom.correspondence import roster_map

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "relabel",
    "GeometricTransfer",
]


@overload
def relabel(bundle: Point3Bundle, correspondence: Mapping[str, str] | RosterMap) -> Point3Bundle: ...
@overload
def relabel(bundle: TransformBundle, correspondence: Mapping[str, str] | RosterMap) -> TransformBundle: ...
def relabel(
    bundle: Point3Bundle | TransformBundle,
    correspondence: Mapping[str, str] | RosterMap,
) -> Point3Bundle | TransformBundle:
    """Identity transfer: re-key ``bundle`` through ``correspondence``.

    ``correspondence`` may be a ready :class:`RosterMap` or a plain
    ``{source: target}`` mapping (built into one here). Out-of-domain keys drop;
    collapsing two source keys onto one target is ``Unresolvable`` (fungeom's
    rule), surfaced lazily on ``decide()``.
    """
    mapping = correspondence if isinstance(correspondence, RosterMap) else roster_map(correspondence)
    return bundle.relabel(mapping)


class GeometricTransfer(Protocol):
    """PARKED seam for the numeric retarget transfer (out of scope here).

    Identity correspondence is ``RosterMap`` + :func:`relabel`. The *geometric*
    map — where a target joint goes given source motion — is parked numerics, in
    the same spirit as DTW/ICP: fungeom (and this adapter) *call* such a thing,
    they do not *be* it. This Protocol is provisional; its exact shape is to be
    designed when the estimator is pulled in (handoff open question #5). It pins
    only the input the spec already fixes — a source correspondence plus
    source/target clouds — and a fitted per-entity geometric map as output.
    """

    def fit(
        self,
        *,
        correspondence: RosterMap,
        source: Point3Bundle,
        target: Point3Bundle,
    ) -> TransformBundle: ...
