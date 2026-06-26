"""Authored marker↔joint correspondence → fungeom ``RosterMap``.

``RosterMap`` is the *identity* seam: which source key (e.g. a marker name)
corresponds to which target key (e.g. a joint name). retarget has no authored
marker↔joint correspondence today — it is supplied explicitly here as a plain
``{source: target}`` mapping over the authored string names, which ARE the
identity (no enums, no new schema concept). The *geometric* transfer that layers
on top is parked numerics; see :mod:`retarget.fungeom.transfer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fungeom import Roster, RosterMap

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

__all__ = [
    "roster_map",
    "identity_map",
]


def roster_map(correspondence: Mapping[str, str]) -> RosterMap:
    """Build a ``RosterMap`` from an authored ``{source: target}`` correspondence.

    Composition narrows to keys that chain through; ``inverse()`` is partial
    (``Unresolvable`` when two sources share a target). Those are fungeom's
    guarantees — this is the thin constructor that reads from retarget names.
    """
    # str is Hashable; the cast only works around Mapping key-type invariance.
    return RosterMap.of(cast("Mapping[Hashable, Hashable]", dict(correspondence)))


def identity_map(keys: Roster | Sequence[str]) -> RosterMap:
    """The identity correspondence over ``keys`` (a ``Roster`` or a name sequence).

    Pass a bundle's ``support()`` to get the identity over exactly the keys that
    are present.
    """
    return RosterMap.identity(keys)
