from dataclasses import dataclass

from retarget.core import Patch, Patches


@dataclass(frozen=True, slots=True)
class LeftShoePatches(Patches):
    sole: Patch


@dataclass(frozen=True, slots=True)
class BalanceBoardPatches(Patches):
    surface: Patch