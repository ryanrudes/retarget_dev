from dataclasses import dataclass

from retarget.core import Segments, Segment

from .markers import LeftShoeMarkers, BalanceBoardMarkers
from .patches import LeftShoePatches, BalanceBoardPatches


@dataclass(frozen=True, slots=True)
class LeftFootSegments(Segments):
    shoe: Segment[LeftShoeMarkers, LeftShoePatches]


@dataclass(frozen=True, slots=True)
class BalanceBoardSegments(Segments):
    board: Segment[BalanceBoardMarkers, BalanceBoardPatches]