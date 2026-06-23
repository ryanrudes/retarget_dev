from retarget.core import Segments, Segment

from .markers import LeftShoeMarkers, BalanceBoardMarkers
from .patches import LeftShoePatches, BalanceBoardPatches


class LeftFootSegments(Segments):
    shoe: Segment[LeftShoeMarkers, LeftShoePatches]


class BalanceBoardSegments(Segments):
    board: Segment[BalanceBoardMarkers, BalanceBoardPatches]