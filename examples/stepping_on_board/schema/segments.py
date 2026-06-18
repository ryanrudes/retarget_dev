from retarget.core import Segments, Segment

from .markers import LeftShoeMarkers, RightShoeMarkers, BalanceBoardMarkers
from .patches import LeftShoePatches, RightShoePatches, BalanceBoardPatches


class LeftFootSegments(Segments):
    shoe: Segment[LeftShoeMarkers, LeftShoePatches]


class RightFootSegments(Segments):
    shoe: Segment[RightShoeMarkers, RightShoePatches]


class BalanceBoardSegments(Segments):
    board: Segment[BalanceBoardMarkers, BalanceBoardPatches]