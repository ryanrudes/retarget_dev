from retarget.core import Segments, Segment

from .markers import LeftShoeMarkers, SkateboardMarkers
from .patches import LeftShoePatches, SkateboardPatches


class LeftFootSegments(Segments):
    shoe: Segment[LeftShoeMarkers, LeftShoePatches]


class SkateboardSegments(Segments):
    deck: Segment[SkateboardMarkers, SkateboardPatches]
