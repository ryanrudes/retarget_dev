from retarget.core import Segments, Segment

from .markers import LeftShoeMarkers
from .patches import LeftShoePatches


class LeftFootSegments(Segments):
    shoe: Segment[LeftShoeMarkers, LeftShoePatches]