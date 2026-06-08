from dataclasses import dataclass

from retarget.core.enums import MarkerId, PatchId, SegmentId


@dataclass(frozen=True, slots=True)
class MarkerHandle[M: MarkerId]:
    """
    Typed symbolic reference to a marker on a segment.

    A handle does not store marker geometry, observed data, or world-frame state.
    It only identifies which marker is being referred to.
    """

    segment: SegmentId
    """The segment this marker belongs to."""

    marker: M
    """The marker identifier."""

    @property
    def label(self) -> str:
        """The marker's external/user-facing label."""
        return self.marker.label

    @property
    def index(self) -> int:
        """The marker's index within its marker vocabulary."""
        return self.marker.index


@dataclass(frozen=True, slots=True)
class PatchHandle[P: PatchId]:
    """
    Typed symbolic reference to a patch on a segment.

    A handle does not store patch geometry or world-frame state.
    It only identifies which patch is being referred to.
    """

    segment: SegmentId
    """The segment this patch belongs to."""

    patch: P
    """The patch identifier."""

    @property
    def label(self) -> str:
        """The patch's external/user-facing label."""
        return self.patch.label

    @property
    def index(self) -> int:
        """The patch's index within its patch vocabulary."""
        return self.patch.index