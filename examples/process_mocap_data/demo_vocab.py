from retarget.core.enums import TrackId


class GroundEstimationTrackId(TrackId):
    """Track identifiers for the ground-estimation demonstration."""

    MOCAP = "mocap"
    VIDEO = "video"
    SMPL = "smpl"
    CONTACTS = "contacts"
