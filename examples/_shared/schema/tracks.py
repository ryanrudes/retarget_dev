from retarget.demo import Tracks, MocapTrack

from .subjects import ExampleSubjects


class ExampleTracks(Tracks):
    mocap: MocapTrack[ExampleSubjects]