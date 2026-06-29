from dataclasses import dataclass

from retarget.demo import Tracks, MocapTrack

from .subjects import ExampleSubjects


@dataclass(frozen=True, slots=True)
class ExampleTracks(Tracks):
    mocap: MocapTrack[ExampleSubjects]