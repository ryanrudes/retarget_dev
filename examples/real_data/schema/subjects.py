from retarget.core import Subjects, Subject

from .segments import LeftFootSegments


class ExampleSubjects(Subjects):
    left_foot: Subject[LeftFootSegments]