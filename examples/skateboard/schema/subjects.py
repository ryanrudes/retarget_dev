from retarget.core import Subjects, Subject

from .segments import LeftFootSegments, SkateboardSegments


class ExampleSubjects(Subjects):
    left_foot: Subject[LeftFootSegments]
    skateboard: Subject[SkateboardSegments]
