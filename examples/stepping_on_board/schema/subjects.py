from retarget.core import Subjects, Subject

from .segments import LeftFootSegments, RightFootSegments, BalanceBoardSegments


class ExampleSubjects(Subjects):
    left_foot: Subject[LeftFootSegments]
    right_foot: Subject[RightFootSegments]
    balance_board: Subject[BalanceBoardSegments]