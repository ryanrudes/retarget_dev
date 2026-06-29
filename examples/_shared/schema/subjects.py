from dataclasses import dataclass

from retarget.core import Subjects, Subject

from .segments import LeftFootSegments, BalanceBoardSegments


@dataclass(frozen=True, slots=True)
class ExampleSubjects(Subjects):
    left_foot: Subject[LeftFootSegments]
    balance_board: Subject[BalanceBoardSegments]