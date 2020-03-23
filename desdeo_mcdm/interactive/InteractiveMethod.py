"""Implements a base class defining a common interface for all interactive methods.
"""

from typing import List

from desdeo_tools.interaction.request import BaseRequest
from desdeo_problem.Problem import ProblemBase


class InteractiveMethod:
    def __init__(self, problem: ProblemBase):
        self._problem = problem

    def requests(self) -> List[BaseRequest]:
        pass
