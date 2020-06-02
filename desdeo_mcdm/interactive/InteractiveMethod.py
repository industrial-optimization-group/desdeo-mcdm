from typing import List

from desdeo_problem.Problem import MOProblem

from desdeo_tools.interaction.request import BaseRequest


class InteractiveMethod:
    def __init__(self, problem: MOProblem):
        self._problem = problem
