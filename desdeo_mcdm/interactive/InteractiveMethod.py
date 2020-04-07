"""Implements a base class defining a common interface for all interactive methods.
"""

from typing import List

from desdeo_tools.interaction.request import BaseRequest
from desdeo_problem.Problem import MOProblem


class InteractiveMethod:
    def __init__(self, problem: MOProblem):
        self._problem = problem
