from desdeo_problem.Problem import MOProblem


class InteractiveMethod:
    """The base class for interactive methods.

    Args:
        problem (MOProblem): The problem being solved in an interactive method.
    """

    def __init__(self, problem: MOProblem):
        self._problem = problem
