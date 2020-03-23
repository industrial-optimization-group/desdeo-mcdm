"""Implements various useful solvers.

"""
import numpy as np

import logging

# this has to be fixed, importing ProblemBase enables DEBUG
# logging for all modules, causes matplotlib spam...
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

from desdeo_problem.Problem import ProblemBase

from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer


def payoff_table_method(problem: ProblemBase) -> np.ndarray:
    pass


if __name__ == "__main__":
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder
    from desdeo_tools.scalarization.Scalarizer import Scalarizer

    # create the problem
    def f_1(x):
        res = 4.07 + 2.27 * x[:, 0]
        return -res

    def f_2(x):
        res = (
            2.60
            + 0.03 * x[:, 0]
            + 0.02 * x[:, 1]
            + 0.01 / (1.39 - x[:, 0] ** 2)
            + 0.30 / (1.39 - x[:, 1] ** 2)
        )
        return -res

    def f_3(x):
        res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
        return -res

    def f_4(x):
        res = 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)
        return -res

    def f_5(x):
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    f1 = _ScalarObjective(name="f1", evaluator=f_1)
    f2 = _ScalarObjective(name="f2", evaluator=f_2)
    f3 = _ScalarObjective(name="f3", evaluator=f_3)
    f4 = _ScalarObjective(name="f4", evaluator=f_4)
    f5 = _ScalarObjective(name="f5", evaluator=f_5)
    varsl = variable_builder(
        ["x_1", "x_2"],
        initial_values=[0.5, 0.5],
        lower_bounds=[0.3, 0.3],
        upper_bounds=[1.0, 1.0],
    )
    problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5])
    scalarizer = Scalarizer(
        lambda xs: problem.evaluate(xs).objectives,
        lambda ys: np.sum(ys, axis=1),
    )
    # res = scalarizer(np.array([[0.5, 0.5], [0.4, 0.4]]))
    # print(problem.get_variable_bounds())
    solver = ScalarMinimizer(
        scalarizer,
        problem.get_variable_bounds(),
        # lambda xs: problem.evaluate(xs).constraints,
        None,
        None,
    )
    opt_res = solver.minimize(np.array([0.5, 0.5]))
    print(opt_res.x)
