"""Implements various useful solvers.

"""
import numpy as np

import logging
from typing import Tuple

# this has to be fixed, importing ProblemBase enables DEBUG
# logging for all modules, causes matplotlib spam...
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

from desdeo_problem.Problem import MOProblem

from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer


def weighted_scalarizer(xs: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """A simple linear weight based scalarizer.
    
    Args:
        xs (np.ndarray): Values to be scalarized.
        ws (np.ndarray): Weights to multiply each value in the summation of xs.
    
    Returns:
        np.ndarray: An array of scalar values with length equal to the first dimension of xs.
    """
    return np.sum(np.atleast_2d(xs) * ws, axis=1)


def payoff_table_method(
    problem: MOProblem, initial_guess: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Uses the payoff table method to solve for the ideal and nadir points of a MOProblem.
    
    Args:
        problem (MOProblem): The problem defined as a MOProblem class instance.
        initial_guess (np.ndarray): The initial guess of decision variables
        to be used in the solver. If None, uses the lower bounds defined for
        the variables in MOProblem. Defaults to None.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ideal and nadir points
    """
    scalarizer = Scalarizer(
        lambda xs: problem.evaluate(xs).objectives,
        weighted_scalarizer,
        scalarizer_args={"ws": None},
    )

    if problem.n_of_constraints > 0:
        constraints = (lambda xs: problem.evaluate(xs).constraints,)
    else:
        constraints = None

    solver = ScalarMinimizer(
        scalarizer, problem.get_variable_bounds(), constraints, None,
    )

    ws = np.eye(problem.n_of_objectives)
    po_table = np.zeros((problem.n_of_objectives, problem.n_of_objectives))
    if initial_guess is None:
        initial_guess = problem.get_variable_lower_bounds()

    for i in range(problem.n_of_objectives):
        scalarizer._scalarizer_args = {"ws": ws[i]}
        opt_res = solver.minimize(initial_guess)
        po_table[i] = problem.evaluate(opt_res.x).objectives

    ideal = np.diag(po_table)
    nadir = np.max(po_table, axis=0)

    return ideal, nadir


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
    res = payoff_table_method(problem)
    print(res)
    # scalarizer = Scalarizer(
    #     lambda xs: problem.evaluate(xs).objectives,
    #     weighted_scalarizer,
    #     scalarizer_args={"ws": np.ones(5)},
    # )
    # # res = scalarizer(np.array([[0.5, 0.5], [0.4, 0.4]]))
    # # print(problem.get_variable_bounds())
    # solver = ScalarMinimizer(
    #     scalarizer,
    #     problem.get_variable_bounds(),
    #     # lambda xs: problem.evaluate(xs).constraints,
    #     None,
    #     None,
    # )
    # opt_res = solver.minimize(np.array([0.5, 0.5]))
    # print(opt_res.x)
