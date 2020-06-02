"""Implements various useful solvers.

"""
import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
from desdeo_problem.Problem import MOProblem

from desdeo_tools.scalarization.ASF import ASFBase, PointMethodASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMethod, ScalarMinimizer


class MCDMUtilityException(Exception):
    """Raised when an exception is encountered in some of the utilities.

    """

    pass


def weighted_scalarizer(xs: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """A simple linear weight based scalarizer.
    
    Args:
        xs (np.ndarray): Values to be scalarized.
        ws (np.ndarray): Weights to multiply each value in the summation of xs.
    
    Returns:
        np.ndarray: An array of scalar values with length equal to the first dimension of xs.
    """
    return np.sum(np.atleast_2d(xs) * ws, axis=1)


def payoff_table_method_general(
    objective_evaluator: Callable[[np.ndarray], np.ndarray],
    n_of_objectives: int,
    variable_bounds: np.ndarray,
    constraint_evaluator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    initial_guess: Optional[np.ndarray] = None,
    solver_method: Optional[Union[ScalarMethod, str]] = "scipy_de",
) -> Tuple[np.ndarray, np.ndarray]:
    """Solves a representation for the nadir and ideal points for a
    multiobjective minimization problem with objectives defined as the result
    of some objective evaluator.
    
    Args:
        objective_evaluator (Callable[[np.ndarray], np.ndarray]): The
            evaluator which returns the objective values given a set of
            variabels.
        n_of_objectives (int): Number of objectives returned by calling
            objective_evaluator.
        variable_bounds (np.ndarray): The lower and upper bounds of the
            variables passed as argument to objective_evaluator. Should be a 2D
            numpy array with the limits for each variable being on each row. The
            first column should contain the lower bounds, and the second column
            the upper bounds. Use np.inf to indicate no bounds.
        constraint_evaluator (Optional[Callable[[np.ndarray], np.ndarray]], optional):
            An evaluator accepting the same arguments as
            objective_evaluator, which returns the constraint values of the
            multiobjective minimization problem being solved. A negative
            constraint value indicates a broken constraint. Defaults to None.
        initial_guess (Optional[np.ndarray], optional): The initial guess
            used for the variable values while solving the payoff table. The
            relevancy of this parameter depends on the solver_method being used.
            Defaults to None.
        solver_method (Optional[Union[ScalarMethod, str]], optional): The
            method to solve the scalarized problems in the payoff table method.
            Defaults to "scipy_de", which ignores initial_guess.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The representations computed using the
        payoff table for the ideal and nadir points respectively.
    """
    scalarizer = Scalarizer(objective_evaluator, weighted_scalarizer, scalarizer_args={"ws": None},)

    solver = ScalarMinimizer(scalarizer, variable_bounds, constraint_evaluator, solver_method,)

    ws = np.eye(n_of_objectives)
    po_table = np.zeros((n_of_objectives, n_of_objectives))
    if initial_guess is None:
        initial_guess = variable_bounds[:, 0]

    for i in range(n_of_objectives):
        scalarizer._scalarizer_args = {"ws": ws[i]}
        opt_res = solver.minimize(initial_guess)
        if not opt_res["success"]:
            print("Unsuccessful optimization result encountered while computing a payoff table!")
        po_table[i] = objective_evaluator(opt_res["x"])

    ideal = np.diag(po_table)
    nadir = np.max(po_table, axis=0)

    return ideal, nadir


def payoff_table_method(
    problem: MOProblem,
    initial_guess: Optional[np.ndarray] = None,
    solver_method: Optional[Union[ScalarMethod, str]] = "scipy_de",
) -> Tuple[np.ndarray, np.ndarray]:
    """Uses the payoff table method to solve for the ideal and nadir points of a MOProblem.
    Call through to payoff_table_method_general.
    
    Args:
        problem (MOProblem): The problem defined as a MOProblem class instance.
        initial_guess (Optional[np.ndarray]): The initial guess of decision variables
            to be used in the solver. If None, uses the lower bounds defined for
            the variables in MOProblem. Defaults to None.
        solver_method (Optional[Union[ScalarMethod, str]]): The method used to minimize the
            invidual problems in the payoff table method. Defaults to 'scipy_de'.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The ideal and nadir points
    """
    if problem.n_of_constraints > 0:
        constraints = lambda x: problem.evaluate(x).constraints.squeeze()
    else:
        constraints = None

    return payoff_table_method_general(
        lambda xs: problem.evaluate(xs).objectives,
        problem.n_of_objectives,
        problem.get_variable_bounds(),
        constraints,
        initial_guess,
        solver_method,
    )


def solve_pareto_front_representation_general(
    objective_evaluator: Callable[[np.ndarray], np.ndarray],
    n_of_objectives: int,
    variable_bounds: np.ndarray,
    step: Optional[Union[np.ndarray, float]] = 0.1,
    eps: Optional[float] = 1e-6,
    ideal: Optional[np.ndarray] = None,
    nadir: Optional[np.ndarray] = None,
    constraint_evaluator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    solver_method: Optional[Union[ScalarMethod, str]] = "scipy_de",
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a representation of a Pareto efficient front from a
    multiobjective minimizatino problem. Does so by generating an evenly spaced
    set of reference points (in the objective space), in the space spanned by
    the supplied ideal and nadir points. The generated reference points are
    then used to formulate achievement scalaraization problems, which when
    solved, yield a representation of a Pareto efficient solution. The result
    is guaranteed to contain only non-dominated solutions.
    
    Args:
        objective_evaluator (Callable[[np.ndarray], np.ndarray]): A vector
            valued function returning objective values given an array of decision
            variables.
        n_of_objectives (int): Numbr of objectives returned by
            objective_evaluator.
        variable_bounds (np.ndarray): The upper and lower bounds of the
            decision variables. Bound for each variable should be on the rows,
            with the first column containing lower bounds, and the second column
            upper bounds. Use np.inf to indicate no bounds.
        step (Optional[Union[np.ndarray, float]], optional): Etiher an float
            or an array of floats. If a single float is given, generates
            reference points with the objectives having values a step apart
            between the ideal and nadir points. If an array of floats is given,
            use the steps defined in the array for each objective's values.
            Default to 0.1.
        eps (Optional[float], optional): An offset to be added to the nadir
            value to keep the nadir inside the range when generating reference
            points. Defaults to 1e-6.
        ideal (Optional[np.ndarray], optional): The ideal point of the
            problem being solved. Defaults to None.
        nadir (Optional[np.ndarray], optional): The nadir point of the
            problem being solved. Defaults to None.
        constraint_evaluator (Optional[Callable[[np.ndarray], np.ndarray]], optional):
            An evaluator returning values for the constraints defined
            for the problem. A negative value for a constraint indicates a breach
            of that constraint. Defaults to None.
        solver_method (Optional[Union[ScalarMethod, str]], optional): The
            method used to minimize the achievement scalarization problems
            arising when calculating Pareto efficient solutions. Defaults to
            "scipy_de".

    Raises:
        MCDMUtilityException: Mismatching sizes of the supplied ideal and
        nadir points between the step, when step is an array. Or the type of
        step is something else than np.ndarray of float.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing representationns of
        the Pareto optimal variable values, and the corresponsing objective
        values.
    """
    if ideal is None or nadir is None:
        # compure ideal and nadir using payoff table
        ideal, nadir = payoff_table_method_general(
            objective_evaluator, n_of_objectives, variable_bounds, constraint_evaluator,
        )

    # use ASF to (almost) gurantee Pareto optimality.
    asf = PointMethodASF(nadir, ideal)

    scalarizer = Scalarizer(objective_evaluator, asf, scalarizer_args={"reference_point": None})
    solver = ScalarMinimizer(scalarizer, bounds=variable_bounds, method=solver_method)

    if type(step) is float:
        slices = [slice(start, stop + eps, step) for (start, stop) in zip(ideal, nadir)]

    elif type(step) is np.ndarray:
        if not ideal.shape == nadir.shape == step.shape:
            raise MCDMUtilityException(
                "The shapes of the supplied step array does not match the " "shape of the ideal and nadir points."
            )
        slices = [slice(start, stop + eps, s) for (start, stop, s) in zip(ideal, nadir, step)]

    else:
        raise MCDMUtilityException("step must be either a numpy array or an float.")

    z_mesh = np.mgrid[slices].reshape(len(ideal), -1).T

    p_front_objectives = np.zeros(z_mesh.shape)
    p_front_variables = np.zeros((len(p_front_objectives), len(variable_bounds.squeeze())))

    for i, z in enumerate(z_mesh):
        scalarizer._scalarizer_args = {"reference_point": z}
        res = solver.minimize(None)

        if not res["success"]:
            print("Non successfull optimization")
            p_front_objectives[i] = np.nan
            p_front_variables[i] = np.nan
            continue

        # check for dominance, accept only non-dominated solutions
        f_i = objective_evaluator(res["x"])
        if not np.all(f_i > p_front_objectives[:i][~np.all(np.isnan(p_front_objectives[:i]), axis=1)]):
            p_front_objectives[i] = f_i
            p_front_variables[i] = res["x"]
        elif i < 1:
            p_front_objectives[i] = f_i
            p_front_variables[i] = res["x"]
        else:
            print(f"{f_i} is dominated by {p_front_objectives[:i][np.all(f_i > p_front_objectives[:i], axis=1)]}")
            p_front_objectives[i] = np.nan
            p_front_variables[i] = np.nan

    return (
        p_front_variables[~np.all(np.isnan(p_front_variables), axis=1)],
        p_front_objectives[~np.all(np.isnan(p_front_objectives), axis=1)],
    )


def solve_pareto_front_representation(
    problem: MOProblem,
    step: Optional[Union[np.ndarray, float]] = 0.1,
    eps: Optional[float] = 1e-6,
    solver_method: Optional[Union[ScalarMethod, str]] = "scipy_de",
) -> Tuple[np.ndarray, np.ndarray]:
    """Pass through to solve_pareto_front_representation_general when the
    problem for which the front is being calculated for is defined as an
    MOProblem object.
    
    Computes a representation of a Pareto efficient front
    from a multiobjective minimizatino problem. Does so by generating an
    evenly spaced set of reference points (in the objective space), in the
    space spanned by the supplied ideal and nadir points. The generated
    reference points are then used to formulate achievement scalaraization
    problems, which when solved, yield a representation of a Pareto efficient
    solution.
    
    Args:
        problem (MOProblem): The multiobjective minimization problem for which the front is to be solved for.
            step (Optional[Union[np.ndarray, float]], optional): Etiher an float
            or an array of floats. If a single float is given, generates
            reference points with the objectives having values a step apart
            between the ideal and nadir points. If an array of floats is given,
            use the steps defined in the array for each objective's values.
            Default to 0.1.
        eps (Optional[float], optional): An offset to be added to the nadir
            value to keep the nadir inside the range when generating reference
            points. Defaults to 1e-6.
        solver_method (Optional[Union[ScalarMethod, str]], optional): The
            method used to minimize the achievement scalarization problems
            arising when calculating Pareto efficient solutions. Defaults to
            "scipy_de".
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing representationns of
        the Pareto optimal variable values, and the corresponsing objective
        values.
    """
    if problem.n_of_constraints > 0:
        constraints = lambda x: problem.evaluate(x).constraints.squeeze()
    else:
        constraints = None

    return solve_pareto_front_representation_general(
        lambda x: problem.evaluate(x).objectives,
        problem.n_of_objectives,
        problem.get_variable_bounds(),
        step,
        eps,
        problem.ideal,
        problem.nadir,
        constraints,
        solver_method,
    )


if __name__ == "__main__":
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.testproblems.TestProblems import test_problem_builder
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder
    from desdeo_problem.Constraint import ScalarConstraint
    from desdeo_tools.scalarization.Scalarizer import Scalarizer
    from desdeo_tools.solver.ScalarSolver import ScalarMethod

    from scipy.optimize import differential_evolution

    # # create the problem
    # def f_1(x):
    #     res = 4.07 + 2.27 * x[:, 0]
    #     return -res

    # def f_2(x):
    #     res = (
    #         2.60
    #         + 0.03 * x[:, 0]
    #         + 0.02 * x[:, 1]
    #         + 0.01 / (1.39 - x[:, 0] ** 2)
    #         + 0.30 / (1.39 - x[:, 1] ** 2)
    #     )
    #     return -res

    # def f_3(x):
    #     res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
    #     return -res

    # def f_4(x):
    #     res = 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)
    #     return -res

    # def f_5(x):
    #     return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    # def c_1(x, f=None):
    #     x = x.squeeze()
    #     return (x[0] + x[1]) - 0.2

    # f1 = _ScalarObjective(name="f1", evaluator=f_1)
    # f2 = _ScalarObjective(name="f2", evaluator=f_2)
    # f3 = _ScalarObjective(name="f3", evaluator=f_3)
    # f4 = _ScalarObjective(name="f4", evaluator=f_4)
    # f5 = _ScalarObjective(name="f5", evaluator=f_5)
    # c1 = ScalarConstraint("c1", 2, 5, evaluator=c_1)
    # varsl = variable_builder(
    #     ["x_1", "x_2"],
    #     initial_values=[0.5, 0.5],
    #     lower_bounds=[0.3, 0.3],
    #     upper_bounds=[1.0, 1.0],
    # )
    # problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5],)

    scalar_method = ScalarMethod(
        lambda x, _, **ys: differential_evolution(x, **ys),
        method_args={"polish": True, "maxiter": 1000},
        use_scipy=True,
    )

    problem = test_problem_builder("DTLZ1", 3, 2)
    print(problem.get_variable_bounds())
    problem.ideal = np.zeros(2)
    problem.nadir = np.ones(2)

    res = solve_pareto_front_representation_general(
        lambda x: problem.evaluate(x).objectives,
        problem.n_of_objectives,
        problem.get_variable_bounds(),
        step=0.2,
        eps=1e-6,
        ideal=problem.ideal,
        nadir=problem.nadir,
        constraint_evaluator=None,
        solver_method=scalar_method,
    )

    print(res[1].shape)
