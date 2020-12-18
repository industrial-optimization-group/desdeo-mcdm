from typing import Dict, List, Optional, Union, Callable

import numpy as np
from desdeo_problem.Objective import _ScalarObjective, VectorObjective
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Variable import variable_builder

from desdeo_tools.interaction.request import BaseRequest
from desdeo_tools.scalarization import ReferencePointASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer, ScalarMethod

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod

from scipy.optimize import differential_evolution

"""
Reference Point Method (RPM)
"""


class RPMException(Exception):
    """
    Raised when an exception related to Reference Point Method (RFM) is encountered.
    """

    pass


def validate_reference_point(ref_point: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> None:
    """
    Validate Decion maker's reference point.

    Args:
        ref_point (np.ndarray): Reference point.
        ideal (np.ndarray): Ideal vector.
        nadir (np.ndarray): Nadir vector.

    Returns:

    Raises:
        RPMException: In case reference point is invalid.

    """

    if not ideal.shape == ref_point.shape:
        raise RPMException("The dimension of the ideal and reference point do not match.")

    if all(np.less(nadir, ref_point)):
        raise RPMException("Reference point cannot be worse than nadir point.")  # or can it?


class RPMInitialRequest(BaseRequest):
    """
    A request class to handle the Decision Maker's initial preferences for the first iteration round.
    """

    def __init__(self, ideal: np.ndarray, nadir: np.ndarray) -> None:
        """
        Initialize with ideal and nadir vectors.

        Args:
            ideal (np.ndarray): Ideal vector.
            nadir (np.ndarray): Nadir vector.
        """

        self.n_objectives = len(ideal)
        self._ideal = ideal
        self._nadir = nadir

        msg = (
            "Please specify a reference point as 'reference_point'."
        )
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
        }

        super().__init__("reference_point_preference", "required", content=content)

    @classmethod
    def init_with_method(cls, method: InteractiveMethod):
        """
        Initialize request with given instance of ReferencePointMethod.

        Args:
            method (ReferencePointMethod): Instance of ReferencePointMethod-class.
        Returns:
            RPMInitialRequest: Initial request.
        """

        return cls(method._ideal, method._nadir)

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for initial request.

        Args:
            response (Dict): The Decision Maker's response.

        Raises:
            RPMException: In case reference point is missing.
        """

        if 'reference_point' not in response:
            msg = "Reference point missing. Please specify a reference point as 'reference_point."
            raise RPMException(msg)
        else:
            validate_reference_point(response['reference_point'], self._ideal, self._nadir)

        self._response = response


class RPMRequest(BaseRequest):
    """
    A request class to handle the Decision Maker's preferences after the first iteration round.
    """

    def __init__(
            self,
            f_current: np.ndarray,
            f_additionals: np.ndarray,
            ideal: np.ndarray,
            nadir: np.ndarray,
    ) -> None:
        """
        Initialize request with current iterations's solution process information.

        Args:
            f_current (np.ndarray): Current solution.
            f_additionals (np.ndarray): Additional solutions.
            ideal (np.ndarray): Idea vector.
            nadir (np.ndarray): Nadir vector.
        """

        self._f_current = f_current
        self._f_additionals = f_additionals
        self._ideal = ideal
        self._nadir = nadir

        msg = (
            "In case you are satisfied with one of the solutions, please state:  "
            "1. 'satisfied' as 'True'."
            "2. 'solution_index' as the index number of the solution you choose, so that first solution has index "
            "number of 0, second 1 and so on."
            "Otherwise, please state 'satisfied' as 'False and specify a new reference point as 'reference_point'."
        )

        content = {
            "message": msg,
            "current_solution": f_current,
            "additional_solutions": f_additionals
        }

        super().__init__("reference_point_preference", "required", content=content)

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for request.

        Args:
            response (Dict): The Decision Maker's response.

        Raises:
            RPMException: In case response is invalid.
        """

        if 'satisfied' in response and response['satisfied']:
            if not response['solution_index']:
                raise RPMException("If you are satisfied with one of the solutions, please specify the index of the "
                                   "solution as 'solution_index'.")
            if not (0 <= response['solution_index'] <= self._f_current.shape[0]):
                msg = "Solution index must range from 0 to number of objectives - 1 '{}'. Given solution index: '{}." \
                    .format(self._f_current.shape[0], response['solution_index'])
                raise RPMException(msg)
        else:
            if 'reference_point' not in response:
                raise RPMException("New reference point information missing. Please specify it as 'reference_point'.")
            else:
                validate_reference_point(response['reference_point'], self._ideal, self._nadir)

        self._response = response


class RPMStopRequest(BaseRequest):
    """
    A request class to handle termination.
    """

    def __init__(self, x_h: np.ndarray, f_h: np.ndarray) -> None:
        """
        Initialize termination request with final solution and objective vector.

        Args:
            x_h (np.ndarray): Solution (decision variables).
            f_h (np.ndarray): Objective vector.
        """

        msg = "Final solution found."
        content = {"message": msg, "solution": x_h, "objective_vector": f_h}

        super().__init__("print", "no_interaction", content=content)


class ReferencePointMethod(InteractiveMethod):
    """
    Implements the Reference Point Method as presented in |Wierzbicki_1982|.

    In the Reference Point Method, the Decision Maker (DM) specifies **desirable aspiration levels** for objective
    functions. Vectors formed of these aspiration levels are then used to derive scalarizing functions having minimal
    values at weakly, properly or Pareto optimal solutions. It is important that reference points are intuitive and easy
    for the DM to specify, their consistency is not an essential requirement. Before the solution process starts, some
    information is given to the DM about the problem. If possible, the ideal objective vector and the
    (approximated) nadir objective vector are presented.

    At each iteration, the DM is asked to give desired aspiration levels for the objective functions. Using this
    information to formulate a **reference point**, achievement function is minimized and a (weakly, properly or) Pareto
    optimal solution is obtained. This solution is then presented to the DM. In addition, k other (weakly,
    properly or) Pareto optimal solutions are calculated using **perturbed reference points**, where k is the
    number of objectives in the problem. The alternative solutions are also presented to the DM. If (s)he finds any of
    the k + 1 solutions satisfactory, the solution process is ended. Otherwise, the DM is asked to present a new
    reference point and the iteration described above is repeated.

    The idea in perturbed reference points is that the DM gets **better understanding** of the possible solutions around
    the current solution. If the reference point is far from the Pareto optimal set, the DM gets a **wider** description
    of the Pareto optimal set and if the reference point is near the Pareto optimal set, then a **finer** description of
    the Pareto optimal set is given.

    In this method, the DM has to specify aspiration levels and compare objective vectors. The DM is **free to change**
    her/his mind during the process and can direct the solution process without being forced to understand complicated
    concepts and their meaning. On the other hand, the method does not necessarily help the DM to find more satisfactory
    solutions.

    Args:
        problem (MOProblem): Problem to be solved.
        ideal (np.ndarray): The ideal objective vector of the problem.
        nadir (np.ndarray): The nadir objective vector of the problem. This may also be the "worst" objective vector
                            provided by the Decision Maker if the approximation of Nadir vector is not applicable or if
                            the Decision Maker wishes to provide even worse objective vector than what the
                            approximated Nadir vector is.
        epsilon (float): A small number used in calculating the utopian point.
        epsilon (float): A small number used in calculating the utopian point. Default value is 1e-6.
        objective_names (Optional[List[str]], optional): Names of the objectives. The length of the list must match the
                                                         number of elements in ideal vector.
        minimize (Optional[List[int]], optional): Multipliers for each objective. '-1' indicates maximization
                                                  and '1' minimization. Defaults to all objective values being
                                                  minimized.

    Raises:
        RPMException: Dimensions of ideal, nadir, objective_names, and minimize-list do not match.
    """

    def __init__(
            self,
            problem: MOProblem,
            ideal: np.ndarray,
            nadir: np.ndarray,
            epsilon: float = 1e-6,
            objective_names: Optional[List[str]] = None,
            minimize: Optional[List[int]] = None,
    ):

        if not ideal.shape == nadir.shape:
            raise RPMException("The dimensions of the ideal and nadir point do not match.")

        if objective_names:
            if not len(objective_names) == ideal.shape[0]:
                raise RPMException(
                    "The supplied objective names must have a leangth equal to " "the number of objectives."
                )
            self._objective_names = objective_names
        else:
            self._objective_names = [f"f{i + 1}" for i in range(ideal.shape[0])]

        if minimize:
            if not len(objective_names) == ideal.shape[0]:
                raise RPMException("The minimize list must have " "as many elements as there are objectives.")
            self._minimize = minimize
        else:
            self._minimize = [1 for _ in range(ideal.shape[0])]

        # initialize method with problem
        super().__init__(problem)
        self._problem = problem
        self._objectives: Callable = lambda x: self._problem.evaluate(x).objectives
        self._variable_bounds: Union[np.ndarray, None] = problem.get_variable_bounds()
        self._constraints: Optional[Callable] = lambda x: self._problem.evaluate(x).constraints

        self._ideal = ideal
        self._nadir = nadir
        self._utopian = ideal - epsilon
        self._n_objectives = self._ideal.shape[0]

        # current iteration step number
        self._h = 1

        # solutions in decision and objective space, distances and referation points for each iteration
        self._xs = [None] * 10
        self._fs = [None] * 10
        self._ds = [None] * 10
        self._qs = [None] * 10

        # perturbed reference points
        self._pqs = [None] * 10

        # additional solutions
        self._axs = [None] * 10
        self._afs = [None] * 10

        # current reference point
        self._q: Union[None, np.ndarray] = None

        # weighting vector for achievement function
        self._w: np.ndarray = []

        # evolutionary method for minimizing
        self._method_de: ScalarMethod = ScalarMethod(
            lambda x, _, **y: differential_evolution(x, **y),
            method_args={"disp": False, "polish": False, "tol": 0.000001, "popsize": 10, "maxiter": 50000},
            use_scipy=True
        )

    def start(self) -> RPMInitialRequest:
        """
        Start the solution process with initializing the first request.

        Returns:
            RPMInitialRequest: Initial request.
        """

        return RPMInitialRequest.init_with_method(self)

    def iterate(
            self, request: Union[RPMInitialRequest, RPMRequest, RPMStopRequest]
    ) -> Union[RPMRequest, RPMStopRequest]:
        """
        Perform the next logical iteration step based on the given request type.

        Args:
            request (Union[RPMInitialRequest, RPMRequest]): Either initial or intermediate request.

        Returns:
            Union[RPMRequest, RPMStopRequest]: A new request with content depending on the Decision Maker's
            preferences.
        """

        if type(request) is RPMInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is RPMRequest:
            return self.handle_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(self, request: RPMInitialRequest) -> RPMRequest:
        """
        Handles the initial request by parsing the response appropriately.

        Args:
            request (RPMInitialRequest): Initial request including the Decision Maker's initial preferences.

        Returns:
            RPMRequest: New request with updated solution process information.
        """

        # set initial referation point
        self._qs[self._h] = request.response["reference_point"]
        self._q = self._qs[self._h]

        # set weighting vector
        self._w = self._q / (self._utopian - self._nadir)

        # set initial values for decision variables
        x0 = self._problem.get_variable_upper_bounds() / 2

        # solve the ASF-problem
        result = self.solve_asf(self._q, x0, self._w, self._nadir, self._utopian, self._objectives,
                                self._variable_bounds, method=self._method_de)

        # update current solution and objective function values
        self._xs[self._h] = result["x"]
        self._fs[self._h] = self._objectives(self._xs[self._h])[0]

        # calculate perturbed reference points
        self._pqs[self._h] = self.calculate_prp(self._q, self._fs[self._h])

        # calculate n other solutions with perturbed reference points
        results_additional = [self.solve_asf(pqi, x0, self._w, self._nadir, self._utopian, self._objectives,
                                             self._variable_bounds, self._method_de) for pqi in self._pqs[self._h]]

        # store results into arrays
        self._axs[self._h] = [result["x"] for result in results_additional]
        self._afs[self._h] = [self._objectives(xs_i)[0] for xs_i in self._axs[self._h]]

        # return the information from iteration round to be shown to the DM.
        return RPMRequest(
            self._fs[self._h], self._afs[self._h], self._ideal, self._nadir
        )

    def handle_request(self, request: RPMRequest) -> Union[RPMRequest, RPMStopRequest]:
        """
        Handle the Decision Maker's requests after the first iteration round, so-called **intermediate requests.**

        Args:
            request (RPMRequest): Intermediate request including the Decision Maker's response.

        Returns:
            Union[RPMRequest, RPMStopRequest]: In case last iteration, request to stop the solution process.
            Otherwise, new request with updated solution process information.
        """

        resp: dict = request.response

        # end solution finding process
        if 'satisfied' in resp and resp['satisfied']:
            if resp['solution_index'] == 0:  # "original" solution
                return RPMStopRequest(self._xs[self._h], self._fs[self._h])
            else:  # additional solution
                return RPMStopRequest(self._axs[self._h][resp['solution_index'] - 1],
                                      self._afs[self._h][resp['solution_index'] - 1])

        # continue with new reference point given by the DM
        else:
            self._h += 1

            if len(self._qs) - self._h <= 2:
                # "expand" space on arrays
                extra_space = [None] * 10
                self._qs = np.array(np.concatenate((self._qs, extra_space), axis=None), dtype=object)
                self._xs = np.array(np.concatenate((self._xs, extra_space), axis=None), dtype=object)
                self._fs = np.array(np.concatenate((self._fs, extra_space), axis=None), dtype=object)
                self._pqs = np.array(np.concatenate((self._pqs, extra_space), axis=None), dtype=object)
                self._axs = np.array(np.concatenate((self._axs, extra_space), axis=None), dtype=object)
                self._afs = np.array(np.concatenate((self._afs, extra_space), axis=None), dtype=object)

            # set new reference point
            self._qs[self._h] = resp['reference_point']
            self._q = self._qs[self._h]

            # set weighting vector
            self._w = self._q / (self._utopian - self._nadir)

            # set initial values for decision variables
            x0 = self._problem.get_variable_upper_bounds() / 2

            # solve the ASF-problem
            result = self.solve_asf(self._q, x0, self._w, self._nadir, self._utopian, self._objectives,
                                    self._variable_bounds, method=self._method_de)

            # update current solution and objective function values
            self._xs[self._h] = result["x"]
            self._fs[self._h] = self._objectives(self._xs[self._h])[0]

            # calculate perturbed reference points
            self._pqs[self._h] = self.calculate_prp(self._q, self._fs[self._h])

            # calculate n other solutions with perturbed reference points
            results_additional = [self.solve_asf(pqi, x0, self._w, self._nadir, self._utopian, self._objectives,
                                                 self._variable_bounds, self._method_de) for pqi in self._pqs[self._h]]

            # store results into arrays
            self._axs[self._h] = [result["x"] for result in results_additional]
            self._afs[self._h] = [self._objectives(xs_i)[0] for xs_i in self._axs[self._h]]

            # return the information from iteration round to be shown to the DM.
            return RPMRequest(
                self._fs[self._h], self._afs[self._h], self._ideal, self._nadir
            )

    def calculate_prp(self, ref_point: np.ndarray, f_current: np.ndarray) -> np.ndarray:
        """
        Calculate perturbed reference points.

        Args:
            ref_point (np.ndarray): Current reference point.
            f_current (np.ndarray): Current solution.

        Returns:
            np.ndarray: Perturbed reference points.
        """

        # distance
        d = np.linalg.norm(np.atleast_2d(ref_point - f_current))

        # unit vectors
        ei = np.array([np.zeros(len(ref_point))])
        es = np.repeat(ei, len(ref_point), axis=0)

        for i, j in enumerate(es):
            for ind, _ in enumerate(j):
                if ind == i:
                    j[ind] = 1

        return ref_point + (d * es)

    def solve_asf(self,
                  ref_point: np.ndarray,
                  x0: np.ndarray,
                  preferential_factors: np.ndarray,
                  nadir: np.ndarray,
                  utopian: np.ndarray,
                  objectives: Callable,
                  variable_bounds: Optional[np.ndarray] = None,
                  method: Union[ScalarMethod, str, None] = None
                  ) -> dict:
        """
        Solve Achievement scalarizing function.

        Args:
            ref_point (np.ndarray): Reference point.
            x0 (np.ndarray): Initial values for decision variables.
            preferential_factors (np.ndarray): Preferential factors on how much would the Decision Maker wish to improve
                                             the values of each objective function.
            nadir (np.ndarray): Nadir vector.
            utopian (np.ndarray): Utopian vector.
            objectives (np.ndarray): The objective function values for each input vector.
            variable_bounds (Optional[np.ndarray)]: Lower and upper bounds of each variable
                                                   as a 2D numpy array. If undefined variables, None instead.
            method (Union[ScalarMethod, str, None]): The optimization method the scalarizer should be minimized with.

        Returns:
            dict: A dictionary with at least the following entries: 'x' indicating the optimal variables found,
            'fun' the optimal value of the optimized function, and 'success' a boolean indicating whether
            the optimization was conducted successfully.
        """

        if variable_bounds is None:
            # set all bounds as [-inf, inf]
            variable_bounds = np.array([[-np.inf, np.inf]] * x0.shape[0])

        # scalarize problem using reference point
        asf = ReferencePointASF(preferential_factors, nadir, utopian, rho=1e-4)
        asf_scalarizer = Scalarizer(
            evaluator=objectives,
            scalarizer=asf,
            scalarizer_args={"reference_point": ref_point})

        # minimize
        minimizer = ScalarMinimizer(asf_scalarizer, variable_bounds, method=method)
        return minimizer.minimize(x0)


# testing the method
if __name__ == "__main__":
    print("Reference point method")

    # Objectives
    def f1(xs):
        xs = np.atleast_2d(xs)
        return -4.07 - 2.27 * xs[:, 0]


    def f2(xs):
        xs = np.atleast_2d(xs)
        return -2.60 - 0.03 * xs[:, 0] - 0.02 * xs[:, 1] - (0.01 / (1.39 - xs[:, 0] ** 2)) - (
                0.30 / (1.39 - xs[:, 1] ** 2))


    def f3(xs):
        xs = np.atleast_2d(xs)
        return -8.21 + (0.71 / (1.09 - xs[:, 0] ** 2))


    def f4(xs):
        xs = np.atleast_2d(xs)
        return -0.96 + (0.96 / (1.09 - xs[:, 1] ** 2))


    def objectives(xs):
        return np.stack((f1(xs), f2(xs), f3(xs), f4(xs))).T


    obj1 = _ScalarObjective("obj1", f1)
    obj2 = _ScalarObjective("obj2", f2)
    obj3 = _ScalarObjective("obj3", f3)
    obj4 = _ScalarObjective("obj4", f4)

    objkaikki = VectorObjective("obj", objectives)

    # variables
    var_names = ["x1", "x2"]  # Make sure that the variable names are meaningful to you.

    initial_values = np.array([0.5, 0.5])
    lower_bounds = [0.3, 0.3]
    upper_bounds = [1.0, 1.0]
    bounds = np.stack((lower_bounds, upper_bounds))
    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

    # problem
    prob = MOProblem(objectives=[obj1, obj2, obj3, obj4], variables=variables)  # objectives "seperately"

    # solved in Nautilus.py
    ideal = np.array([-6.34, -3.44487179, -7.5, 0])
    nadir = np.array([-4.751, -2.86054116, -0.32111111, 9.70666666])

    # start solving
    method = ReferencePointMethod(problem=prob, ideal=ideal, nadir=nadir)

    # Pareto optimal solution: [-6.30, -3.26, -2.60, 3.63]

    print("Let's start solving\n")
    req = method.start()
    rp = np.array([-5., -3., -3., 5.])
    req.response = {
        "reference_point": rp,
    }

    print("Step number: 0")

    # 1 - continue with same preferences
    req = method.iterate(req)
    step = 1
    print("\nStep number: ", method._h)
    print("Reference point: ", rp)
    print("Pareto optimal solution: ", req.content["current_solution"])
    print("Additional solutions: ", req.content["additional_solutions"])

    while step < 15:
        step +=1
        rp = np.array([np.random.uniform(i, n) for i, n in zip(ideal, nadir)])
        req.response = {
            "reference_point": rp,
        }
        req = method.iterate(req)
        print("\nStep number: ", method._h)
        print("Reference point: ", rp)
        print("Pareto optimal solution: ", req.content["current_solution"])
        print("Additional solutions: ", req.content["additional_solutions"])




