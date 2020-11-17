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


class RPMInitialRequest(BaseRequest):
    """
    A request class to handle the initial preferences.
    """

    def __init__(self, ideal: np.ndarray, nadir: np.ndarray):
        """
        Initialize with ideal and nadir vectors.
        Args:
            ideal (np.ndarray): Ideal vector.
            nadir (np.ndarray): Nadir vector.
        """

        self.n_objectives = len(ideal)
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
        Set Decision maker's response information for initial request.
        Args:
            response (Dict): Decision maker's response.
        """

        # validate_response(self.n_objectives, z_current=self._nadir, nadir=self._nadir, response=response,
        #                 first_iteration_bool=True)
        self._response = response


class RPMRequest(BaseRequest):
    """
    A request class to handle the intermediate requests.
    """

    def __init__(
            self,
            z_current: np.ndarray,
            nadir: np.ndarray,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray,
            distance: np.ndarray,
    ):
        """
        Initialize request with current iterations's solution process information.
        Args:
            z_current (np.ndarray): Current iteration point.
            nadir (np.ndarray): Nadir point.
            lower_bounds (np.ndarray): Lower bounds for objective functions for next iteration.
            upper_bounds (np.ndarray): Upper bounds for objective functions for next iteration.
            distance (np.ndarray): Closeness to Pareto optimal front.
        """

        self._n_objectives = len(nadir)
        self._z_current = z_current
        self._nadir = nadir

        msg = (
            "In case you wish to change the number of remaining iterations, please specify the number as "
            "'n_iterations'.\n "
            "In case you wish to take a step back to the previous iteration point, please state 'True' as "
            "'step_back'. "
            "Otherwise state 'False' as 'step_back'\n"
            "In case you wish to take a step back and take a shorter step with the previous preference information,"
            "please state 'True' as 'short_step'. Otherwise, please state 'False' as 'short_step'. \n"
            "In case you wish to use preference information from previous iteration, please state 'True' as "
            "'use_previous_preference'. Otherwise state 'False' as 'use_previous_preference' \n"
            "In case you chose to not to use preference information from previous iteration, \n"
            "Please specify as 'preference_method' whether to \n"
            "1. Give directly components for direction of improvement.\n"
            "2. Give improvement ratios between two different objectives. Choose one objective's improvement ratio as 1,"
            "and specify other objectives' improvement ratios in relatation to that."
            "3. Give a pair of objectives (i, j) and provide a value T > 0 as the desirable improvement ratio of this pair."
            "For example: [((1,2), 2), ((1,3), 1), ((3,4), 1.5)]."
            "Depending on your selection on 'preference_method', please specify either the direct components, "
            "improvement ratios or objective pairs and values of T for each objective as 'preference_info'."

        )
        content = {
            "message": msg,
            "current_iteration_point": z_current,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "distance": distance,
        }

        super().__init__("reference_point_preference", "required", content=content)

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set Decision maker's response information for request.
        Args:
            response (Dict): Decision maker's response.
        """

        # validate_response(self._n_objectives, self._z_current, self._nadir, response, first_iteration_bool=False)
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
    TODO: Docstring
    """

    def __init__(
            self,
            problem: MOProblem,
            starting_point: np.ndarray,
            ideal: np.ndarray,
            nadir: np.ndarray,
            epsilon: float = 1e-4,
            objective_names: Optional[List[str]] = None,
            minimize: Optional[List[int]] = None,
    ):

        if not ideal.shape == nadir.shape:
            raise RPMException("The dimensions of the ideal and nadir point do not match.")

        if not ideal.shape == starting_point.shape:
            raise RPMException("The dimension of the ideal and starting point do not match.")

        if all(np.less(nadir, starting_point)):
            raise RPMException("Starting point cannot be worse than nadir point.")

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
        self._starting_point = starting_point
        self._n_objectives = self._ideal.shape[0]

        # current iteration step number
        self._h = 1

        # solutions, objectives, distances and iteration points for each iteration
        self._xs = [None] * 10
        self._fs = [None] * 10
        self._ds = [None] * 10
        self._zs = [None] * 10

        # perturbed reference points
        self._pqs = [None] * 10

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
            Union[RPMRequest, RPMStopRequest]: A new request with content depending on the Decision maker's
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
        Handles the initial request by parsing the response appropiately.

        Args:
            request (RPMInitialRequest): Initial request including Decision maker's initial preferences.
        Returns:
            RPMRequest: New request with updated solution process information.
        """

        # set initial iteration point
        self._zs[self._h - 1] = self._starting_point

        # set reference point
        self._q = self._zs[self._h - 1]

        # set weighting vector
        self._w = self._q / (self._utopian - self._nadir)

        # set initial values for decision variables
        x0 = self._problem.get_variable_upper_bounds() / 2

        # solve the ASF-problem
        result = self.solve_asf(self._q, x0, self._w, self._nadir, self._ideal, self._objectives,
                                self._variable_bounds, method=self._method_de)

        # update current solution and objective function values
        self._xs[self._h] = result["x"]
        self._fs[self._h] = self._objectives(self._xs[self._h])[0]

        # calculate perturbed reference points
        self._pqs[self._h] = self.calculate_prp(self._q, self._fs[self._h])

        # calculate a number of n_objectives other solutions
        res = [self.solve_asf(pqi, x0, self._w, self._nadir, self._ideal, self._objectives,
                              self._variable_bounds, self._method_de) for pqi in self._pqs[self._h]]

        # TODO: Continue from here, store perturbed results to array, present to DM.

        p = 2

    def calculate_prp(self, ref_point: np.ndarray, f_current: np.ndarray):
        """
        Calculate perturbed reference points.

        Args:
            ref_point (np.ndarray): Current reference point
            f_current (np.ndarray): Current solution.

        Returns:
            np.ndarray: Perturbed reference points
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

    def handle_request(self, request: RPMRequest) -> Union[RPMRequest, RPMStopRequest]:
        """
        Handle Decision maker's intermediate requests.
        Args:
            request (RPMRequest): Intermediate request including Decision maker's response.
        Returns:
            Union[RPMRequest, RPMStopRequest]: In case last iteration, request to stop the solution process.
            Otherwise, new request with updated solution process information.
        """

        resp: dict = request.response

        # TODO: handle request

    def solve_asf(self,
                  ref_point: np.ndarray,
                  x0: np.ndarray,
                  preference_factors: np.ndarray,
                  nadir: np.ndarray,
                  utopian: np.ndarray,
                  objectives: Callable,
                  variable_bounds: Optional[np.ndarray],
                  method: Union[ScalarMethod, str, None]
                  ) -> dict:
        """
        Solve Achievement scalarizing function.
        Args:
            ref_point (np.ndarray): Reference point.
            x0 (np.ndarray): Initial values for decison variables.
            preference_factors (np.ndarray): Preference factors on how much would the decision maker wish to improve
                                             the values of each objective function.
            nadir (np.ndarray): Nadir vector.
            utopian (np.ndarray): Utopian vector.
            objectives (np.ndarray): The objective function values for each input vector.
            variable_bounds (Optional[np.ndarray): Lower and upper bounds of each variable
                                                   as a 2D numpy array. If undefined variables, None instead.
            method (Union[ScalarMethod, str, None): The optimization method the scalarizer should be minimized with
        Returns:
            Dict: A dictionary with at least the following entries: 'x' indicating the optimal variables found,
            'fun' the optimal value of the optimized functoin, and 'success' a boolean indicating whether
            the optimization was conducted successfully.
        """

        # scalarize problem using reference point
        asf = ReferencePointASF([1 / preference_factors], nadir, utopian, rho=1e-5)
        asf_scalarizer = Scalarizer(
            evaluator=objectives,
            scalarizer=asf,
            scalarizer_args={"reference_point": ref_point})

        # minimize
        minimizer = ScalarMinimizer(asf_scalarizer, variable_bounds, method=method)
        return minimizer.minimize(x0)

    def calculate_iteration_point(self, itn: int, z_prev: np.ndarray, f_current: np.ndarray) -> np.ndarray:
        """
        Calculate next iteration point towards the Pareto optimal solution.
        Args:
            itn (int): Number of iterations left.
            z_prev(np.ndarray): Previous iteration point.
            f_current (np.ndarray): Current optimal objective vector.
        Returns:
            np.ndarray: Next iteration point.
        """

        return (((itn - 1) / itn) * z_prev) + ((1 / itn) * f_current)

    def calculate_distance(self, z_current: np.ndarray, starting_point: np.ndarray,
                           f_current: np.ndarray) -> np.ndarray:
        """
        Calculates the distance from current iteration point to the Pareto optimal set.
        Args:
            z_current (np.ndarray): Current iteration point.
            starting_point (np.ndarray): Starting iteration point.
            f_current (np.ndarray): Current optimal objective vector.
        Returns:
            np.ndarray: Distance to the Pareto optimal set.
        """

        dist = (np.linalg.norm(np.atleast_2d(z_current) - starting_point, ord=2, axis=1)) \
               / (np.linalg.norm(np.atleast_2d(f_current) - starting_point, ord=2, axis=1))
        return dist * 100


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

    # starting point
    z0 = np.array(([-4.07, -2.82, -3, 4]))

    # start solving
    method = ReferencePointMethod(problem=prob, starting_point=z0, ideal=ideal, nadir=nadir)

    print("Let's start solving\n")
    req = method.start()

    # initial preferences
    n_iterations = 3
    req.response = {
        "n_iterations": n_iterations,
        "preference_method": 3,  # pairs
        # remember to specify "dtype=object" when using preference method 3.
        "preference_info": np.array([((1, 2), 0.5), ((3, 4), 1), ((2, 3), 1.5)], dtype=object)
    }
    print("Step number: 0")
    print("Iteration point: ", nadir)
    print("Lower bounds of objectives: ", ideal)

    # 1 - continue with same preferences
    req = method.iterate(req)
    print("\nStep number: ", method._h)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._h])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    # print("Upper bounds of objectives:", req.content["upper_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])

    req.response = {
        "step_back": False,
        "short_step": False,
        "use_previous_preference": False,
        "preference_method": 3,  # deltas directly
        "preference_info": np.array([((1, 3), 0.5), ((2, 4), 1), ((2, 3), (2 / 3))], dtype=object),

    }

    # 2 - take a step back and give new preferences
    req = method.iterate(req)
    print("\nStep number: ", method._h)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._h])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])

    req.response = {
        "step_back": False,
        "short_step": False,
        "use_previous_preference": False,
        "preference_method": 1,  # deltas directly
        "preference_info": np.array([2, 1, 5, 10]),
    }

    # 3 - give new preferences
    req = method.iterate(req)
    print("\nStep number: ", method._h)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._h])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])

    # give last iteration preferences
    req.response = {
        "step_back": False,
        "use_previous_preference": False,
        "preference_method": 1,  # deltas directly
        "preference_info": np.array([1, 2, 1, 2]),
    }

    req = method.iterate(req)
    print("\nStep number: ", method._h)
    print(req.content["message"])
    print("Solution: ", req.content["solution"])
    print("Objective function values: ", req.content["objective_vector"])
