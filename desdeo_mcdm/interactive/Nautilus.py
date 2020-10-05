from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np

from desdeo_problem.Variable import variable_builder
from desdeo_problem.Objective import VectorObjective, _ScalarObjective
from desdeo_problem.Constraint import ScalarConstraint
from desdeo_problem.Problem import MOProblem
from desdeo_tools.interaction.request import BaseRequest
from desdeo_tools.scalarization import ReferencePointASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer, ScalarMethod

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod

"""
Epsilon constraint method
"""


class ECMError(Exception):
    """Raised when an error related to the Epsilon Constraint Method is encountered.
    """


class EpsilonConstraintMethod:
    """A class to represent a class for scalarizing MOO problems using the epsilon
        constraint method.
    Attributes:
        objectives (Callable): Objective functions.
        to_be_minimized (int): Integer representing which objective function
        should be minimized.
        epsilons (np.ndarray): Upper bounds chosen by the decison maker.
                               Epsilon constraint functions are defined in a following form:
                                    f_i(x) <= eps_i
                               If the constraint function is of form
                                    f_i(x) >= eps_i
                               Remember to multiply the epsilon value with -1!
        constraints (Optional[Callable]): Function that returns definitions of other constraints, if existing.
    """

    def __init__(
            self, objectives: Callable, to_be_minimized: int, epsilons: np.ndarray,
            constraints: Optional[Callable]
    ):
        self.objectives = objectives
        self._to_be_minimized = to_be_minimized
        self.epsilons = epsilons
        self.constraints = constraints

    def evaluate_constraints(self, xs) -> np.ndarray:
        """
        Returns values of constraints with given decison variables.
        Args:
            xs (np.ndarray): Decision variables.
        Returns: Values of constraint functions (both "original" constraints as well as epsilon constraints) in a vector.
        """
        xs = np.atleast_2d(xs)

        # evaluate epsilon constraint function "left-side" values with given decision variables
        epsilon_left_side = np.array(
            [self.objectives(xs)[0][i] for i, _ in enumerate(self.objectives(xs)[0]) if i != self._to_be_minimized])

        if len(epsilon_left_side) != len(self.epsilons):
            msg = ("The lenght of the epsilons array ({}) must match the total number of objectives - 1 ({})."
                   ).format(len(self.epsilons), len(self.objectives(xs)[0]) - 1)
            raise ECMError(msg)

        # evaluate values of epsilon constraint functions
        e: np.ndarray = np.array([-(f - v) for f, v in zip(epsilon_left_side, self.epsilons)])

        if self.constraints:
            c = self.constraints(xs, 0)
            return np.concatenate([c, e])
        else:
            return e

    def __call__(self, objective_vector: np.ndarray) -> float:
        """
        Returns the value of objective function to be minimized.
        Args:
            objective_vector (np.ndarray): Values of objective functions.
        Returns: Value of objective function to be minimized.
        """
        return objective_vector[0][self._to_be_minimized]


"""
NAUTILUS
"""


def validate_response(n_objectives, response: Dict) -> None:
    """
    Validate decision maker's response.
    """
    if "n_iterations" not in response:
        raise NautilusException("'n_iterations' entry missing")
    validate_preferences(n_objectives, response)
    n_iterations = response["n_iterations"]

    if not isinstance(n_iterations, int) or int(n_iterations) < 1:
        raise NautilusException("'n_iterations' must be a positive integer greater than zero")


def validate_preferences(n_objectives: int, response: Dict) -> None:
    """
    Validate decision maker's preferences.
    """
    if "preference_method" not in response:
        raise NautilusException("'preference_method entry missing")
    if response["preference_method"] not in [1, 2]:
        raise NautilusException("please specify either preference method 1 (rank) or 2 (percentages).")
    if "preference_info" not in response:
        raise NautilusException("'preference_info entry missing")
    if response["preference_method"] == 1:  # ranks
        if len(response["preference_info"]) < n_objectives:
            msg = "Number of ranks ({}) do not match the number of objectives '({})." \
                .format(len(response["preference_info"]), n_objectives)
            raise NautilusException(msg)
        elif not (1 <= max(response["preference_info"]) <= n_objectives):
            msg = "The minimum index of importance must be greater or equal "
            "to 1 and the maximum index of improtance must be less "
            "than or equal to the number of objectives in the "
            "problem, which is {}. Check the indices {}" \
                .format(n_objectives, response["preference_info"])
            raise NautilusException(msg)
    elif response["preference_method"] == 2:  # percentages
        if len(response["preference_info"]) < n_objectives:
            msg = "Number of given percentages ({}) do not match the number of objectives '({})." \
                .format(len(response["preference_info"]), n_objectives)
            raise NautilusException(msg)
        elif np.sum(response["preference_info"]) != 100:
            msg = (
                "The sum of the percentages must be 100. Current sum" " is {}."
            ).format(np.sum(response["preference_info"]))
            raise NautilusException(msg)


def validate_itn(itn: int) -> None:
    """
    Validate decision maker's new preference for number of iterations left.
    """
    if itn < 0:
        msg = (
            "The given number of iterations left "
            "should be positive. Given iterations '{}'".format(str(itn))
        )
        raise NautilusException(msg)


class NautilusException(Exception):
    """Raised when an exception related to Nautilus is encountered.

    """

    pass


class NautilusInitialRequest(BaseRequest):
    """ A request class to handle the initial preferences.

    """

    def __init__(self, ideal: np.ndarray, nadir: np.ndarray):
        self.n_objectives = len(ideal)
        msg = (
            "Please specify the number of iterations as 'n_iterations' to be carried out.\n"
            "Please specify as 'preference_method' whether to \n"
            "1. Rank the objectives in increasing order according to the importance of improving their value.\n"
            "2. Specify percentages reflecting how much would you like to improve each of the current objective "
            "values."
            "Depending on your selection on 'preference_method', please specify either the ranks or percentages for "
            "each objective as 'preference_info'."
        )
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
        }

        super().__init__("reference_point_preference", "required", content=content)

    @classmethod
    def init_with_method(cls, method):
        return cls(method._ideal, method._nadir)

    @BaseRequest.response.setter
    def response(self, response: Dict):
        validate_response(self.n_objectives, response)
        self._response = response


class NautilusRequest(BaseRequest):
    """A request class to handle the intermediate requests.

    """

    def __init__(
            self,
            ideal: np.ndarray,
            nadir: np.ndarray,
            n_iterations: int,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray,
            distances: np.ndarray,
            minimize: List[int],
    ):
        self.n_objectives = len(ideal)
        msg = (
            "In case you wish to change the number of remaining iterations lower, please specify the number as "
            "'n_iterations'.\n "
            # how dm communicates this in ui? 
            "In case you wish to take a step back to the previous iteration point, please state 'True' here.\n"
            "Please specify as 'preference_method' whether to \n"
            "1. Rank the objectives in increasing order according to the importance of improving their value.\n"
            "2. Specify percentages reflecting how much would you like to improve each of the current objective "
            "values."
            "Depending on your selection on 'preference_method', please specify either the ranks or percentages for "
            "each objective as 'preference_info'.")
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "distances": distances,
            "minimize": minimize,
        }

        super().__init__("reference_point_preference", "required", content=content)

    @BaseRequest.response.setter
    def response(self, response: Dict):
        validate_response(self.n_objectives, response)
        if response["n_iterations"]:
            validate_itn(response["n_iterations"])
        self._response = response


class NautilusStopRequest(BaseRequest):
    """A request class to handle termination.

    """

    def __init__(self, preferred_point: np.ndarray):
        msg = "Most preferred solution found."
        content = {"message": msg, "solution": preferred_point}

        super().__init__("print", "no_interaction", content=content)


class Nautilus(InteractiveMethod):
    """
    Implements the basic NAUTILUS methods as presented in `Miettinen 2010`

        Args:
            ideal (np.ndarray): The ideal objective vector of the problem
            being represented by the Pareto front.
            nadir (np.ndarray): The nadir objective vector of the problem
            being represented by the Pareto front.
            epsilon (float): A small number used in calculating the utopian point.
            objective_names (Optional[List[str]], optional): Names of the
            objectives. List must match the number of columns in
            pareto_front. Defaults to 'f1', 'f2', 'f3', ...
            minimize (Optional[List[int]], optional): Multipliers for each
            objective. '-1' indicates maximization and '1' minimization.
            Defaults to all objective values being minimized.

        Raises:
            NautilusException: One or more dimension mismatches are
            encountered among the supplies arguments.
        """

    def __init__(
            self,
            problem: MOProblem,
            ideal: np.ndarray,
            nadir: np.ndarray,
            epsilon: float = 0.0,
            objective_names: Optional[List[str]] = None,
            minimize: Optional[List[int]] = None,
    ):

        if not ideal.shape == nadir.shape:
            raise NautilusException("The dimensions of the ideal and nadir point do not match.")

        if objective_names:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusException(
                    "The supplied objective names must have a leangth equal to " "the numbr of objectives."
                )
            self._objective_names = objective_names
        else:
            self._objective_names = [f"f{i + 1}" for i in range(ideal.shape[0])]

        if minimize:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusException("The minimize list must have " "as many elements as there are objectives.")
            self._minimize = minimize
        else:
            self._minimize = [1 for _ in range(ideal.shape[0])]

        # initialize problem
        super().__init__(problem)
        self._problem = problem
        self._objectives: np.ndarray = lambda x: self._problem.evaluate(x).objectives  # objective function values
        self._variable_bounds: Union[np.ndarray, None] = problem.get_variable_bounds()
        self._constraint = [c.evaluator for c in self._problem.constraints]

        # Used to calculate the utopian point from the ideal point
        self._epsilon = epsilon
        self._ideal = ideal
        self._nadir = nadir

        # calculate utopian vector
        self._utopian = [ideal_i - self._epsilon for ideal_i in self._ideal]

        # bounds of the reachable region
        self._reachable_ub = self._nadir
        self._reachable_lb = self._ideal
        self._lower_bounds: List[np.ndarray] = []
        self._upper_bounds: List[np.ndarray] = []

        # current iteration step number
        self._step_number = 1

        # iteration points
        self._zs: List[np.ndarray] = []

        # solutions, objectives, and distances for each iteration
        self._xs: List[np.ndarray] = []
        self._fs: List[np.ndarray] = []
        self._ds: List[np.ndarray] = []

        # The current reference point
        self._q: np.ndarray = None

        self._distance = None

        # preference information
        self._preference_method = None
        self._preference_info = None
        self._preference_factors = None

        # number of total iterations and iterations left
        self._n_iterations = None
        self._n_iterations_left = None

        # flags for the iteration phase
        self._use_previous_preference: bool = False
        self._step_back: bool = False
        self._short_step: bool = False
        self._first_iteration: bool = True

    def start(self) -> NautilusInitialRequest:
        return NautilusInitialRequest.init_with_method(self)

    def iterate(
            self, request: Union[NautilusInitialRequest, NautilusRequest]
    ) -> Union[NautilusRequest, NautilusStopRequest]:
        """Perform the next logical iteration step based on the given request type.

        """
        if type(request) is NautilusInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is NautilusRequest:
            return self.handle_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(self, request: NautilusInitialRequest) -> NautilusRequest:
        """Handles the initial request by parsing the response appropiately.

        """

        # set iteration number info and first iteration point (nadir point)
        self._n_iterations: int = request.response["n_iterations"]
        self._n_iterations_left: int = self._n_iterations

        # set up arrays for storing information from obtained solutions, function values, distances, and bounds
        self._xs = [None] * (self._n_iterations + 1)
        self._fs = [None] * (self._n_iterations + 1)
        self._ds = [None] * (self._n_iterations + 1)
        self._zs = [None] * (self._n_iterations + 1)
        self._lower_bounds = [None] * (self._n_iterations + 1)
        self._upper_bounds = [None] * (self._n_iterations + 1)

        # set initial iteration point
        self._zs[self._step_number - 1] = self._nadir

        # set preference information
        self._preference_method: int = request.response["preference_method"]
        self._preference_info: np.ndarray = request.response["preference_info"]
        self._preference_factors = self.calculate_preference_factors(self._preference_method, self._preference_info,
                                                                     self._nadir, self._utopian)

        # set reference point, initial values for decision variables and solve the problem
        self._q = self._zs[self._step_number - 1]
        x0 = self._problem.get_variable_upper_bounds() / 2
        print(x0)
        result = self.solve_asf(self._q, x0, self._preference_factors, self._nadir, self._utopian, self._objectives,
                                self._variable_bounds, method=None)  # include preference info on method?

        # update current solution and objective function values
        self._xs[self._step_number - 1] = result["x"]
        self._fs[self._step_number - 1] = result["fun"]

        # calculate next iteration point
        self._zs[self._step_number] = self.calculate_iteration_point(self._n_iterations_left,
                                                                     self._zs[self._step_number - 1],
                                                                     self._fs[self._step_number - 1])
        # calculate new bounds and store the information
        new_lower_bounds = self.calculate_bounds(self._objectives, len(self._objective_names), x0,
                                                 self._zs[self._step_number - 1], self._variable_bounds,
                                                 self._constraint[0], None)

        self._lower_bounds[self._step_number] = new_lower_bounds
        self._upper_bounds[self._step_number] = self._zs[self._step_number]

        # calculate distance from current iteration point to Pareto optimal set
        d = self.calculate_distance(self._zs[self._step_number], self._nadir, self._fs[self._step_number - 1])
        self._ds[self._step_number - 1] = d

        return NautilusRequest(
            self._ideal, self._nadir, self._n_iterations, self._ideal, self._nadir, [], self._minimize
        )

    def calculate_preference_factors(self, pref_method: int, pref_info: np.ndarray, nadir: np.ndarray,
                                     utopian: np.ndarray) -> np.ndarray:
        """
        Calculate preference factors based on decision maker's preference information.
        """
        if pref_method == 1:  # ranks
            return [1 / (r_i * (n_i - u_i)) for r_i, n_i, u_i in zip(pref_info, nadir, utopian)]
        elif pref_method == 2:  # percentages
            delta_q = pref_info / 100
            return [1 / (d_i * (n_i - u_i)) for d_i, n_i, u_i in zip(delta_q, nadir, utopian)]

    def solve_asf(self,
                  ref_point: np.ndarray,
                  x0: np.ndarray,
                  preference_factors: np.ndarray,
                  nadir: np.ndarray,
                  utopian: np.ndarray,
                  objectives: np.ndarray,
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

        Returns: Dict: A dictionary with at least the following entries: 'x' indicating the optimal
                 variables found, 'fun' the optimal value of the optimized functoin, and 'success' a boolean
                 indicating whether the optimizaton was conducted successfully.

        """

        # scalarize problem using reference point
        asf = ReferencePointASF(preference_factors, nadir, utopian)
        asf_scalarizer = Scalarizer(
            objectives,
            asf,
            scalarizer_args={"reference_point": ref_point})

        # minimize
        minimizer = ScalarMinimizer(asf_scalarizer, variable_bounds, method=method)
        return minimizer.minimize(x0)

    def calculate_iteration_point(self, itn: int, z_prev: np.ndarray, f_current: np.ndarray) -> np.ndarray:
        """

        Args:
            itn (int): Number of iterations left.
            z_prev(np.ndarray): Previous iteration point.
            f_current (np.ndarray): Current optimal objective vector.

        Returns:
            z_next (np.ndarray): Next iteration point.

        """

        return ((itn - 1) / itn) * z_prev + ((1 / itn) * f_current)

    def calculate_bounds(self, objectives: Callable, n_objectives: int, x0: np.ndarray, epsilons: np.ndarray,
                         bounds: Union[np.ndarray, None], constraints: Callable,
                         method: Union[ScalarMethod, str, None]):
        """
        Calculate the new bounds using Epsilon constraint method.
        Args:
            objectives (np.ndarray): The objective function values for each input vector.
            n_objectives (int): Total number of objectives.
            x0 (np.ndarray): Initial values for decison variables.
            epsilons (np.ndarray): Previous iteration point.
            bounds (Union[np.ndarray, None): Bounds for decision variables.
            constraints (Callable): Constraints of the problem.
            method (Union[ScalarMethod, str, None]): The optimization method the scalarizer should be minimized with.

        Returns:

        """
        new_lower_bounds: np.ndarray = [None] * n_objectives

        # solve new lower bounds for each objective
        for i in range(n_objectives):
            eps = EpsilonConstraintMethod(objectives,
                                          i,
                                          # take out the objective to be minimized
                                          [val for ind, val in enumerate(epsilons) if ind != i],
                                          constraints=constraints)
            cons_evaluate = eps.evaluate_constraints
            scalarized_objective = Scalarizer(objectives, eps)
            minimizer = ScalarMinimizer(scalarized_objective, bounds, constraint_evaluator=cons_evaluate, method=None)
            res = minimizer.minimize(x0)

            # store objective function values as new lower bounds
            new_lower_bounds[i] = objectives(res["x"])[0][i]

        return new_lower_bounds

    def calculate_distance(self, z_current: np.ndarray, nadir: np.ndarray, f_current: np.ndarray) -> np.ndarray:
        """
        Calculates the distance from current iteration point to the Pareto optimal set.
        Args:
            z_current (np.ndarray): Current iteration point.
            nadir (np.ndarray): Nadir vector.
            f_current (np.ndarray): Current optimal objective vector.

        Returns:
            (np.ndarray): Distance to the Pareto optimal set.

        """
        dist = (np.linalg.norm(np.atleast_2d(z_current) - nadir, ord=2, axis=1)) \
               / (np.linalg.norm(np.atleast_2d(f_current) - nadir, ord=2, axis=1))
        return dist * 100


# testing the method
if __name__ == "__main__":
    # variables
    var_names = ["r", "h"]  # Make sure that the variable names are meaningful to you.

    initial_values = [2.5, 11]
    lower_bounds = [2.5, 10]
    upper_bounds = [15, 50]
    bounds = np.stack((lower_bounds, upper_bounds))

    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)


    # objectives

    def volume(r, h):
        return np.pi * r ** 2 * h


    def area(r, h):
        return 2 * np.pi ** 2 + np.pi * r * h


    def objective(xs):
        # xs is a 2d array like, which has different values for r and h on its first and second columns respectively.
        xs = np.atleast_2d(xs)
        return np.stack((volume(xs[:, 0], xs[:, 1]), -area(xs[:, 0], xs[:, 1]))).T


    f1 = _ScalarObjective("y1", volume, maximize=True)
    f2 = _ScalarObjective("y2", area)
    f1_2 = VectorObjective("y3", objective)


    # constraints
    def con_golden(xs, _):
        # constraints are defined in DESDEO in a way were a positive value indicates an agreement with a constraint, and
        # a negative one a disagreement.
        xs = np.atleast_2d(xs)
        return -(xs[:, 0] / xs[:, 1] - 1.618)


    cons1 = ScalarConstraint(name="c_1", n_objective_funs=2, n_decision_vars=2, evaluator=con_golden)

    # problem
    prob = MOProblem(objectives=[f1_2], variables=variables, constraints=[cons1])

    # ideal and nadir

    """
    def simple_sum(xs):
        xs = np.atleast_2d(xs)
        return np.sum(xs, axis=1)

    # define a new scalarizing function so that each of the objectives can be optimized independently
    def weighted_sum(xs, ws):
        # ws stand for weights
        return np.sum(ws * xs, axis=1)


    scalarized_objective = Scalarizer(objective, simple_sum)
    minimizer = ScalarMinimizer(scalarized_objective, bounds, constraint_evaluator=con_golden, method=None)

    # minimize the first objective
    x0 = np.array([2.6, 11])
    weighted_scalarized_objective = Scalarizer(objective, weighted_sum, scalarizer_args={"ws": np.array([1, 0])})
    minimizer._scalarizer = weighted_scalarized_objective
    res = minimizer.minimize(x0)
    first_obj_vals = objective(res["x"])

    # minimize the second objective
    weighted_scalarized_objective._scalarizer_args = {"ws": np.array([0, 1])}
    res = minimizer.minimize(x0)
    second_obj_vals = objective(res["x"])

    # payoff table
    po_table = np.stack((first_obj_vals, second_obj_vals)).squeeze()

    ideal = np.diagonal(po_table)
    nadir = np.max(po_table, axis=0)
    """

    ideal = np.array([196.34971768, -2375.93349431])
    nadir = np.array([35342.91192077, -98.27906444])

    method = Nautilus(prob, ideal, nadir)

    req = method.start()

    n_iterations = 11

    req.response = {
        "n_iterations": n_iterations,
        "preference_method": 1,
        "preference_info": [1, 2],
    }

    req = method.iterate(req)

    """
    req.response = {"preferred_point_index": 0}

    while method._n_iterations_left > 1:
        print(method._n_iterations_left)
        req = method.iterate(req)
        print(req.content["points"])
        req.response = {"preferred_point_index": 0}

    print(method._n_iterations_left)
    req = method.iterate(req)
    print(method._n_iterations_left)
    print(method._distance)
    print(req.content["solution"])
    
    """
