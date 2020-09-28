from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from desdeo_problem.Variable import variable_builder
from desdeo_problem.Objective import VectorObjective, _ScalarObjective
from desdeo_problem.Constraint import ScalarConstraint
from desdeo_problem.Problem import MOProblem
from desdeo_tools.interaction.request import BaseRequest
from desdeo_tools.scalarization import ReferencePointASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod


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

        super().__init__(problem)
        self._problem = problem

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

        # Used to calculate the utopian point from the ideal point
        self._epsilon = epsilon

        self._ideal = ideal
        self._nadir = nadir

        # calculate utopian vector
        self._utopian = [ideal_i - self._epsilon for ideal_i in self._ideal]

        # bounds of the reachable region
        self._reachable_ub = self._nadir
        self._reachable_lb = self._ideal

        # current iteration step number
        self._step_number = 1

        # iteration points
        self._zs: List[np.ndarray] = []

        # The current reference point
        self._q: np.ndarray = None

        self._distance = None

        # preference information
        self._preference_method = None
        self._preference_info = None
        self._preference_factors = None

        self._n_iterations = None
        self._n_iterations_left = None

        #self._scalar_solver = ScalarSolver()

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

        # set interation number info and first iteration point (nadir point)
        self._n_iterations: int = request.response["n_iterations"]
        self._n_iterations_left: int = self._n_iterations
        self._zs.append(self._nadir)

        # set preference information
        self._preference_method: int = request.response["preference_method"]
        self._preference_info: np.ndarray = request.response["preference_info"]
        self._preference_factors = self.calculate_preference_factors(self._preference_method, self._preference_info,
                                                                     self._nadir, self._utopian)

        # set reference point
        self._q = self._zs[self._step_number-1]

        # solve problem using achievement scalarizing function method
        asf = ReferencePointASF(self._preference_factors, self._nadir, self._utopian)
        asf_scalarizer = Scalarizer(self._problem.objectives, asf, scalarizer_args={"reference_point": self._q})
        # TODO: continue on solving the asf problem, how to include bounds?

        return NautilusRequest(
            self._ideal, self._nadir, self._n_iterations, self._ideal, self._nadir, [], self._minimize
        )

    def handle_request(self, request: NautilusRequest) -> Union[NautilusRequest, NautilusStopRequest]:
        """Handles the intermediate requests.

        """

        if self._n_iterations_left <= 1:
            self._n_iterations_left = 0
            return NautilusStopRequest(self._preferred_point)

        self._reachable_lb = request.content["lower_bounds"][preferred_point_index]
        self._reachable_ub = request.content["upper_bounds"][preferred_point_index]
        self._distance = request.content["distances"][preferred_point_index]

        self._reachable_idx = self.calculate_reachable_point_indices(
            self._pareto_front, self._reachable_lb, self._reachable_ub
        )

        # increment and decrement iterations
        self._n_iterations_left -= 1
        self._step_number += 1

        # Start again
        zbars = self.calculate_representative_points(self._pareto_front, self._reachable_idx, self._n_points)
        zs = self.calculate_intermediate_points(self._preferred_point, zbars, self._n_iterations_left)
        new_lower_bounds, new_upper_bounds = self.calculate_bounds(self._pareto_front, zs)
        distances = self.calculate_distances(zs, zbars, self._nadir)

        return NautilusRequest(
            self._ideal, self._nadir, zs, new_lower_bounds, new_upper_bounds, distances, self._minimize
        )

    def calculate_preference_factors(self, pref_method: int, pref_info: np.ndarray, nadir: np.ndarray, utopian: np.ndarray) -> np.ndarray:
        """
        Calculate preference factors based on decision maker's preference information.
        """
        if pref_method == 1:  # ranks
            return [1/(r_i*(n_i-u_i)) for r_i, n_i, u_i in zip(pref_info, nadir, utopian)]
        elif pref_method == 2:  # percentages
            delta_q = pref_info/100
            return [1/(d_i*(n_i-u_i)) for d_i, n_i, u_i in zip(delta_q, nadir, utopian)]


# testing the method
if __name__ == "__main__":

    f1 = np.linspace(1, 100, 50)
    f2 = f1[::-1] ** 2

    front = np.stack((f1, f2)).T
    #ideal = np.min(front, axis=0)
    #nadir = np.max(front, axis=0)


    # variables
    var_names = ["a", "b", "c"]  # Make sure that the variable names are meaningful to you.

    initial_values = [1, 1, 1]
    lower_bounds = [-2, -1, 0]
    upper_bounds = [5, 10, 3]

    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

    # objectives
    def obj1_2(x):  # This is a "simulator" that returns more than one objective at a time. Hence, use VectorObjective
        y1 = x[:, 0] + x[:, 1] + x[:, 2]
        y2 = x[:, 0] * x[:, 1] * x[:, 2]
        return (y1, y2)


    def obj3(x):  # This is a "simulator" that returns only one objective at a time. Hence, use ScalarObjective
        y3 = x[:, 0] * x[:, 1] + x[:, 2]
        return y3


    f1_2 = VectorObjective(["y1", "y2"], obj1_2)
    f3 = _ScalarObjective("y3", obj3, maximize=True)  # Note: f3 = VectorObjective(["y3"], obj3) will also work.

    # constraints

    const_func = lambda x, y: 10 - (x[:, 0] + x[:, 1] + x[:, 2])

    # Args: name, number of variables, number of objectives, callable

    cons1 = ScalarConstraint("c_1", 3, 3, const_func)

    # problem
    prob = MOProblem(objectives=[f1_2, f3], variables=variables, constraints=[cons1])

    ideal = np.array([0, 0, 0])
    nadir = np.array([6,6,6])

    method = Nautilus(prob, ideal, nadir)

    req = method.start()

    n_iterations = 11

    req.response = {
        "n_iterations": n_iterations,
        "preference_method": 1,
        "preference_info": [1, 2, 3],
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
