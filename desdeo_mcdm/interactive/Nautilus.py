"""
NAUTILUS 1
"""
from typing import Dict, List, Optional, Union, Callable

import numpy as np

from desdeo_problem.Variable import variable_builder
from desdeo_problem.Objective import VectorObjective, _ScalarObjective
from desdeo_problem.Problem import MOProblem
from desdeo_tools.interaction.request import BaseRequest
from desdeo_tools.scalarization import ReferencePointASF
from desdeo_tools.scalarization import EpsilonConstraintMethod as ECM
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer, ScalarMethod

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod

from scipy.optimize import differential_evolution


def validate_response(n_objectives: int,
                      z_current: np.ndarray,
                      nadir: np.ndarray,
                      response: Dict,
                      first_iteration_bool: bool) -> None:
    """
    Validate decision maker's response.

    Args:
        n_objectives (int): Number of objectives.
        z_current (np.ndarray): Current iteration point.
        nadir (np.ndarray): Nadir point.
        response (Dict) : Decision maker's response containing preference information.
        first_iteration_bool (bool) : Indicating whether the iteration round is the first one (True) or not (False).

    Raises:
        NautilusException: In case Decision maker's response is not valid.
    """

    if first_iteration_bool:
        if "n_iterations" not in response:
            raise NautilusException("'n_iterations' entry missing")
        if "step_back" in response:
            raise NautilusException("Cannot take a step back on first iteration.")
        if "use_previous_preference" in response:
            raise NautilusException("Cannot use previous preferences on first iteration.")
        validate_preferences(n_objectives, response)
    else:
        # if current iteration point is nadir point
        if response["step_back"] and np.array_equal(z_current, nadir):
            raise NautilusException("Cannot take more steps back, current iteration point is the nadir point.")
        # if dm wants to provide new preference info
        if not response["use_previous_preference"]:
            validate_preferences(n_objectives, response)
    if "n_iterations" in response:  # both for providing initial and new numbers of iterations.
        validate_n_iterations(response["n_iterations"])


def validate_preferences(n_objectives: int, response: Dict) -> None:
    """
    Validate decision maker's preferences.

    Args:
        n_objectives (int): Number of objectives in problem.
        response (Dict): Decision maker's response containing preference information.

    Raises:
        NautilusException: In case preference info is not valid.

    """

    if "preference_method" not in response:
        raise NautilusException("'preference_method entry missing")
    if "preference_info" not in response:
        raise NautilusException("'preference_info entry missing")
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


def validate_n_iterations(n_it: int) -> None:
    """
    Validate decision maker's preference for number of iterations.

    Args:
        n_it (int): Number of iterations.

    Raises:
        NautilusException: If number of iterations given is not an positive integer greater than zero.

    """

    if not isinstance(n_it, int) or int(n_it) < 1:
        msg = (
            "The given number of iterations left "
            "should be a positive integer greater than zero. Given iterations '{}'".format(str(n_it))
        )
        raise NautilusException(msg)


class NautilusException(Exception):
    """
    Raised when an exception related to Nautilus is encountered.
    """

    pass


class NautilusInitialRequest(BaseRequest):
    """
    A request class to handle the Decision maker's initial preferences for the first iteration round.
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
    def init_with_method(cls, method: InteractiveMethod):
        """
        Initialize request with given instance of Nautilus method.

        Args:
            method (Nautilus): Instance of Nautilus-class.

        Returns:
            NautilusInitialRequest: Initial request.

        """

        return cls(method._ideal, method._nadir)

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set Decision maker's response information for initial request.

        Args:
            response (Dict): Decision maker's response.

        """

        validate_response(self.n_objectives, z_current=self._nadir, nadir=self._nadir, response=response,
                          first_iteration_bool=True)
        self._response = response


class NautilusRequest(BaseRequest):
    """
    A request class to handle the Decision maker's preferences after the first iteration round.
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
            "1. Rank the objectives in increasing order according to the importance of improving their value.\n"
            "2. Specify percentages reflecting how much would you like to improve each of the current objective "
            "values."
            "Depending on your selection on 'preference_method', please specify either the ranks or percentages for "
            "each objective as 'preference_info'."
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

        validate_response(self._n_objectives, self._z_current, self._nadir, response, first_iteration_bool=False)
        self._response = response


class NautilusStopRequest(BaseRequest):
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
        content = {"message": msg, "solution": x_h, "objective vector": f_h}

        super().__init__("print", "no_interaction", content=content)


class Nautilus(InteractiveMethod):
    """
    Implements the basic NAUTILUS method as presented in |Miettinen_2010|.

    In NAUTILUS, starting from the nadir point,
    a solution is obtained at each iteration which dominates the previous one.
    Although only the last solution will be Pareto optimal, the decision maker never looses sight of the
    Pareto optimal set, and the search is oriented so that (s)he progressively focusses on the preferred part of
    the Pareto optimal set. Each new solution is obtained by minimizing an achievement scalarizing function including
    preferences about desired improvements in objective function values.

    The decision maker has **two possibilities** to provide her/his preferences:

    1. The decision maker can **rank** the objectives according to the **relative** importance of improving each current
    objective value.

    Note:
        This ranking is not a global preference ranking of the objectives, but represents the local importance of
        improving each of the current objective values **at that moment**.

    2. The decision maker can specify **percentages** reflecting how (s)he would like to improve the current objective
    values, by answering to the following question:

    *"Assuming you have one hundred points available, how would you distribute
    them among the current objective values so that the more points you allocate, the more improvement on the
    corresponding current objective value is desired?"*

    After each iteration round, the decision maker specifies whether (s)he wishes to continue with the previous
    preference information, or define a new one.

    In addition to this, the decision maker can influence the solution finding process by taking a **step back** to
    previous iteration point. This enables the decision maker to provide new preferences and change the direction of
    solution seeking process. Furthermore, the decision maker can also take a **half-step** in case (s)he feels that a
    full step limits the reachable area of Pareto optimal set too much.

    NAUTILUS is specially suitable for avoiding  undesired anchoring effects, for example in negotiation support
    problems, or just as a means of finding an initial Pareto optimal solution for any interactive procedure.

    Args:
        problem (MOProblem): Problem to be solved.
        ideal (np.ndarray): The ideal objective vector of the problem.
        nadir (np.ndarray): The nadir objective vector of the problem. This may also be the "worst" objective vector
                            provided by the Decision maker if the approximation of Nadir vector is not applicable or if
                            the Decision maker wishes to provide even worse objective vector than what the
                            approximated Nadir vector is.
        epsilon (float): A small number used in calculating the utopian point.
        objective_names (Optional[List[str]], optional): Names of the objectives. List must match the number of columns
                                                         in ideal.
        minimize (Optional[List[int]], optional): Multipliers for each objective. '-1' indicates maximization
                                                  and '1' minimization. Defaults to all objective values being
                                                  minimized.

    Raises:
        NautilusException: One or more dimension mismatches are encountered among the supplies arguments.
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
            raise NautilusException("The dimensions of the ideal and nadir point do not match.")

        if objective_names:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusException(
                    "The supplied objective names must have a length equal to " "the numbr of objectives."
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

        # initialize method with problem
        super().__init__(problem)
        self._problem = problem
        self._objectives: Callable = lambda x: self._problem.evaluate(x).objectives
        self._variable_bounds: Union[np.ndarray, None] = problem.get_variable_bounds()
        self._constraints: Optional[Callable] = lambda x: self._problem.evaluate(x).constraints

        # Used to calculate the utopian point from the ideal point
        self._epsilon = epsilon
        self._ideal = ideal
        self._nadir = nadir

        # calculate utopian vector
        self._utopian = [ideal_i - self._epsilon for ideal_i in self._ideal]

        # bounds of the reachable region
        self._lower_bounds: List[np.ndarray] = []
        self._upper_bounds: List[np.ndarray] = []

        # current iteration step number
        self._step_number = 1

        # iteration points
        self._zs: np.ndarray = []

        # solutions, objectives, and distances for each iteration
        self._xs: np.ndarray = []
        self._fs: np.ndarray = []
        self._ds: np.ndarray = []

        # The current reference point
        self._q: Union[None, np.ndarray] = None

        # preference information
        self._preference_method = None
        self._preference_info = None
        self._preferential_factors = None

        # number of total iterations and iterations left
        self._n_iterations = None
        self._n_iterations_left = None

        # flags for the iteration phase
        # not utilized atm
        self._use_previous_preference: bool = False
        self._step_back: bool = False
        self._short_step: bool = False
        self._first_iteration: bool = True

        # evolutionary method for minimizing
        self._method_de: ScalarMethod = ScalarMethod(
            lambda x, _, **y: differential_evolution(x, **y),
            method_args={"disp": False, "polish": False, "tol": 0.000001, "popsize": 10, "maxiter": 50000},
            use_scipy=True
        )

    def start(self) -> NautilusInitialRequest:
        """
        Start the solution process with initializing the first request.

        Returns:
            NautilusInitialRequest: Initial request.

        """

        return NautilusInitialRequest.init_with_method(self)

    def iterate(
            self, request: Union[NautilusInitialRequest, NautilusRequest, NautilusStopRequest]
    ) -> Union[NautilusRequest, NautilusStopRequest]:
        """
        Perform the next logical iteration step based on the given request type.

        Args:
            request (Union[NautilusInitialRequest, NautilusRequest]): Either initial or intermediate request.

        Returns:
            Union[NautilusRequest, NautilusStopRequest]: A new request with content depending on the Decision maker's
            preferences.

        """

        if type(request) is NautilusInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is NautilusRequest:
            return self.handle_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(self, request: NautilusInitialRequest) -> NautilusRequest:
        """
        Handles the initial request by parsing the response appropriately.

        Args:
            request (NautilusInitialRequest): Initial request including Decision maker's initial preferences.

        Returns:
            NautilusRequest: New request with updated solution process information.

        """

        # set iteration number info and first iteration point (nadir point)
        self._n_iterations: int = request.response["n_iterations"]
        self._n_iterations_left: int = self._n_iterations

        # set up arrays for storing information from obtained solutions, function values, distances, and bounds
        self._xs = [None] * (self._n_iterations + 2)
        self._fs = [None] * (self._n_iterations + 2)
        self._ds = [None] * (self._n_iterations + 2)
        self._zs = [None] * (self._n_iterations + 2)
        self._lower_bounds = [None] * (self._n_iterations + 2)
        self._upper_bounds = [None] * (self._n_iterations + 2)

        # set initial iteration point
        self._zs[self._step_number - 1] = self._nadir

        # set preference information
        self._preference_method: int = request.response["preference_method"]
        self._preference_info: np.ndarray = request.response["preference_info"]
        self._preferential_factors = self.calculate_preferential_factors(self._preference_method, self._preference_info,
                                                                         self._nadir, self._utopian)

        # set reference point, initial values for decision variables, lower and upper bounds for objective functions
        self._q = self._zs[self._step_number - 1]
        x0 = self._problem.get_variable_upper_bounds() / 2

        self._lower_bounds[self._step_number] = self._ideal
        self._upper_bounds[self._step_number] = self._nadir

        # solve the ASF-problem
        result = self.solve_asf(self._q, x0, self._preferential_factors, self._nadir, self._utopian, self._objectives,
                                self._variable_bounds, method=self._method_de)

        # update current solution and objective function values
        self._xs[self._step_number] = result["x"]
        self._fs[self._step_number] = self._objectives(self._xs[self._step_number])[0]

        # calculate next iteration point
        self._zs[self._step_number] = self.calculate_iteration_point(self._n_iterations_left,
                                                                     self._zs[self._step_number - 1],
                                                                     self._fs[self._step_number])
        # calculate new bounds and store the information
        new_lower_bounds = self.calculate_bounds(self._objectives, len(self._objective_names), x0,
                                                 self._zs[self._step_number], self._variable_bounds,
                                                 self._constraints, None)

        self._lower_bounds[self._step_number + 1] = new_lower_bounds
        self._upper_bounds[self._step_number + 1] = self._zs[self._step_number]

        # calculate distance from current iteration point to Pareto optimal set
        self._ds[self._step_number] = self.calculate_distance(self._zs[self._step_number],
                                                              self._nadir,
                                                              self._fs[self._step_number])

        # return the information from iteration round to be shown to the DM.
        return NautilusRequest(
            self._zs[self._step_number], self._nadir, self._lower_bounds[self._step_number + 1],
            self._upper_bounds[self._step_number + 1], self._ds[self._step_number]
        )

    def handle_request(self, request: NautilusRequest) -> Union[NautilusRequest, NautilusStopRequest]:
        """
        Handle Decision maker's requests after the first iteration round, so called **intermediate requests.**

        Args:
            request (NautilusRequest): Intermediate request including Decision maker's response.

        Returns:
            Union[NautilusRequest, NautilusStopRequest]: In case last iteration, request to stop the solution process.
            Otherwise, new request with updated solution process information.

        """

        resp: dict = request.response

        # change the number of iterations
        if "n_iterations" in resp:

            # "expand" the numpy arrays used for storing information from iteration rounds
            if resp["n_iterations"] > self._n_iterations:
                extra_space = [None] * (resp["n_iterations"] - self._n_iterations)
                self._zs = np.array(np.concatenate((self._zs, extra_space), axis=None), dtype=object)
                self._xs = np.array(np.concatenate((self._xs, extra_space), axis=None), dtype=object)
                self._fs = np.array(np.concatenate((self._fs, extra_space), axis=None), dtype=object)
                self._ds = np.array(np.concatenate((self._ds, extra_space), axis=None), dtype=object)
                self._lower_bounds = np.array(np.concatenate((self._lower_bounds, extra_space), axis=None),
                                              dtype=object)
                self._upper_bounds = np.array(np.concatenate((self._upper_bounds, extra_space), axis=None),
                                              dtype=object)

            self._n_iterations_left = resp["n_iterations"]

        # last iteration, stop solution process
        if self._n_iterations_left <= 1:
            self._n_iterations_left = 0
            return NautilusStopRequest(self._xs[self._step_number], self._fs[self._step_number])

        # don't step back...
        if not resp["step_back"]:
            self._step_back = False
            self._n_iterations_left -= 1
            self._step_number += 1

            # ... and continue with same preferences
            if resp["use_previous_preference"]:
                # use the solution and objective of last step
                self._xs[self._step_number] = self._xs[self._step_number - 1]
                self._fs[self._step_number] = self._fs[self._step_number - 1]

            # ... and give new preferences
            else:
                # step 1
                # set preference information
                self._preference_method: int = resp["preference_method"]
                self._preference_info: np.ndarray = resp["preference_info"]
                self._preferential_factors = self.calculate_preferential_factors(self._preference_method,
                                                                                 self._preference_info,
                                                                                 self._nadir, self._utopian)

                # set reference point, initial values for decision variables and solve the problem
                self._q = self._zs[self._step_number - 1]
                x0 = self._problem.get_variable_upper_bounds() / 2
                result = self.solve_asf(self._q, x0, self._preferential_factors, self._nadir, self._utopian,
                                        self._objectives,
                                        self._variable_bounds, method=self._method_de)

                # update current solution and objective function values
                self._xs[self._step_number] = result["x"]
                self._fs[self._step_number] = self._objectives(self._xs[self._step_number])[0]

            # continue from step 3
            # calculate next iteration point
            self._zs[self._step_number] = self.calculate_iteration_point(self._n_iterations_left,
                                                                         self._zs[self._step_number - 1],
                                                                         self._fs[self._step_number])

            # calculate new bounds and store the information
            new_lower_bounds = self.calculate_bounds(self._objectives, len(self._objective_names),
                                                     self._problem.get_variable_upper_bounds() / 2,
                                                     self._zs[self._step_number], self._variable_bounds,
                                                     self._constraints, None)

            self._lower_bounds[self._step_number + 1] = new_lower_bounds
            self._upper_bounds[self._step_number + 1] = self._zs[self._step_number]

            # calculate distance from current iteration point to Pareto optimal set
            self._ds[self._step_number] = self.calculate_distance(self._zs[self._step_number],
                                                                  self._nadir,
                                                                  self._fs[self._step_number])

            # return the information from iteration round to be shown to the DM.
            return NautilusRequest(
                self._zs[self._step_number], self._nadir, self._lower_bounds[self._step_number + 1],
                self._upper_bounds[self._step_number + 1], self._ds[self._step_number]
            )

        # take a step back...
        if resp["step_back"]:
            self._step_back = True

            # ... and take a short step
            if resp["short_step"]:
                self._short_step = True
                self._zs[self._step_number] = 0.5 * self._zs[self._step_number] + 0.5 * self._zs[self._step_number - 1]

                # calculate new bounds and store the information
                new_lower_bounds = self.calculate_bounds(self._objectives, len(self._objective_names),
                                                         self._problem.get_variable_upper_bounds() / 2,
                                                         self._zs[self._step_number], self._variable_bounds,
                                                         self._constraints, None)

                self._lower_bounds[self._step_number + 1] = new_lower_bounds
                self._upper_bounds[self._step_number + 1] = self._zs[self._step_number]

                # calculate distance from current iteration point to Pareto optimal set
                self._ds[self._step_number] = self.calculate_distance(self._zs[self._step_number],
                                                                      self._nadir,
                                                                      self._fs[self._step_number])

                # return the information from iteration round to be shown to the DM.
                return NautilusRequest(
                    self._zs[self._step_number], self._nadir, self._lower_bounds[self._step_number + 1],
                    self._upper_bounds[self._step_number + 1], self._ds[self._step_number]
                )

            # ... and use new preferences
            elif not resp["use_previous_preference"]:

                # set preference information
                self._preference_method: int = resp["preference_method"]
                self._preference_info: np.ndarray = resp["preference_info"]
                self._preferential_factors = self.calculate_preferential_factors(self._preference_method,
                                                                                 self._preference_info,
                                                                                 self._nadir, self._utopian)

                # set reference point, initial values for decision variables and solve the problem
                self._q = self._zs[self._step_number - 1]
                x0 = self._problem.get_variable_upper_bounds() / 2
                result = self.solve_asf(self._q, x0, self._preferential_factors, self._nadir, self._utopian,
                                        self._objectives,
                                        self._variable_bounds, method=self._method_de)

                # update current solution and objective function values
                self._xs[self._step_number] = result["x"]
                self._fs[self._step_number] = self._objectives(self._xs[self._step_number])[0]

                # calculate next iteration point
                self._zs[self._step_number] = self.calculate_iteration_point(self._n_iterations_left,
                                                                             self._zs[self._step_number - 1],
                                                                             self._fs[self._step_number])

                # calculate new bounds and store the information
                new_lower_bounds = self.calculate_bounds(self._objectives, len(self._objective_names), x0,
                                                         self._zs[self._step_number], self._variable_bounds,
                                                         self._constraints, None)

                self._lower_bounds[self._step_number + 1] = new_lower_bounds
                self._upper_bounds[self._step_number + 1] = self._zs[self._step_number]

                # calculate distance from current iteration point to Pareto optimal set
                self._ds[self._step_number] = self.calculate_distance(self._zs[self._step_number],
                                                                      self._nadir,
                                                                      self._fs[self._step_number])

                # return the information from iteration round to be shown to the DM.
                return NautilusRequest(
                    self._zs[self._step_number], self._nadir, self._lower_bounds[self._step_number + 1],
                    self._upper_bounds[self._step_number + 1], self._ds[self._step_number]
                )

    def calculate_preferential_factors(self, pref_method: int, pref_info: np.ndarray, nadir: np.ndarray,
                                       utopian: np.ndarray) -> np.ndarray:
        """
        Calculate preferential factors based on the Decision maker's preference information. These preferential
        factors are used as weights for objectives when solving an Achievement scalarizing function. The Decision maker
        (DM) has **two possibilities** to provide her/his preferences:

        1. The DM can rank the objectives according to the **relative** importance of improving each current objective
        value.

        Note:
            This ranking is not a global preference ranking of the objectives, but represents the local importance of
            improving each of the current objective values **at that moment**.

        2. The DM can specify percentages reflecting how (s)he would like to improve the current objective values,
        by answering to the following question:

        *"Assuming you have one hundred points available, how would you distribute
        them among the current objective values so that the more points you allocate, the more improvement on the
        corresponding current objective value is desired?"*

        Args:
            pref_method (int): Preference information method (either ranks (1) or percentages (2)).
            pref_info (np.ndarray): Preference information on how the DM wishes to improve the values of each objective
                                    function.
            nadir (np.ndarray): Nadir vector.
            utopian (np.ndarray): Utopian vector.

        Returns:
            np.ndarray: Weights assigned to each of the objective functions in achievement scalarizing function.

        Examples:
            >>> pref_method = 1  # ranks
            >>> pref_info = np.array([2, 2, 1, 1])  # first and second objective are the most important to improve
            >>> nadir = np.array([-4.75, -2.87, -0.32, 9.71])
            >>> utopian = np.array([-6.34, -3.44, -7.5, 0.])
            >>> calculate_preferential_factors(pref_method, pref_info, nadir, utopian)
            array([0.31446541, 0.87719298, 0.13927577, 0.10298661])

            >>> pref_method = 2  # percentages
            >>> pref_info = np.array([10, 30, 40, 20])  # DM wishes to improve most the value of objective 3, then 2,4,1
            >>> nadir = np.array([-4.75, -2.87, -0.32, 9.71])
            >>> utopian = np.array([-6.34, -3.44, -7.5, 0.])
            >>> calculate_preferential_factors(pref_method, pref_info, nadir, utopian)
            array([6.28930818, 5.84795322, 0.34818942, 0.51493306])

        """

        if pref_method == 1:  # ranks
            return np.array([1 / (r_i * (n_i - u_i)) for r_i, n_i, u_i in zip(pref_info, nadir, utopian)])
        elif pref_method == 2:  # percentages
            delta_q = pref_info / 100
            return np.array([1 / (d_i * (n_i - u_i)) for d_i, n_i, u_i in zip(delta_q, nadir, utopian)])

    def solve_asf(self,
                  ref_point: np.ndarray,
                  x0: np.ndarray,
                  preferential_factors: np.ndarray,
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
            x0 (np.ndarray): Initial values for decision variables.
            preferential_factors (np.ndarray): preferential factors on how much would the decision maker wish to improve
                                             the values of each objective function.
            nadir (np.ndarray): Nadir vector.
            utopian (np.ndarray): Utopian vector.
            objectives (np.ndarray): The objective function values for each input vector.
            variable_bounds (Optional[np.ndarray): Lower and upper bounds of each variable
                                                   as a 2D numpy array. If undefined variables, None instead.
            method (Union[ScalarMethod, str, None): The optimization method the scalarizer should be minimized with

        Returns:
            Dict: A dictionary with at least the following entries: 'x' indicating the optimal variables found,
            'fun' the optimal value of the optimized function, and 'success' a boolean indicating whether
            the optimization was conducted successfully.

        """

        # scalarize problem using reference point
        asf = ReferencePointASF(preferential_factors, nadir, utopian, rho=1e-6)
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

    def calculate_bounds(self, objectives: Callable, n_objectives: int, x0: np.ndarray, epsilons: np.ndarray,
                         bounds: Union[np.ndarray, None], constraints: Optional[Callable],
                         method: Union[ScalarMethod, str, None]) -> np.ndarray:
        """
        Calculate the new bounds using Epsilon constraint method.

        Args:
            objectives (np.ndarray): The objective function values for each input vector.
            n_objectives (int): Total number of objectives.
            x0 (np.ndarray): Initial values for decision variables.
            epsilons (np.ndarray): Previous iteration point.
            bounds (Union[np.ndarray, None): Bounds for decision variables.
            constraints (Callable): Constraints of the problem.
            method (Union[ScalarMethod, str, None]): The optimization method the scalarizer should be minimized with.

        Returns:
            new_lower_bounds (np.ndarray): New lower bounds for objective functions.
        """

        new_lower_bounds: np.ndarray = [None] * n_objectives

        # set polish to False
        method_e: ScalarMethod = ScalarMethod(
            lambda x, _, **y: differential_evolution(x, **y),
            method_args={"disp": False, "polish": False, "tol": 0.000001, "popsize": 10, "maxiter": 50000},
            use_scipy=True
        )

        # solve new lower bounds for each objective
        for i in range(n_objectives):
            eps = ECM.EpsilonConstraintMethod(objectives,
                                              i,
                                              np.array([val for ind, val in enumerate(epsilons) if ind != i]),
                                              constraints=constraints
                                              )
            cons_evaluate = eps.evaluate_constraints
            scalarized_objective = Scalarizer(objectives, eps)

            minimizer = ScalarMinimizer(scalarized_objective, bounds, constraint_evaluator=cons_evaluate,
                                        method=method_e)
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
            np.ndarray: Distance to the Pareto optimal set.

        """

        dist = (np.linalg.norm(np.atleast_2d(z_current) - nadir, ord=2, axis=1)) \
               / (np.linalg.norm(np.atleast_2d(f_current) - nadir, ord=2, axis=1))
        return dist * 100


# testing the method
if __name__ == "__main__":
    # example problem from article

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

    ideal = np.array([-6.34, -3.44487179, -7.5, 0.])
    nadir = np.array([-4.751, -2.86054116, -0.32111111, 9.70666666])
    print("Ideal: ", ideal)
    print("Nadir: ", nadir)

    # start solving
    method = Nautilus(problem=prob, ideal=ideal, nadir=nadir)

    print("Let's start solving\n")
    req = method.start()

    # initial preferences
    n_iterations = 3
    req.response = {
        "n_iterations": n_iterations,
        "preference_method": 1,
        "preference_info": np.array([2, 2, 1, 1]),
    }
    print("Step number: 0")
    print("Iteration point: ", nadir)
    print("Lower bounds of objectives: ", ideal)

    # 1 - continue with same preferences
    req = method.iterate(req)
    print("\nStep number: ", method._step_number)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._step_number])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    # print("Upper bounds of objectives:", req.content["upper_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])

    req.response = {
        "step_back": False,
        "short_step": False,
        "use_previous_preference": True,
    }

    # 2 - take a step back and give new preferences
    req = method.iterate(req)
    print("\nStep number: ", method._step_number)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._step_number])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])

    req.response = {
        "step_back": True,
        "short_step": False,
        "use_previous_preference": False,
        "preference_method": 1,
        "preference_info": np.array([2, 3, 1, 4]),
    }

    # 3 - give new preferences
    req = method.iterate(req)
    print("\nStep number: ", method._step_number)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._step_number])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])

    req.response = {
        "step_back": False,
        "use_previous_preference": False,
        "preference_method": 1,
        "preference_info": np.array([1, 2, 1, 2]),
    }

    # 4 - take a step back and provide new preferences
    req = method.iterate(req)
    print("\nStep number: ", method._step_number)
    print("Iteration point: ", req.content["current_iteration_point"])
    print("Pareto optimal vector: ", method._fs[method._step_number])
    print("Lower bounds of objectives: ", req.content["lower_bounds"])
    print("Closeness to Pareto optimal front", req.content["distance"])
    """


    req.response = {
        "step_back": True,
        "short_step": False,
        "use_previous_preference": False,
        "preference_method": 2,
        "preference_info": np.array([30, 70]),
    }

    # 5. continue with the same preferences
    while method._n_iterations_left > 1:
        req = method.iterate(req)
        print("\nStep number: ", method._step_number)
        print("Iteration point: ", req.content["current_iteration_point"])
        print("Pareto optimal vector: ", method._fs[method._step_number])
        print("Lower bounds of objectives: ", req.content["lower_bounds"])
        print("Closeness to Pareto optimal front", req.content["distance"])
        req.response = {"step_back": False,
                        "use_previous_preference": True
                        }

    print("\nEnd of solution process")
    req = method.iterate(req)
    print(req.content)
    """
