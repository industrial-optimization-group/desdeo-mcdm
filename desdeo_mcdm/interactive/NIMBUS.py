from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_mcdm.utilities.solvers import payoff_table_method
from desdeo_problem.problem import DiscreteDataProblem, MOProblem
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import AugmentedGuessASF, MaxOfTwoASF, PointMethodASF, SimpleASF, StomASF
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer, Scalarizer
from desdeo_tools.solver.ScalarSolver import DiscreteMinimizer, ScalarMethod, ScalarMinimizer


class NimbusException(Exception):
    """Risen when an error related to NIMBUS is encountered.

    """

    pass


class NimbusClassificationRequest(BaseRequest):
    """A request to handle the classification of objectives in the synchronous NIMBUS method.

    Args:
        method (NIMBUS): The instance of the NIMBUS method the request should be initialized for.
        ref (np.ndarray): Objective values used as a reference the decision maker is classifying the objectives.

    Attributes:
        self._valid_classifications (List[str]): The valid classifications. Defaults is ['<', '<=', '=', '>=', '0']
    """

    def __init__(self, method: NIMBUS, ref: np.ndarray):
        msg = (
            "Please classify each of the objective values in one of the following categories:"
            "\n\t1. values should improve '<'"
            "\n\t2. values should improve until some desired aspiration level is reached '<='"
            "\n\t3. values with an acceptable level '='"
            "\n\t4. values which may be impaired until some upper bound is reached '>='"
            "\n\t5. values which are free to change '0'"
            "\nProvide the aspiration levels and upper bounds as a vector. For categories 1, 3, and 5,"
            "the value in the vector at the objective's position is ignored. Supply also the number of maximum"
            "solutions to be generated."
        )

        self._method = method
        self._valid_classifications = ["<", "<=", "=", ">=", "0"]
        content = {
            "message": msg,
            "objective_values": ref,
            "classifications": [None],
            "levels": [None],
            "number_of_solutions": 1,
        }
        super().__init__("classification_preference", "required", content=content)

    def validator(self, response: Dict) -> None:
        """Validates a dictionary containing the response of a decision maker. Should contain the keys 
        'classifications', 'levels', and 'number_of_solutions'.

        'classifications' should be a list of strings, where the number of
        elements is equal to the number of objectives being classified, and
        the elements are found in `_valid_classifications`. 'levels' should
        have either aspiration levels or bounds for each objective depending
        on that objective's classification. 'number_of_solutions' should be
        an integer between 1 and 4 indicating the number of intermediate solutions to be
        computed.

        Args:
            response (Dict): See the documentation for `validator`.

        Raises:
            NimbusException: Some discrepancy is encountered in the parsing of the response.
        """
        if "classifications" not in response:
            raise NimbusException("'classifications' entry missing.")

        if "levels" not in response:
            raise NimbusException("'levels' entry missing.")

        if "number_of_solutions" not in response:
            raise NimbusException("'number_of_solutions' entry missing.")

        # check the classifications
        is_valid_cls = map(lambda x: x in self._valid_classifications, response["classifications"],)

        if not all(list(is_valid_cls)):
            raise NimbusException(f"Invalid classification found in {response['classifications']}")

        # check the levels
        if len(np.array(response["levels"]).squeeze()) != self._method._problem.n_of_objectives:
            raise NimbusException(f"Wrong number of levels supplied in {response['levels']}")

        improve_until_inds = np.where(np.array(response["classifications"]) == "<=")[0]

        impaire_until_inds = np.where(np.array(response["classifications"]) == ">=")[0]

        if len(improve_until_inds) > 0:
            # some objectives classified to be improved until some level
            if not np.all(
                np.array(response["levels"])[improve_until_inds] >= self._method._ideal[improve_until_inds]
            ) or not np.all(
                np.array(response["levels"])[improve_until_inds] <= self._method._nadir[improve_until_inds]
            ):
                raise NimbusException("Given levels must be between the nadir and ideal points!")

        if len(impaire_until_inds) > 0:
            # some objectives classified to be improved until some level
            if not np.all(
                np.array(response["levels"])[impaire_until_inds] >= self._method._ideal[impaire_until_inds]
            ) or not np.all(
                np.array(response["levels"])[impaire_until_inds] <= self._method._nadir[impaire_until_inds]
            ):
                raise NimbusException("Given levels must be between the nadir and ideal points!")

        # check maximum number of solutions
        if response["number_of_solutions"] > 4 or response["number_of_solutions"] < 1:
            raise NimbusException("The number of solutions must be between 1 and 4.")

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusSaveRequest(BaseRequest):
    """A request to handle archiving of the solutions computed with NIMBUS.

    Args:
        solution_vectors (List[np.ndarray]): A list of numpy arrays each representing a decision variable vector.
        objective_vectors (List[np.ndarray]): A list of numpy arrays each representing an objective vector.

    Note:
        The objective vector at position 'i' in `objective_vectors` should correspond to the decision variables at
        position 'i' in `solution_vectors`.
    """

    def __init__(
        self, solution_vectors: List[np.ndarray], objective_vectors: List[np.ndarray],
    ):
        msg = (
            "Please specify which solutions shown you would like to save for later viewing. Supply the "
            "indices of such solutions as a list, or supply an empty list if none of the shown solutions "
            "should be saved."
        )
        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "indices": [],
        }
        super().__init__("classification_preference", "required", content=content)

    def validator(self, response: Dict) -> None:
        """Validates a response dictionary. The dictionary should contain the keys 'indices'.

        'indices' should be a list of integers representing an index to the
        lists `solutions_vectors` and `objective_vectors`.

        Args:
            response (Dict): See the documentation for `validator`.

        Raises:
            NimbusException: Some discrepancy is encountered in the parsing of `response`.
        """
        if "indices" not in response:
            raise NimbusException("'indices' entry missing")

        if len(response["indices"]) == 0:
            # nothing to save, continue to next state
            return

        if len(response["indices"]) > len(self.content["objectives"]) or np.min(response["indices"]) < 0:
            # wrong number of indices
            raise NimbusException(f"Invalid indices {response['indices']}")

        if np.max(response["indices"]) >= len(self.content["objectives"]) or np.min(response["indices"]) < 0:
            # out of bounds index
            raise NimbusException(f"Incides {response['indices']} out of bounds.")

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusIntermediateSolutionsRequest(BaseRequest):
    """A request to handle the computation of intermediate points between two previously computed points.

    Args:
        solution_vectors (List[np.ndarray]): A list of numpy arrays each representing a decision variable vector.
        objective_vectors (List[np.ndarray]): A list of numpy arrays each representing an objective vector.

    Note:
        The objective vector at position 'i' in `objective_vectors` should correspond to the decision variables at
        position 'i' in `solution_vectors`. Only the two first entries in each of the lists is relevant. The
        rest is ignored.
    """

    def __init__(
        self, solution_vectors: List[np.ndarray], objective_vectors: List[np.ndarray],
    ):
        msg = (
            "Would you like to see intermediate solutions between two previously computed solutions? "
            "If so, please supply two indices corresponding to the solutions."
        )

        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "indices": [],
            "number_of_desired_solutions": 0,
        }

        super().__init__("classification_preference", "required", content=content)

    def validator(self, response: Dict):
        """Validates a response dictionary. The dictionary should contain the keys 'indices' and 'number_of_solutions'.

        'indices' should be a list of integers representing an index to the
        lists `solutions_vectors` and `objective_vectors`. 'number_of_solutions' should be an integer greater or equal 
        to 1.

        Args:
            response (Dict): See the documentation for `validator`.

        Raises:
            NimbusException: Some discrepancy is encountered in the parsing of `response`.
        """
        if "indices" not in response:
            raise NimbusException("'indices' entry missing.")

        if "number_of_desired_solutions" not in response:
            raise NimbusException("'number_of_desired_solutions' entry missing.")

        if response["number_of_desired_solutions"] < 0:
            raise NimbusException(f"Invalid number of desired solutions {response['number_of_desired_solutions']}.")

        if len(response["indices"]) > 0 and response["number_of_desired_solutions"] == 0:
            raise NimbusException("Indices supplied yet number of desired solutions is zero.")

        if len(response["indices"]) == 0 and response["number_of_desired_solutions"] > 0:
            raise NimbusException("Indices not supplied yet number of desired solutions is greater than zero.")

        if len(response["indices"]) == 0:
            return

        if np.max(response["indices"]) >= len(self.content["objectives"]) or np.min(response["indices"]) < 0:
            # indices out of bounds
            raise NimbusException(f"Invalid indices {response['indices']}")

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusMostPreferredRequest(BaseRequest):
    """A request to handle the indication of a preferred point.

    Args:
        solution_vectors (List[np.ndarray]): A list of numpy arrays each representing a decision variable vector.
        objective_vectors (List[np.ndarray]): A list of numpy arrays each representing an objective vector.

    Note:
        The objective vector at position 'i' in `objective_vectors` should correspond to the decision variables at
        position 'i' in `solution_vectors`. Only the two first entries in each of the lists are relevant. The preferred
        solution will be selected from `objective_vectors`.

    """

    def __init__(
        self, solution_vectors: List[np.ndarray], objective_vectors: List[np.ndarray],
    ):
        msg = "Please select your most preferred solution and whether you would like to continue. "

        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "index": -1,
            "continue": True,
        }

        super().__init__("classification_preference", "required", content=content)

    def validator(self, response: Dict):
        """Validates a response dictionary. The dictionary should contain the keys 'index' and 'continue'.

        'index' is an integer and should indicate the index of the preferred solution is `objective_vectors`. 
        'continue' is a boolean and indicates whether to stop or continue the iteration of Synchronous NIMBUS.

        Args:
            response (Dict): See the documentation for `validator`.

        Raises:
            NimbusException: Some discrepancy is encountered in the parsing of `response`.
        """
        if "index" not in response:
            raise NimbusException(f"'index' entry missing.")

        if "continue" not in response:
            raise NimbusException(f"'continue' entry missing.")

        if not type(response["index"]) == int:
            raise NimbusException(f"The index must be a single integer, found {response['index']}")

        if not type(response["continue"]) == bool:
            raise NimbusException(f"Continue must be a boolean value, found {response['index']}")

        if not (response["index"] >= 0 and response["index"] < len(self.content["objectives"])):
            raise NimbusException(
                (
                    f"The index must be a positive integer less than "
                    f"{len(self.content['objectives'])}, found {response['index']}."
                )
            )

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusStopRequest(BaseRequest):
    """A request to handle the termination of Synchronous NIMBUS.

    Args:
        solutions_final (np.ndarray): A numpy array containing the final decision variable values.
        objective_final (np.ndarray): A numpy array containing the final objective variables which correspond to
        `solution_final`.

    Note:
        This request expects no response.
    """

    def __init__(
        self, solution_final: np.ndarray, objective_final: np.ndarray,
    ):
        msg = "The final solution computed."

        content = {
            "message": msg,
            "solution": solution_final,
            "objective": objective_final,
        }

        super().__init__("classification_preference", "no_interaction", content=content)


class NIMBUS(InteractiveMethod):
    """Implements the synchronous NIMBUS algorithm.

    Args:
        problem (MOProblem): The problem to be solved.
        scalar_method (Optional[Union[ScalarMethod, str]], optional): The method used to solve
            the various ASF minimization problems present in the method. Defaults to 'scipy_de' (differential evolution).
        starting_point(Optional[np.ndarray], optional): The initial solution (objectives) to start classification from.
            If None, a neutral starting point will be computed. 
    Note:
        When a starting point is supplied, decision variables of that point will be approximated to be the variables of the
        solution closest to the starting point. In other words, the decision variables associated to the initial point may be
        inaccurate!
    """

    def __init__(self, problem: Union[MOProblem, DiscreteDataProblem], scalar_method: Optional[Union[ScalarMethod, str]] = "scipy_de", starting_point: Optional[np.ndarray] = None):
        # check if ideal and nadir are defined
        if problem.ideal is None or problem.nadir is None:
            # TODO: use same method as defined in scalar_method
            ideal, nadir = payoff_table_method(problem)
            self._ideal = ideal
            self._nadir = nadir
        else:
            self._ideal = problem.ideal
            self._nadir = problem.nadir

        self._scalar_method = scalar_method

        # check starting point if given
        if starting_point is not None:
            if np.squeeze(starting_point).shape != np.squeeze(self._ideal.shape):
                raise NimbusException(
                    f"The given starting point {starting_point} has mismatching dimensions {starting_point.shape}."
                    )

        if isinstance(problem, MOProblem):
            # Change me to be the right ASF
            # if starting point was given, use that as the reference point
            if starting_point is None:
                reference_point = (self._ideal + self._nadir) / 2
            else:
                reference_point = starting_point

            asf = PointMethodASF(self._nadir, self._ideal)
            scalarizer = Scalarizer(
                lambda x: problem.evaluate(x).objectives,
                asf,
                scalarizer_args={"reference_point": reference_point},
            )

            if problem.n_of_constraints > 0:
                _con_eval = lambda x: problem.evaluate(x).constraints.squeeze()
            else:
                _con_eval = None

            solver = ScalarMinimizer(
                scalarizer, problem.get_variable_bounds(), constraint_evaluator=_con_eval, method=self._scalar_method,
            )
            # TODO: fix tools to check for scipy methods in general and delete me!
            solver._use_scipy = True

            res = solver.minimize(problem.get_variable_upper_bounds() / 2)

            if res["success"]:
                self._current_solution = res["x"]
                self._current_objectives = problem.evaluate(self._current_solution).objectives.squeeze()
            else:
                raise NimbusException("Could not solve the initial ASF.")

        elif isinstance(problem, DiscreteDataProblem):
            # discrete case
            if starting_point is None:
                reference_point = (self._ideal + self._nadir) / 2
            else:
                reference_point = starting_point

            asf = PointMethodASF(self._nadir, self._ideal)
            scalarizer = DiscreteScalarizer(asf, scalarizer_args={"reference_point": reference_point})
            solver = DiscreteMinimizer(scalarizer)

            res = solver.minimize(problem.objectives)
            self._current_solution = problem.decision_variables[res["x"]]
            self._current_objectives = problem.objectives[res["x"]]

        else:
            # unsupported problem type
            raise NimbusException(f"Unsupported problem type {type(problem)}.")

        self._archive_solutions = []
        self._archive_objectives = []
        self._state = "classify"

        super().__init__(problem)

    def start(self) -> Tuple[NimbusClassificationRequest, SimplePlotRequest]:
        """Return the first request to start iterating NIMBUS.
        
        Returns:
            Tuple[NimbusClassificationRequest, SimplePlotRequest]: The first request and
            and a plot request to visualize relevant data.
        """
        return self.request_classification()

    def request_classification(self,) -> Tuple[NimbusClassificationRequest, SimplePlotRequest]:
        return (
            NimbusClassificationRequest(self, self._current_objectives.squeeze()),
            None,
        )

    def create_plot_request(self, objectives: np.ndarray, msg: str) -> SimplePlotRequest:
        """Used to create a plot request for visualizing objective values.

        Args:
            objectives (np.ndarray): A 2D numpy array containing objective vectors to be visualized.
            msg (str): A message to be displayed in the context of a visualization.

        Returns:
            SimplePlotRequest: A plot request to create a visualization.
        """
        if isinstance(self._problem, MOProblem):
            dimensions_data = pd.DataFrame(
                index=["minimize", "ideal", "nadir"], columns=self._problem.get_objective_names(),
            )
            dimensions_data.loc["minimize"] = self._problem._max_multiplier
            dimensions_data.loc["ideal"] = self._ideal
            dimensions_data.loc["nadir"] = self._nadir

            data = pd.DataFrame(objectives, columns=self._problem.get_objective_names())
        else:
            dimensions_data = pd.DataFrame(index=["minimize", "ideal", "nadir"], columns=self._problem.objective_names,)
            dimensions_data.loc["minimize"] = [1 for _ in self._problem.objective_names]
            dimensions_data.loc["ideal"] = self._ideal
            dimensions_data.loc["nadir"] = self._nadir

            data = pd.DataFrame(objectives, columns=self._problem.objective_names)

        plot_request = SimplePlotRequest(data=data, dimensions_data=dimensions_data, message=msg,)

        return plot_request

    def handle_classification_request(
        self, request: NimbusClassificationRequest
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        """Handles a classification request.
        
        Args:
            request (NimbusClassificationRequest): A classification request with the
                response attribute set.
        
        Returns:
            Tuple[NimbusSaveRequest, SimplePlotRequest]: A NIMBUS save request and a plot request
            with the solutions the decision maker can choose from to save for alter use.
        """
        improve_inds = np.where(np.array(request.response["classifications"]) == "<")[0]

        acceptable_inds = np.where(np.array(request.response["classifications"]) == "=")[0]

        free_inds = np.where(np.array(request.response["classifications"]) == "0")[0]

        improve_until_inds = np.where(np.array(request.response["classifications"]) == "<=")[0]

        impaire_until_inds = np.where(np.array(request.response["classifications"]) == ">=")[0]

        # calculate the new solutions
        return self.calculate_new_solutions(
            int(request.response["number_of_solutions"]),
            np.array(request.response["levels"]),
            improve_inds,
            improve_until_inds,
            acceptable_inds,
            impaire_until_inds,
            free_inds,
        )

    def handle_save_request(
        self, request: NimbusSaveRequest
    ) -> Tuple[NimbusIntermediateSolutionsRequest, SimplePlotRequest]:
        """Handles a save request.
        
        Args:
            request (NimbusSaveRequest): A save request with the response attribute set.
        
        Returns:
            Tuple[NimbusIntermediateSolutionsRequest, SimplePlotRequest]: Return an
            intermediate solution request where the decision maker can specify whether they
            would like to see intermediate solution between two previously computed solutions.
            The plot request has the available solutions.
        """
        return self.save_solutions_to_archive(
            np.array(request.content["objectives"]),
            np.array(request.content["solutions"]),
            np.array(request.response["indices"]),
        )

    def handle_intermediate_solutions_request(
        self, request: NimbusIntermediateSolutionsRequest
    ) -> Tuple[
        Union[NimbusSaveRequest, NimbusMostPreferredRequest], SimplePlotRequest,
    ]:
        """Handles an intermediate solutions request.
        
        Args:
            request (NimbusIntermediateSolutionsRequest): A NIMBUS intermediate solutions
                request with the response attribute set.
        
        Returns:
            Tuple[Union[NimbusSaveRequest, NimbusMostPreferredRequest], SimplePlotRequest,]:
            Return either a save request or a preferred solution request. The former is returned if the
            decision maker wishes to see intermediate points, the latter otherwise. Also a plot request is
            returned with the solutions available in it.
        """
        if len(request.response["indices"]) > 0:
            return self.compute_intermediate_solutions(
                np.array(request.content["solutions"])[request.response["indices"]],
                int(request.response["number_of_desired_solutions"]),
            )

        return self.request_most_preferred_solution(
            np.array(request.content["solutions"]), np.array(request.content["objectives"]),
        )

    def handle_most_preferred_request(
        self, request: NimbusMostPreferredRequest
    ) -> Tuple[Union[NimbusClassificationRequest, NimbusStopRequest], SimplePlotRequest]:
        """Handles a preferred solution request.
        
        Args:
            request (NimbusMostPreferredRequest): A NIMBUS preferred solution request with the
                response attribute set.
        
        Returns:
            Tuple[Union[NimbusClassificationRequest, NimbusStopRequest], SimplePlotRequest]:
            Return a classification request if the decision maker wishes to continue. If the
            decision maker wishes to stop, return a stop request. Also return a plot 
            request with all the solutions saved so far.
        """
        self.update_current_solution(
            np.array(request.content["solutions"]),
            np.array(request.content["objectives"]),
            np.array(request.response["index"]),
        )

        if not request.response["continue"]:
            return self.request_stop()

        else:
            return self.request_classification()

    def request_stop(self) -> Tuple[NimbusStopRequest, SimplePlotRequest]:
        """Create a NimbusStopRequest based on self.
        
        Returns:
            Tuple[NimbusStopRequest, SimplePlotRequest]: A stop request and a plot
            request with the final solution chosen in it.
        """
        request = NimbusStopRequest(self._current_solution, self._current_objectives)

        msg = "Final solution reached"
        plot_request = self.create_plot_request(np.atleast_2d(self._current_objectives), msg)

        return request, plot_request

    def request_most_preferred_solution(
        self, solutions: np.ndarray, objectives: np.ndarray
    ) -> Tuple[NimbusMostPreferredRequest, SimplePlotRequest]:
        """Create a NimbusMostPreferredRequest.

        Args:
            solutions (np.ndarray): A 2D numpy array of decision variable vectors.
            objectives (np.ndarray): A 2D numpy array of objective value vectors.

        Returns:
            Tuple[NimbusMostPreferredRequest, SimplePlotRequest]: The requests based on the given arguments.

        Note:
            The 'i'th decision variable vector in `solutions` should correspond to the 'i'th objective value vector in
            `objectives`.

        """
        # request most preferred solution
        request = NimbusMostPreferredRequest(list(solutions), list(objectives))

        msg = "Computed solutions"
        plot_request = self.create_plot_request(objectives, msg)

        return request, plot_request

    def compute_intermediate_solutions(
        self, solutions: np.ndarray, n_desired: int,
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        """Computes intermediate solution between two solutions computed earlier.

        Args:
            solutions (np.ndarray): The solutions between which the intermediate solutions should
                be computed.
            n_desired (int): The number of intermediate solutions desired.
        
        Raises:
            NimbusException
        
        Returns:
            Tuple[NimbusSaveRequest, SimplePlotRequest]: A save request with the computed intermediate
            points, and a plot request to visualize said points.
        """
        # vector between the two solutions
        between = solutions[0] - solutions[1]
        norm = np.linalg.norm(between)
        between_norm = between / norm

        # the plus 2 assumes we are interested only in n_desired points BETWEEN the
        # two supplied solutions
        step_size = norm / (2 + n_desired)

        intermediate_points = np.array([solutions[1] + i * step_size * between_norm for i in range(1, n_desired + 1)])

        # project each of the intermediate solutions to the Pareto front
        intermediate_solutions = np.zeros(intermediate_points.shape)
        intermediate_objectives = np.zeros((n_desired, self._problem.n_of_objectives))
        asf = PointMethodASF(self._nadir, self._ideal)

        for i in range(n_desired):
            if isinstance(self._problem, MOProblem):
                scalarizer = Scalarizer(
                    lambda x: self._problem.evaluate(x).objectives,
                    asf,
                    scalarizer_args={"reference_point": self._problem.evaluate(intermediate_points[i]).objectives},
                )

                if self._problem.n_of_constraints > 0:
                    cons = lambda x: self._problem.evaluate(x).constraints.squeeze()
                else:
                    cons = None

                solver = ScalarMinimizer(
                    scalarizer, self._problem.get_variable_bounds(), cons, method=self._scalar_method,
                )

                res = solver.minimize(self._current_solution)
                intermediate_solutions[i] = res["x"]
                intermediate_objectives[i] = self._problem.evaluate(res["x"]).objectives

            else:
                # discrete case
                scalarizer = DiscreteScalarizer(
                    asf,
                    scalarizer_args={
                        "reference_point": self._problem.objectives[self._problem.find_closest(intermediate_points[i])]
                    },
                )
                solver = DiscreteMinimizer(scalarizer)

                res = solver.minimize(self._problem.objectives)
                intermediate_solutions[i] = self._problem.decision_variables[res["x"]]
                intermediate_objectives[i] = self._problem.objectives[res["x"]]

        # create appropriate requests
        save_request = NimbusSaveRequest(list(intermediate_solutions), list(intermediate_objectives))

        msg = "Computed intermediate solutions"
        plot_request = self.create_plot_request(intermediate_objectives, msg)

        return save_request, plot_request

    def save_solutions_to_archive(
        self, objectives: np.ndarray, decision_variables: np.ndarray, indices: List[int],
    ) -> Tuple[NimbusIntermediateSolutionsRequest, None]:
        """Save solutions to the archive. Saves also the corresponding objective function
        values.
        
        Args:
            objectives (np.ndarray): Available objectives.
            decision_variables (np.ndarray): Available solutions.
            indices (List[int]): Indices of the solutions to be saved.
        
        Returns:
            Tuple[NimbusIntermediateSolutionsRequest, None]: An intermediate solutions request asking the
            decision maker whether they would like to generate intermediata solutions between two existing solutions.
            Also returns a plot request to visualize the available solutions between which the intermediate solutions
            should be computed.
        """
        mask = np.ones(objectives.shape[0], dtype=bool)

        if len(indices) > 0:
            self._archive_objectives.extend(list(objectives[indices]))
            self._archive_solutions.extend(list(decision_variables[indices]))

            mask[indices] = False

        req_objectives = self._archive_objectives + list(objectives[mask])
        req_solutions = self._archive_solutions + list(decision_variables[mask])

        # create intermediate solutions request
        request = NimbusIntermediateSolutionsRequest(req_solutions, req_objectives)

        msg = "Computed new solutions"
        plot_request = self.create_plot_request(req_objectives, msg)

        return request, plot_request

    def calculate_new_solutions(
        self,
        number_of_solutions: int,
        levels: np.ndarray,
        improve_inds: np.ndarray,
        improve_until_inds: np.ndarray,
        acceptable_inds: np.ndarray,
        impaire_until_inds: np.ndarray,
        free_inds: np.ndarray,
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        """Calculates new solutions based on classifications supplied by the decision maker by
            solving ASF problems.
        
        Args:
            number_of_solutions (int): Number of solutions, should be between 1 and 4.
            levels (np.ndarray): Aspiration and upper bounds relevant to the some of the classifications.
            improve_inds (np.ndarray): Indices corresponding to the objectives which should be improved.
            improve_until_inds (np.ndarray): Like above, but improved until an aspiration level is reached.
            acceptable_inds (np.ndarray): Indices of objectives which are acceptable as they are now.
            impaire_until_inds (np.ndarray): Indices of objectives which may be impaired until an upper limit is
                reached.
            free_inds (np.ndarray): Indices of objectives which may change freely.
        
        Returns:
            Tuple[NimbusSaveRequest, SimplePlotRequest]: A save request with the newly computed solutions, and 
            a plot request to visualize said solutions.
        """
        results = []

        # always computed
        asf_1 = MaxOfTwoASF(self._nadir, self._ideal, improve_inds, improve_until_inds)

        if isinstance(self._problem, MOProblem):

            def cons_1(
                x: np.ndarray,
                f_current: np.ndarray = self._current_objectives,
                levels: np.ndarray = levels,
                improve_until_inds: np.ndarray = improve_until_inds,
                improve_inds: np.ndarray = improve_inds,
                impaire_until_inds: np.ndarray = impaire_until_inds,
                acceptable_inds: np.ndarray = acceptable_inds,
            ):
                f = self._problem.evaluate(x).objectives.squeeze()

                res_1 = f_current[improve_inds] - f[improve_inds]
                res_2 = f_current[improve_until_inds] - f[improve_until_inds]
                res_3 = f_current[acceptable_inds] - f[acceptable_inds]
                res_4 = levels[impaire_until_inds] - f[impaire_until_inds]

                res = np.hstack((res_1, res_2, res_3, res_4))

                if self._problem.n_of_constraints > 0:
                    res_prob = self._problem.evaluate(x).constraints.squeeze()

                    return np.hstack((res_prob, res))

                else:
                    return res

            scalarizer_1 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives, asf_1, scalarizer_args={"reference_point": levels},
            )

            solver_1 = ScalarMinimizer(
                scalarizer_1, self._problem.get_variable_bounds(), cons_1, method=self._scalar_method,
            )

            res_1 = solver_1.minimize(self._current_solution)

        else:
            # discrete case
            def cons_1(
                objective_vectors: np.ndarray,
                f_current: np.ndarray = self._current_objectives,
                levels: np.ndarray = levels,
                improve_until_inds: np.ndarray = improve_until_inds,
                improve_inds: np.ndarray = improve_inds,
                impaire_until_inds: np.ndarray = impaire_until_inds,
                acceptable_inds: np.ndarray = acceptable_inds,
            ):
                res_1 = f_current[improve_inds] - objective_vectors[:, improve_inds]
                res_2 = f_current[improve_until_inds] - objective_vectors[:, improve_until_inds]
                res_3 = f_current[acceptable_inds] - objective_vectors[:, acceptable_inds]
                res_4 = levels[impaire_until_inds] - objective_vectors[:, impaire_until_inds]

                res = np.hstack((res_1, res_2, res_3, res_4)) >= 0

                return np.all(res, axis=1)

            scalarizer_1 = DiscreteScalarizer(asf_1, scalarizer_args={"reference_point": levels})

            solver_1 = DiscreteMinimizer(scalarizer_1, cons_1)

            res_1 = solver_1.minimize(self._problem.objectives)

        results.append(res_1)

        if number_of_solutions > 1:
            # create the reference point needed in the rest of the ASFs
            z_bar = np.zeros(self._problem.n_of_objectives)
            z_bar[improve_inds] = self._ideal[improve_inds]
            z_bar[improve_until_inds] = levels[improve_until_inds]
            z_bar[acceptable_inds] = self._current_objectives[acceptable_inds]
            z_bar[impaire_until_inds] = levels[impaire_until_inds]
            z_bar[free_inds] = self._nadir[free_inds]

            # second ASF
            asf_2 = StomASF(self._ideal)

            # cons_2 can be used in the rest of the ASF scalarizations, it's not a bug!
            if isinstance(self._problem, MOProblem) and self._problem.n_of_constraints > 0:
                cons_2 = lambda x: self._problem.evaluate(x).constraints.squeeze()
            else:
                cons_2 = None

            if isinstance(self._problem, MOProblem):
                scalarizer_2 = Scalarizer(
                    lambda x: self._problem.evaluate(x).objectives, asf_2, scalarizer_args={"reference_point": z_bar},
                )

                solver_2 = ScalarMinimizer(
                    scalarizer_2, self._problem.get_variable_bounds(), cons_2, method=self._scalar_method,
                )

                res_2 = solver_2.minimize(self._current_solution)

            else:
                # discrete case
                scalarizer_2 = DiscreteScalarizer(asf_2, scalarizer_args={"reference_point": z_bar})

                solver_2 = DiscreteMinimizer(scalarizer_2, cons_2)

                res_2 = solver_2.minimize(self._problem.objectives)

            results.append(res_2)

        if number_of_solutions > 2:
            # asf 3
            asf_3 = PointMethodASF(self._nadir, self._ideal)

            if isinstance(self._problem, MOProblem):
                scalarizer_3 = Scalarizer(
                    lambda x: self._problem.evaluate(x).objectives, asf_3, scalarizer_args={"reference_point": z_bar},
                )

                solver_3 = ScalarMinimizer(
                    scalarizer_3, self._problem.get_variable_bounds(), cons_2, method=self._scalar_method,
                )

                res_3 = solver_3.minimize(self._current_solution)

            else:
                # discrete case
                scalarizer_3 = DiscreteScalarizer(asf_3, scalarizer_args={"reference_point": z_bar})

                solver_3 = DiscreteMinimizer(scalarizer_3, cons_2)

                res_3 = solver_3.minimize(self._problem.objectives)

            results.append(res_3)

        if number_of_solutions > 3:
            # asf 4
            asf_4 = AugmentedGuessASF(self._nadir, self._ideal, free_inds)

            if isinstance(self._problem, MOProblem):

                scalarizer_4 = Scalarizer(
                    lambda x: self._problem.evaluate(x).objectives, asf_4, scalarizer_args={"reference_point": z_bar},
                )

                solver_4 = ScalarMinimizer(
                    scalarizer_4, self._problem.get_variable_bounds(), cons_2, method=self._scalar_method,
                )

                res_4 = solver_4.minimize(self._current_solution)

            else:
                # discrete case
                scalarizer_4 = DiscreteScalarizer(asf_4, scalarizer_args={"reference_point": z_bar})

                solver_4 = DiscreteMinimizer(scalarizer_4, cons_2)

                res_4 = solver_4.minimize(self._problem.objectives)

            results.append(res_4)

        # create the save request
        if isinstance(self._problem, MOProblem):
            solutions = [res["x"] for res in results]
            objectives = [self._problem.evaluate(x).objectives.squeeze() for x in solutions]

        else:
            # discrete case
            solutions = [self._problem.decision_variables[res["x"]] for res in results]
            objectives = [self._problem.objectives[res["x"]] for res in results]

        save_request = NimbusSaveRequest(solutions, objectives)

        msg = "Computed new solutions."
        plot_request = self.create_plot_request(objectives, msg)

        return save_request, plot_request

    def update_current_solution(self, solutions: np.ndarray, objectives: np.ndarray, index: int) -> None:
        """Update the state of self with a new current solution and the corresponding objective values. This solution is
        used in the classification phase of synchronous NIMBUS.

        Args:
            solutions (np.ndarray): A 2D numpy array of decision variable vectors.
            objectives (np.ndarray): A 2D numpy array of objective value vectors.
            index (int): The index of the solution in `solutions` and `objectives`.

        Returns:
            Tuple[NimbusMostPreferredRequest, SimplePlotRequest]: The requests based on the given arguments.

        Note:
            The 'i'th decision variable vector in `solutions` should correspond to the 'i'th objective value vector in
            `objectives`.

        """
        self._current_solution = solutions[index].squeeze()
        self._current_objectives = objectives[index].squeeze()

        return None

    def iterate(
        self,
        request: Union[
            NimbusClassificationRequest,
            NimbusSaveRequest,
            NimbusIntermediateSolutionsRequest,
            NimbusMostPreferredRequest,
            NimbusStopRequest,
        ],
    ) -> Tuple[
        Union[NimbusClassificationRequest, NimbusSaveRequest, NimbusIntermediateSolutionsRequest,],
        Union[SimplePlotRequest, None],
    ]:
        """Implements a finite state machine to iterate over the different steps defined in Synchronous NIMBUS based on a supplied request.
        
        Args:
            request (Union[NimbusClassificationRequest,NimbusSaveRequest,NimbusIntermediateSolutionsRequest,NimbusMostPreferredRequest,NimbusStopRequest,]):
                A request based on the next step in the NIMBUS algorithm is taken.
        
        Raises:
            NimbusException: If a wrong type of request is supplied based on the current state NIMBUS is in.
        
        Returns:
            Tuple[Union[NimbusClassificationRequest,NimbusSaveRequest,NimbusIntermediateSolutionsRequest,],Union[SimplePlotRequest, None],]:
            The next logically sound request.
        """
        if self._state == "classify":
            if type(request) != NimbusClassificationRequest:
                raise NimbusException(
                    f"Expected request type {type(NimbusClassificationRequest)}, was {type(request)}."
                )

            requests = self.handle_classification_request(request)
            self._state = "archive"
            return requests

        elif self._state == "archive":
            if type(request) != NimbusSaveRequest:
                raise NimbusException(f"Expected request type {type(NimbusSaveRequest)}, was {type(request)}.")

            requests = self.handle_save_request(request)
            self._state = "intermediate"
            return requests

        elif self._state == "intermediate":
            if type(request) != NimbusIntermediateSolutionsRequest:
                raise NimbusException(
                    f"Expected request type {type(NimbusIntermediateSolutionsRequest)}, was {type(request)}."
                )

            requests = self.handle_intermediate_solutions_request(request)

            if type(requests[0]) == NimbusSaveRequest:
                self._state = "archive"
            elif type(requests[0]) == NimbusMostPreferredRequest:
                self._state = "preferred"

            return requests

        elif self._state == "preferred":
            if type(request) != NimbusMostPreferredRequest:
                raise NimbusException(f"Expected request type {type(NimbusMostPreferredRequest)}, was {type(request)}.")

            requests = self.handle_most_preferred_request(request)

            if type(requests[0]) == NimbusStopRequest:
                self._state = "end"

            elif type(requests[0]) == NimbusClassificationRequest:
                self._state = "classify"

            return requests

        elif self._state == "end":
            # end
            return request, None

        else:
            # unknown state error
            raise NimbusException(f"Unknown state '{self._state}' encountered.")


if __name__ == "__main__":
    from desdeo_problem.problem.Objective import _ScalarObjective
    from desdeo_problem.problem.Variable import variable_builder
    from desdeo_problem.problem.Constraint import ScalarConstraint

    # create the problem
    def f_1(x):
        res = 4.07 + 2.27 * x[:, 0]
        return -res

    def f_2(x):
        res = 2.60 + 0.03 * x[:, 0] + 0.02 * x[:, 1] + 0.01 / (1.39 - x[:, 0] ** 2) + 0.30 / (1.39 - x[:, 1] ** 2)
        return -res

    def f_3(x):
        res = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
        return -res

    def f_4(x):
        res = 0.96 - 0.96 / (1.09 - x[:, 1] ** 2)
        return -res

    def f_5(x):
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    def c_1(x, f=None):
        x = x.squeeze()
        return (x[0] + x[1]) - 0.5

    f1 = _ScalarObjective(name="f1", evaluator=f_1)
    f2 = _ScalarObjective(name="f2", evaluator=f_2)
    f3 = _ScalarObjective(name="f3", evaluator=f_3)
    f4 = _ScalarObjective(name="f4", evaluator=f_4)
    f5 = _ScalarObjective(name="f5", evaluator=f_5)
    varsl = variable_builder(
        ["x_1", "x_2"], initial_values=[0.5, 0.5], lower_bounds=[0.3, 0.3], upper_bounds=[1.0, 1.0],
    )
    c1 = ScalarConstraint("c1", 2, 5, evaluator=c_1)
    problem = MOProblem(variables=varsl, objectives=[f1, f2, f3, f4, f5], constraints=[c1])

    method = NIMBUS(problem, scalar_method="scipy_de")
    reqs = method.request_classification()[0]

    response = {}
    response["classifications"] = ["<", "<=", "=", ">=", "0"]
    response["levels"] = [-6, -3, -5, 8, 0.349]
    response["number_of_solutions"] = 3
    reqs.response = response
    res_1 = method.iterate(reqs)[0]
    res_1.response = {"indices": []}

    res_2 = method.iterate(res_1)[0]
    response = {}
    response["indices"] = []
    response["number_of_desired_solutions"] = 0
    res_2.response = response

    res_3 = method.iterate(res_2)[0]
    response_pref = {}
    response_pref["index"] = 1
    response_pref["continue"] = True
    res_3.response = response_pref

    res_4 = method.iterate(res_3)

    """
    import matplotlib.pyplot as plt
    from desdeo_problem.problem import _ScalarObjective, variable_builder

    def f_1(xs: np.ndarray):
        xs = np.atleast_2d(xs)
        xs_plusone = np.roll(xs, 1, axis=1)
        return np.sum(-10 * np.exp(-0.2 * np.sqrt(xs[:, :-1] ** 2 + xs_plusone[:, :-1] ** 2)), axis=1)

    def f_2(xs: np.ndarray):
        xs = np.atleast_2d(xs)
        return np.sum(np.abs(xs) ** 0.8 + 5 * np.sin(xs ** 3), axis=1)

    varsl = variable_builder(
        ["x_1", "x_2", "x_3"], initial_values=[0, 0, 0], lower_bounds=[-5, -5, -5], upper_bounds=[5, 5, 5],
    )

    f1 = _ScalarObjective(name="f1", evaluator=f_1)
    f2 = _ScalarObjective(name="f2", evaluator=f_2)

    # For AI_DM ranges are always defined as IDEAL, NADIR.
    f1_range = [-20, -14]
    f2_range = [-14, 0.5]
    x1 = np.linspace(min(f1_range), max(f1_range), 1000)
    x2 = np.linspace(min(f2_range), max(f2_range), 1000)
    y = np.linspace(0, 0, 1000)

    problem = MOProblem(variables=varsl, objectives=[f1, f2], ideal=np.array([-20, -12]), nadir=np.array([-14, 0.5]))

    from desdeo_mcdm.interactive.NIMBUS import NIMBUS
    from scipy.optimize import differential_evolution, minimize

    scalar_method = ScalarMethod(
        lambda x, _, **y: differential_evolution(x, **y), use_scipy=True, method_args={"polish": True, "disp": True}
    )

    method = NIMBUS(problem, scalar_method)

    classification_request, plot_request = method.start()

    # print(classification_request.content.keys())
    # print(classification_request.content["message"])

    print(classification_request.content["objective_values"])

    # Plotting F1!
    plt.scatter(x1, y, label="Range")
    plt.scatter(-15, 0, label="P1", c="r")
    plt.scatter(-18, 0, label="P2", c="r")
    plt.scatter(classification_request.content["objective_values"][0], 0, label="Present Position")
    plt.xlabel("Objective Function")
    plt.ylabel("Value")
    plt.title("Objective Function F1")

    plt.show()

    # Plotting F2!
    plt.scatter(x2, y, label="Range")
    plt.scatter(-6, 0, label="P1", c="r")
    plt.scatter(-9, 0, label="P2", c="r")
    plt.scatter(classification_request.content["objective_values"][1], 0, label="Present Position")
    plt.xlabel("Objective Function")
    plt.ylabel("Value")
    plt.title("Objective Function F2")

    plt.show()

    response = {"classifications": ["<", "="], "levels": [-7.133133133133133, 0], "number_of_solutions": 1}
    classification_request.response = response

    save_request, plot_request = method.iterate(classification_request)

    print(save_request.content["objectives"])
    print(save_request.content["solutions"])

    # Plotting F1!
    plt.scatter(x1, y, label="Range")
    plt.scatter(-15, 0, label="P1", c="r")
    plt.scatter(-18, 0, label="P2", c="r")
    plt.scatter(save_request.content["objectives"][0][0], 0, label="Present Position")
    plt.xlabel("Objective Function")
    plt.ylabel("Value")
    plt.title("Objective Function F1")

    plt.show()

    # Plotting F2!
    plt.scatter(x2, y, label="Range")
    plt.scatter(-6, 0, label="P1", c="r")
    plt.scatter(-9, 0, label="P2", c="r")
    plt.scatter(save_request.content["objectives"][0][1], 0, label="Present Position")
    plt.xlabel("Objective Function")
    plt.ylabel("Value")
    plt.title("Objective Function F2")

    plt.show()

"""
