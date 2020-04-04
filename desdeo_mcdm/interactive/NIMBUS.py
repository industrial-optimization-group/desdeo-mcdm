import numpy as np
import pandas as pd

from typing import List, Union, Tuple, Optional, Dict

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_mcdm.utilities.solvers import payoff_table_method
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import (
    SimpleASF,
    MaxOfTwoASF,
    StomASF,
    PointMethodASF,
    AugmentedGuessASF,
)
from desdeo_tools.solver.ScalarSolver import ScalarMinimizer, ScalarMethod
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_problem.Problem import MOProblem


class NimbusException(Exception):
    pass


class NimbusClassificationRequest(BaseRequest):
    def __init__(self, method, ref: np.ndarray):
        msg = (
            "Please classify each of the objective values in one of the following categories:"
            "\n\t1. values should improve '<'"
            "\n\t2. values should improve until some desired aspiration level is reached '<='"
            "\n\t3. values with an acceptable level '='"
            "\n\t4. values which may be impaired until some upper bound is reached '>='"
            "\n\t5. values which are free to change '0'"
            "\nProvide the aspiration levels and upper bounds as a vector. For categories 1, 3, and 5,"
            "the value in the vector at the objective's position is ignored. Suppy also the number of maximum"
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
        super().__init__(
            "classification_preference", "required", content=content
        )

    def validator(self, response: Dict) -> None:
        if not "classifications" in response:
            raise NimbusException("'classifications' entry missing.")

        if not "levels" in response:
            raise NimbusException("'levels' entry missing.")

        if not "number_of_solutions" in response:
            raise NimbusException("'number_of_solutions' entry missing.")

        # check the classifications
        is_valid_cls = map(
            lambda x: x in self._valid_classifications,
            response["classifications"],
        )

        if not all(list(is_valid_cls)):
            raise NimbusException(
                f"Invalid classificaiton found in {response['classifications']}"
            )

        # check the levels
        if (
            len(np.array(response["levels"]).squeeze())
            != self._method._problem.n_of_objectives
        ):
            raise NimbusException(
                f"Wrong number of levels supplied in {response['levels']}"
            )

        improve_until_inds = np.where(
            np.array(response["classifications"]) == "<="
        )[0]

        impaire_until_inds = np.where(
            np.array(response["classifications"]) == ">="
        )[0]

        if len(improve_until_inds) > 0:
            # some objectives classified to be improved until some level
            if not np.all(
                np.array(response["levels"])[improve_until_inds]
                >= self._method._ideal[improve_until_inds]
            ) or not np.all(
                np.array(response["levels"])[improve_until_inds]
                <= self._method._nadir[improve_until_inds]
            ):
                raise NimbusException(
                    f"Given levels must be between the nadir and ideal points!"
                )

        if len(impaire_until_inds) > 0:
            # some objectives classified to be improved until some level
            if not np.all(
                np.array(response["levels"])[impaire_until_inds]
                >= self._method._ideal[impaire_until_inds]
            ) or not np.all(
                np.array(response["levels"])[impaire_until_inds]
                <= self._method._nadir[impaire_until_inds]
            ):
                raise NimbusException(
                    f"Given levels must be between the nadir and ideal points!"
                )

        # check maximum number of solutions
        if (
            response["number_of_solutions"] > 4
            or response["number_of_solutions"] < 1
        ):
            raise NimbusException(
                f"The number of solutions must be between 1 and 4."
            )

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusSaveRequest(BaseRequest):
    def __init__(
        self,
        solution_vectors: List[np.ndarray],
        objective_vectors: List[np.ndarray],
    ):
        msg = (
            "Please specify which solutions shown you would like to save for later viewing. Supply the "
            "indices of such solutions as a list, or supply an empty list if none of the shown soulutions "
            "should be saved."
        )
        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "indices": [],
        }
        super().__init__(
            "classification_preference", "required", content=content
        )

    def validator(self, response: Dict):
        if "indices" not in response:
            raise NimbusException("'indices' entry missing")

        if not response["indices"]:
            # nothing to save, continue to next state
            return

        if (
            len(response["indices"]) > len(self.content["objectives"])
            or np.min(response["indices"]) < 0
        ):
            # wrong number of indices
            raise NimbusException(f"Invalid indices {response['indices']}")

        if (
            np.max(response["indices"]) >= len(self.content["objectives"])
            or np.min(response["indices"]) < 0
        ):
            # out of bounds index
            raise NimbusException(
                f"Incides {response['indices']} out of bounds."
            )

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusIntermediateSolutionsRequest(BaseRequest):
    def __init__(
        self,
        solution_vectors: List[np.ndarray],
        objective_vectors: List[np.ndarray],
    ):
        msg = (
            "Would you like to see intermediate solutions between two previusly computed solutions? "
            "If so, please supply two indices corresponding to the solutions."
        )

        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "indices": [],
            "number_of_desired_solutions": 0,
        }

        super().__init__(
            "classification_preference", "required", content=content
        )

    def validator(self, response: Dict):
        if not "indices" in response:
            raise NimbusException("'indices' entry missing.")

        if not "number_of_desired_solutions" in response:
            raise NimbusException(
                "'number_of_desired_solutions' entry missing."
            )

        if response["number_of_desired_solutions"] < 0:
            raise NimbusException(
                f"Invalid number of desired solutions {response['number_of_desired_solutions']}."
            )

        if (
            not response["indices"]
            and response["number_of_desired_solutions"] > 0
        ):
            raise NimbusException(
                "Indices supplied yet number of desired soutions is greater than zero."
            )

        if response["indices"] and response["number_of_desired_solutions"] == 0:
            raise NimbusException(
                "Indices not supplied yet number of desired soutions is zero."
            )

        if not response["indices"]:
            return

        if (
            np.max(response["indices"]) >= len(self.content["objectives"])
            or np.min(response["indices"]) < 0
        ):
            # indices out of bounds
            raise NimbusException(f"Invalid indices {response['indices']}")

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NimbusMostPreferredRequest(BaseRequest):
    def __init__(
        self,
        solution_vectors: List[np.ndarray],
        objective_vectors: List[np.ndarray],
    ):
        msg = "Please select your most preferred solution and whether you would like to continue. "

        content = {
            "message": msg,
            "solutions": solution_vectors,
            "objectives": objective_vectors,
            "index": -1,
            "continue": True,
        }

        super().__init__(
            "classification_preference", "required", content=content
        )

    def validator(self, response: Dict):
        if "index" not in response:
            raise NimbusException(f"'index' entry missing.")

        if "continue" not in response:
            raise NimbusException(f"'continue' entry missing.")

        if not type(response["index"]) == int:
            raise NimbusException(
                f"The index must be a single integer, found {response['index']}"
            )

        if not type(response["continue"]) == bool:
            raise NimbusException(
                f"Continue must be a boolean value, found {response['index']}"
            )

        if not (
            response["index"] >= 0
            and response["index"] < len(self.content["objectives"])
        ):
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
    def __init__(
        self, solution_final: np.ndarray, objective_final: np.ndarray,
    ):
        msg = "The final solution computed."

        content = {
            "message": msg,
            "solution": solution_final,
            "objective": objective_final,
        }

        super().__init__(
            "classification_preference", "no_interaction", content=content
        )


class NIMBUS(InteractiveMethod):
    """Implements the synchronous NIMBUS variant.
    """

    def __init__(
        self, problem: MOProblem, scalar_method: Optional[ScalarMethod] = None
    ):
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

        # generate Pareto optimal starting point
        asf = SimpleASF(np.ones(self._ideal.shape))
        scalarizer = Scalarizer(
            lambda x: problem.evaluate(x).objectives,
            asf,
            scalarizer_args={"reference_point": np.atleast_2d(self._ideal)},
        )

        if problem.n_of_constraints > 0:
            _con_eval = lambda x: problem.evaluate(x).constraints.squeeze()
        else:
            _con_eval = None

        solver = ScalarMinimizer(
            scalarizer,
            problem.get_variable_bounds(),
            constraint_evaluator=_con_eval,
            method=self._scalar_method,
        )
        # TODO: fix tools to check for scipy methods in general and delete me!
        solver._use_scipy = True

        res = solver.minimize(problem.get_variable_upper_bounds() / 2)

        if res["success"]:
            self._current_solution = res["x"]
            self._current_objectives = problem.evaluate(
                self._current_solution
            ).objectives.squeeze()

        self._archive_solutions = []
        self._archive_objectives = []
        self._state = "classify"

        super().__init__(problem)

    def start(self) -> Tuple[NimbusClassificationRequest, SimplePlotRequest]:
        return self.request_classification()

    def request_classification(
        self,
    ) -> Tuple[NimbusClassificationRequest, SimplePlotRequest]:
        return (
            NimbusClassificationRequest(
                self, self._current_objectives.squeeze()
            ),
            None,
        )

    def create_plot_request(
        self, objectives: np.ndarray, msg: str
    ) -> SimplePlotRequest:
        dimensions_data = pd.DataFrame(
            index=["minimize", "ideal", "nadir"],
            columns=self._problem.get_objective_names(),
        )
        dimensions_data.loc["minimize"] = self._problem._max_multiplier
        dimensions_data.loc["ideal"] = self._ideal
        dimensions_data.loc["nadir"] = self._nadir

        data = pd.DataFrame(
            objectives, columns=self._problem.get_objective_names()
        )

        plot_request = SimplePlotRequest(
            data=data, dimensions_data=dimensions_data, message=msg,
        )

        return plot_request

    def handle_classification_request(
        self, request: NimbusClassificationRequest
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        improve_inds = np.where(
            np.array(request.response["classifications"]) == "<"
        )[0]

        acceptable_inds = np.where(
            np.array(request.response["classifications"]) == "="
        )[0]

        free_inds = np.where(
            np.array(request.response["classifications"]) == "0"
        )[0]

        improve_until_inds = np.where(
            np.array(request.response["classifications"]) == "<="
        )[0]

        impaire_until_inds = np.where(
            np.array(request.response["classifications"]) == ">="
        )[0]

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
        if request.response["indices"]:
            return self.compute_intermediate_solutions(
                np.array(request.content["solutions"])[
                    request.response["indices"]
                ],
                int(request.response["number_of_desired_solutions"]),
            )

        return self.request_most_preferred_solution(
            np.array(request.content["solutions"]),
            np.array(request.content["objectives"]),
        )

    def handle_most_preferred_request(
        self, request: NimbusMostPreferredRequest
    ) -> Tuple[
        Union[NimbusClassificationRequest, NimbusStopRequest], SimplePlotRequest
    ]:
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
        request = NimbusStopRequest(
            self._current_solution, self._current_objectives
        )

        msg = "Final solution reached"
        plot_request = self.create_plot_request(
            np.atleast_2d(self._current_objectives), msg
        )

        return request, plot_request

    def request_most_preferred_solution(
        self, solutions: np.ndarray, objectives: np.ndarray
    ) -> Tuple[NimbusMostPreferredRequest, SimplePlotRequest]:
        # request most preferred solution
        request = NimbusMostPreferredRequest(list(solutions), list(objectives))

        msg = "Computed solutions"
        plot_request = self.create_plot_request(objectives, msg)

        return request, plot_request

    def compute_intermediate_solutions(
        self, solutions: np.ndarray, n_desired: int,
    ) -> Tuple[NimbusSaveRequest, SimplePlotRequest]:
        # vector between the two solutions
        between = solutions[0] - solutions[1]
        norm = np.linalg.norm(between)
        between_norm = between / norm

        # the plus 2 assumes we are interested only in n_desired points BETWEEN the
        # two supplied solutions
        step_size = norm / (2 + n_desired)

        intermediate_points = np.array(
            [
                solutions[1] + i * step_size * between_norm
                for i in range(1, n_desired + 1)
            ]
        )

        # project each of the intermediate solutions to the Pareto front
        intermediate_solutions = np.zeros(intermediate_points.shape)
        intermediate_objectives = np.zeros(
            (n_desired, self._problem.n_of_objectives)
        )
        asf = PointMethodASF(self._nadir, self._ideal)

        for i in range(n_desired):
            scalarizer = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf,
                scalarizer_args={
                    "reference_point": self._problem.evaluate(
                        intermediate_points[i]
                    ).objectives
                },
            )

            if self._problem.n_of_constraints > 0:
                cons = lambda x: self._problem.evaluate(x).constraints.squeeze()
            else:
                cons = None

            solver = ScalarMinimizer(
                scalarizer,
                self._problem.get_variable_bounds(),
                cons,
                method=self._scalar_method,
            )
            # TODO: fix me
            solver._use_scipy = True

            res = solver.minimize(self._current_solution)
            intermediate_solutions[i] = res["x"]
            intermediate_objectives[i] = self._problem.evaluate(
                res["x"]
            ).objectives

        # create appropiate requests
        save_request = NimbusSaveRequest(
            list(intermediate_solutions), list(intermediate_objectives)
        )

        msg = "Computed intermediate solutions"
        plot_request = self.create_plot_request(intermediate_objectives, msg)

        return save_request, plot_request

    def save_solutions_to_archive(
        self,
        objectives: np.ndarray,
        decision_variables: np.ndarray,
        indices: List[int],
    ) -> Tuple[NimbusIntermediateSolutionsRequest, None]:
        self._archive_objectives.extend(list(objectives[indices]))
        self._archive_solutions.extend(list(decision_variables[indices]))

        mask = np.ones(objectives.shape[0], dtype=bool)
        mask[indices] = False

        req_objectives = self._archive_objectives + list(objectives[mask])
        req_solutions = self._archive_solutions + list(decision_variables[mask])

        # create intermediate solutions request
        request = NimbusIntermediateSolutionsRequest(
            req_solutions, req_objectives
        )

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
        results = []

        # always computed
        asf_1 = MaxOfTwoASF(
            self._nadir, self._ideal, improve_inds, improve_until_inds
        )

        def cons_1(
            x: np.ndarray,
            f_current: np.ndarray = self._current_objectives,
            levels: np.ndarray = levels,
            improve_until_inds: np.ndarray = improve_until_inds,
            improve_inds: np.ndarray = improve_inds,
            impaire_until_inds: np.ndarray = impaire_until_inds,
        ):
            f = self._problem.evaluate(x).objectives.squeeze()

            res_1 = f_current[improve_inds] - f[improve_inds]
            res_2 = f_current[improve_until_inds] - f[improve_until_inds]
            res_3 = levels[impaire_until_inds] - f_current[impaire_until_inds]

            res = np.hstack((res_1, res_2, res_3))

            if self._problem.n_of_constraints > 0:
                res_prob = self._problem.evaluate(x).constraints.squeeze()

                return np.hstack((res_prob, res))

            else:
                return res

        scalarizer_1 = Scalarizer(
            lambda x: self._problem.evaluate(x).objectives,
            asf_1,
            scalarizer_args={"reference_point": levels},
        )

        solver_1 = ScalarMinimizer(
            scalarizer_1,
            self._problem.get_variable_bounds(),
            cons_1,
            method=self._scalar_method,
        )

        res_1 = solver_1.minimize(self._current_solution)
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
            if self._problem.n_of_constraints > 0:
                cons_2 = lambda x: self._problem.evaluate(
                    x
                ).constraints.squeeze()
            else:
                cons_2 = None

            scalarizer_2 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf_2,
                scalarizer_args={"reference_point": z_bar},
            )

            solver_2 = ScalarMinimizer(
                scalarizer_2,
                self._problem.get_variable_bounds(),
                cons_2,
                method=self._scalar_method,
            )

            res_2 = solver_2.minimize(self._current_solution)
            results.append(res_2)

        if number_of_solutions > 2:
            # asf 3
            asf_3 = PointMethodASF(self._nadir, self._ideal)

            scalarizer_3 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf_3,
                scalarizer_args={"reference_point": z_bar},
            )

            solver_3 = ScalarMinimizer(
                scalarizer_3,
                self._problem.get_variable_bounds(),
                cons_2,
                method=self._scalar_method,
            )

            res_3 = solver_3.minimize(self._current_solution)
            results.append(res_3)

        if number_of_solutions > 3:
            # asf 4
            asf_4 = AugmentedGuessASF(self._nadir, self._ideal, free_inds)

            scalarizer_4 = Scalarizer(
                lambda x: self._problem.evaluate(x).objectives,
                asf_4,
                scalarizer_args={"reference_point": z_bar},
            )

            solver_4 = ScalarMinimizer(
                scalarizer_4,
                self._problem.get_variable_bounds(),
                cons_2,
                method=self._scalar_method,
            )

            res_4 = solver_4.minimize(self._current_solution)
            results.append(res_4)

        # create the save request
        solutions = [res["x"] for res in results]
        objectives = [
            self._problem.evaluate(x).objectives.squeeze() for x in solutions
        ]

        save_request = NimbusSaveRequest(solutions, objectives)

        msg = "Computed new solutions."
        plot_request = self.create_plot_request(objectives, msg)

        return save_request, plot_request

    def update_current_solution(
        self, solutions: np.ndarray, objectives: np.ndarray, index: int
    ) -> None:
        self._current_solution = solutions[index].squeeze()
        self._current_objectives = objectives[index].squeeze()

        return None

    def step(
        self,
        request: Union[
            NimbusClassificationRequest,
            NimbusSaveRequest,
            NimbusIntermediateSolutionsRequest,
            NimbusMostPreferredRequest,
            NimbusStopRequest,
        ],
    ) -> Tuple[
        Union[
            NimbusClassificationRequest,
            NimbusSaveRequest,
            NimbusIntermediateSolutionsRequest,
        ],
        Union[SimplePlotRequest, None],
    ]:
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
                raise NimbusException(
                    f"Expected request type {type(NimbusSaveRequest)}, was {type(request)}."
                )

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
                raise NimbusException(
                    f"Expected request type {type(NimbusMostPreferredRequest)}, was {type(request)}."
                )

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
    from desdeo_problem.Problem import MOProblem
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem.Variable import variable_builder
    from desdeo_problem.Constraint import ScalarConstraint
    from desdeo_tools.scalarization.Scalarizer import Scalarizer
    from scipy.optimize import differential_evolution

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

    def c_1(x, f=None):
        x = x.squeeze()
        return (x[0] + x[1]) - 0.5

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
    c1 = ScalarConstraint("c1", 2, 5, evaluator=c_1)
    problem = MOProblem(
        variables=varsl, objectives=[f1, f2, f3, f4, f5], constraints=[c1]
    )

    method = NIMBUS(problem, scalar_method="scipy_de")
    reqs = method.request_classification()[0]

    response = {}
    response["classifications"] = ["<", "<=", "=", ">=", "0"]
    response["levels"] = [-6, -3, -5, 8, 0.349]
    response["number_of_solutions"] = 3
    reqs.response = response
    res_1 = method.step(reqs)[0]
    res_1.response = {"indices": [0, 1, 2]}

    res_2 = method.step(res_1)[0]
    response = {}
    response["indices"] = []
    response["number_of_desired_solutions"] = 0
    res_2.response = response

    res_3 = method.step(res_2)[0]
    response_pref = {}
    response_pref["index"] = 1
    response_pref["continue"] = True
    res_3.response = response_pref

    res_4 = method.step(res_3)
