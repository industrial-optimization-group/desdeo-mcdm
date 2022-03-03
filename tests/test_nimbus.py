from desdeo_mcdm.interactive import NIMBUS, NimbusStopRequest
from desdeo_problem.problem import ScalarObjective
from desdeo_problem.problem import variable_builder
from desdeo_problem.problem import MOProblem
import numpy as np
import numpy.testing as npt
import pytest


@pytest.fixture
def river_problem() -> MOProblem:
    # create the problem (river pollution)

    def f_1(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(
            x
        )  # This step is to guarantee that the function works when called with a single decision variable vector as well.
        return -4.07 - 2.27 * x[:, 0]

    def f_2(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return (
            -2.60
            - 0.03 * x[:, 0]
            - 0.02 * x[:, 1]
            - 0.01 / (1.39 - x[:, 0] ** 2)
            - 0.30 / (1.39 + x[:, 1] ** 2)
        )

    def f_3(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -8.21 + 0.71 / (1.09 - x[:, 0] ** 2)

    def f_4(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -0.96 - 0.96 / (1.09 - x[:, 1] ** 2)

    def f_5(x: np.ndarray) -> np.ndarray:
        return np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)

    f1 = ScalarObjective(name="f1", evaluator=f_1)
    f2 = ScalarObjective(name="f2", evaluator=f_2)
    f3 = ScalarObjective(name="f3", evaluator=f_3)
    f4 = ScalarObjective(name="f4", evaluator=f_4)
    f5 = ScalarObjective(name="f5", evaluator=f_5)
    # relax bounds a bit since MOProblem is so anal about them...
    varsl = variable_builder(
        ["x_1", "x_2"],
        initial_values=[0.5, 0.5],
        lower_bounds=[0.3 - 0.01, 0.3 - 0.01],
        upper_bounds=[1.0 + 0.1, 1.0 + 0.1],
    )

    ideal = np.array([-6.339, -2.864, -7.499, -11.626, 0])
    nadir = np.array([-4.751, -2.767, -0.321, -1.920, 0.349])
    problem = MOProblem(
        variables=varsl, objectives=[f1, f2, f3, f4, f5], ideal=ideal, nadir=nadir
    )

    return problem


@pytest.mark.nimbus
def test_init_w_start_point(river_problem):
    starting_point = np.array([-5, -2.78, -3, -10, 0.2], dtype=float)
    method = NIMBUS(
        river_problem, starting_point=starting_point, scalar_method="scipy_de"
    )
    request, _ = method.request_classification()

    # try to iterate
    response = {}
    response["classifications"] = ["<", "<=", "=", ">=", "0"]
    response["levels"] = [-6.25, -2.77, -5, -6, 0.2]
    response["number_of_solutions"] = 4
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = [0, 1]
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = [1, 3]
    response["number_of_desired_solutions"] = 6
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = []
    response["number_of_desired_solutions"] = 0
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = []
    response["number_of_desired_solutions"] = 0
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["index"] = 1
    response["continue"] = False
    request.response = response

    request, _ = method.iterate(request)

    assert isinstance(request, NimbusStopRequest)


@pytest.mark.nimbus
def test_init_w_no_start_point(river_problem):
    starting_point = None
    method = NIMBUS(
        river_problem, starting_point=starting_point, scalar_method="scipy_minimize"
    )
    request, _ = method.request_classification()

    # try to iterate
    response = {}
    response["classifications"] = ["<", "<=", "=", ">=", "0"]
    response["levels"] = [-6.25, -2.77, -5, -6, 0.2]
    response["number_of_solutions"] = 4
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = [0, 1]
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = [1, 3]
    response["number_of_desired_solutions"] = 6
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = []
    response["number_of_desired_solutions"] = 0
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["indices"] = []
    response["number_of_desired_solutions"] = 0
    request.response = response

    request, _ = method.iterate(request)

    response = {}
    response["index"] = 1
    response["continue"] = False
    request.response = response

    request, _ = method.iterate(request)

    assert isinstance(request, NimbusStopRequest)
