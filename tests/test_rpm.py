import pytest
import numpy as np
import numpy.testing as npt
from desdeo_problem.testproblems import river_pollution_problem
from desdeo_mcdm.interactive import ReferencePointMethod

def test_solve():
    problem = river_pollution_problem()
    problem.ideal = np.array([-6.34, -3.44, -7.5, 0, 0])
    problem.nadir = np.array([-4.75, -2.85, -0.32, 9.70, 0.35])

    method = ReferencePointMethod(problem, problem.ideal, problem.nadir)

    request = method.start()

    request.response = {"reference_point": np.array([-6.34, -3.44, -7.5, 0, 0])}
    request = method.iterate(request)

    solution_1 = request.content["current_solution"]
    additionals_1 = request.content["additional_solutions"]

    request.response = {"reference_point": np.array([-4.75, -2.85, -0.32, 9.70, 0.35]), "satisfied": False}
    request = method.iterate(request)

    solution_2 = request.content["current_solution"]
    additionals_2 = request.content["additional_solutions"]

    # make sure we get different solution for the different reference points
    with npt.assert_raises(AssertionError):
        npt.assert_almost_equal(solution_1, solution_2)

    with npt.assert_raises(AssertionError):
        npt.assert_almost_equal(additionals_1, additionals_2)

    request.response = {"satisfied": True, "solution_index": 3}
    request = method.iterate(request)

    npt.assert_almost_equal(request.content["objective_vector"], additionals_2[2])
