import numpy as np
import numpy.testing as npt
import pytest
from desdeo_mcdm.interactive import NautilusNavigator, NautilusNavigatorException
from desdeo_tools.scalarization import PointMethodASF


@pytest.fixture()
def pareto_front():
    # dummy, non-dominated discreet front
    pareto_front = np.array(
        [
            [-1.2, 0, 2.1, 2],
            [1.0, -0.99, 3.2, 2.2],
            [0.7, 2.2, 1.1, 1.9],
            [1.9, 2.1, 1.01, 0.5],
            [-0.4, -0.3, 10.5, 12.3],
        ]
    )
    return pareto_front


@pytest.fixture()
def decision_variables():
    decision_variables = np.array(
        [[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8], [9.9, 10.1],]
    )
    return decision_variables


@pytest.fixture()
def ideal(pareto_front):
    return np.min(pareto_front, axis=0)


@pytest.fixture()
def nadir(pareto_front):
    return np.max(pareto_front, axis=0)


@pytest.fixture()
def asf_problem():
    fun = NautilusNavigator.solve_nautilus_asf_problem
    return fun


@pytest.fixture()
def asf(ideal, nadir):
    asf = PointMethodASF(nadir, ideal)
    return asf


class TestNavigation:
    """Test the navigation method"""

    def test_decision_variables(self, pareto_front, ideal, nadir, decision_variables):
        """Test initialization with decision variables."""
        # correct
        method = NautilusNavigator(pareto_front, ideal, nadir, decision_variables)
        npt.assert_almost_equal(method._decision_variables, decision_variables)

        # wrong dimensinos
        bad_decision = np.array([[1.1, 2.2], [3.1, 9.2], [1.3, 3.2],])
        with pytest.raises(NautilusNavigatorException) as e:
            method_bad = NautilusNavigator(pareto_front, ideal, nadir, bad_decision)
            assert "The supplied decision variables must" in str(e)

    def test_navigation_end(self, pareto_front, ideal, nadir, decision_variables):
        """Test navigation and stopping at end of navigation"""
        method = NautilusNavigator(pareto_front, ideal, nadir, decision_variables)
        method._steps_remaining = 10
        request = method.start()

        while True:
            response = {
                "reference_point": np.array([0.7, 2.2, 1.1, 1.9]),
                "speed": 5,
                "go_to_previous": False,
                "stop": False,
                "user_bounds": [None, None, None, None],
            }

            request.response = response

            request = method.iterate(request)

            if request.content["steps_remaining"] == 1:
                break

        response = {
            "reference_point": np.array([0.75, 2.15, 1.15, 1.85]),
            "speed": 5,
            "go_to_previous": False,
            "stop": True,
            "user_bounds": [None, None, None, None],
        }

        request.response = response

        request = method.iterate(request)

        # The final solution should be the reference point
        final_objectives = request.content["objective_vectors"]
        final_variables = request.content["decision_vectors"]

        npt.assert_almost_equal(final_objectives, pareto_front[2])
        npt.assert_almost_equal(final_variables, decision_variables[2])

    def test_navigation_intermediate_stop(
        self, pareto_front, ideal, nadir, decision_variables
    ):
        """Test navigation and stopping at an intermediate iteration"""
        method = NautilusNavigator(pareto_front, ideal, nadir, decision_variables)
        method._steps_remaining = 10
        request = method.start()

        while True:
            response = {
                "reference_point": np.array([0.7, 2.2, 1.1, 1.9]),
                "speed": 5,
                "go_to_previous": False,
                "stop": False,
                "user_bounds": [None, None, None, None],
            }

            request.response = response

            request = method.iterate(request)

            if request.content["steps_remaining"] == 5:
                break

        response = {
            "reference_point": np.array([0.75, 2.15, 1.15, 1.85]),
            "speed": 5,
            "go_to_previous": False,
            "stop": True,
            "user_bounds": [None, None, None, None],
        }

        request.response = response

        request = method.iterate(request)

        # The solutions found
        final_objectives = request.content["objective_vectors"]
        final_variables = request.content["decision_vectors"]
        reachable = request.content["reachable_idx"]

        # check that the indices match
        for i, reachable_i in enumerate(reachable):
            npt.assert_almost_equal(pareto_front[reachable_i], final_objectives[i])
            npt.assert_almost_equal(decision_variables[reachable_i], final_variables[i])

    def test_navigation_end_novars(self, pareto_front, ideal, nadir):
        """Test navigation and stopping at end of navigation without variables"""
        method = NautilusNavigator(pareto_front, ideal, nadir)
        method._steps_remaining = 10
        request = method.start()

        while True:
            response = {
                "reference_point": np.array([0.7, 2.2, 1.1, 1.9]),
                "speed": 5,
                "go_to_previous": False,
                "stop": False,
                "user_bounds": [None, None, None, None],
            }

            request.response = response

            request = method.iterate(request)

            if request.content["steps_remaining"] == 1:
                break

        response = {
            "reference_point": np.array([0.75, 2.15, 1.15, 1.85]),
            "speed": 5,
            "go_to_previous": False,
            "stop": True,
            "user_bounds": [None, None, None, None],
        }

        request.response = response

        request = method.iterate(request)

        # The final solution should be the reference point
        final_objectives = request.content["objective_vectors"]
        final_variables = request.content["decision_vectors"]

        npt.assert_almost_equal(final_objectives, pareto_front[2])
        assert final_variables is None

    def test_navigation_intermediate_stop_novars(self, pareto_front, ideal, nadir):
        """Test navigation and stopping at an intermediate iteration without variables"""
        method = NautilusNavigator(pareto_front, ideal, nadir)
        method._steps_remaining = 10
        request = method.start()

        while True:
            response = {
                "reference_point": np.array([0.7, 2.2, 1.1, 1.9]),
                "speed": 5,
                "go_to_previous": False,
                "stop": False,
                "user_bounds": [None, None, None, None],
            }

            request.response = response

            request = method.iterate(request)

            if request.content["steps_remaining"] == 5:
                break

        response = {
            "reference_point": np.array([0.75, 2.15, 1.15, 1.85]),
            "speed": 5,
            "go_to_previous": False,
            "stop": True,
            "user_bounds": [None, None, None, None],
        }

        request.response = response

        request = method.iterate(request)

        # The solutions found
        final_objectives = request.content["objective_vectors"]
        final_variables = request.content["decision_vectors"]
        reachable = request.content["reachable_idx"]

        # check that the indices match
        for i, reachable_i in enumerate(reachable):
            npt.assert_almost_equal(pareto_front[reachable_i], final_objectives[i])

        assert final_variables is None


class TestRefPointProjection:
    def test_no_bounds(self, asf_problem, pareto_front, ideal, nadir, asf):
        """Test the projection to the Pareto front without specifying any bounds.
        """
        bounds = np.repeat(np.nan, ideal.size)
        ref_points = [
            [0.5, 1, 2, 3],
            [1.8, 2.0, 1.05, 0.33],
            [0.9, -0.88, 3.1, 2.1],
            [100, 100, 100, 100],
            [-100, -100, -100, -100],
            [0, 0, 0, 0],
        ]

        for ref_point in ref_points:
            proj_i = asf_problem(
                pareto_front,
                list(range(0, pareto_front.shape[0])),
                np.array(ref_point),
                ideal,
                nadir,
                bounds,
            )

            # The projection should be the point on the Pareto front with the shortest distance to the reference point
            # (metric dictated by use ASF)
            should_be = np.argmin(asf(pareto_front, ref_point))

            assert proj_i == should_be

    def test_w_subset_i(self, asf_problem, pareto_front, ideal, nadir, asf):
        """Test the projection to a subset of the Pareto front.
        """
        bounds = np.repeat(np.nan, ideal.size)
        subset = np.array([1, 3, 4], dtype=int)
        ref_points = [
            [0.5, 1, 2, 3],
            [1.8, 2.0, 1.05, 0.33],
            [0.9, -0.88, 3.1, 2.1],
            [100, 100, 100, 100],
            [-100, -100, -100, -100],
            [0, 0, 0, 0],
        ]
        pf_mask = np.repeat(False, pareto_front.shape[0])
        pf_mask[subset] = True
        filtered_pf = np.copy(pareto_front)
        filtered_pf[~pf_mask] = np.nan

        for ref_point in ref_points:
            proj_i = asf_problem(
                pareto_front, subset, np.array(ref_point), ideal, nadir, bounds
            )

            # The projection should be the point on the Pareto front with the shortest distance to the reference point
            # (metric dictated by use ASF)
            should_be = np.nanargmin(asf(filtered_pf, ref_point))
            print(should_be)

            assert proj_i == should_be

    def test_w_subset_i_and_bounds(self, asf_problem, pareto_front, ideal, nadir, asf):
        """Test the projection to a subset of the Pareto front.
        """
        bounds = np.array([np.nan, 1.9, np.nan, np.nan])
        subset = np.array([1, 3, 4], dtype=int)
        ref_points = [
            [0.5, 1, 2, 3],
            [1.8, 2.0, 1.05, 0.33],
            [0.9, -0.88, 3.1, 2.1],
            [100, 100, 100, 100],
            [-100, -100, -100, -100],
            [0, 0, 0, 0],
        ]
        pf_mask = np.repeat(False, pareto_front.shape[0])
        pf_mask[subset] = True
        filtered_pf = np.copy(pareto_front)
        filtered_pf[~pf_mask] = np.nan
        bound_mask = np.any(filtered_pf > bounds, axis=1)
        filtered_pf[bound_mask] = np.nan

        for ref_point in ref_points:
            proj_i = asf_problem(
                pareto_front, subset, np.array(ref_point), ideal, nadir, bounds
            )

            # The projection should be the point on the Pareto front with the shortest distance to the reference point
            # (metric dictated by use ASF)
            should_be = np.nanargmin(asf(filtered_pf, ref_point))
            print(should_be)

            assert proj_i == should_be
