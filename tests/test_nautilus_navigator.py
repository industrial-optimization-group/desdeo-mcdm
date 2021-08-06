import numpy as np
import pytest
from desdeo_mcdm.interactive import NautilusNavigator
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
def ideal(pareto_front):
    return np.min(pareto_front, axis=0)


@pytest.fixture()
def nadir(pareto_front):
    return np.max(pareto_front, axis=0)


@pytest.fixture()
def fun():
    fun = NautilusNavigator.solve_nautilus_asf_problem
    return fun


@pytest.fixture()
def asf(ideal, nadir):
    asf = PointMethodASF(nadir, ideal)
    return asf


class TestRefPointProjection:
    def test_no_bounds(self, fun, pareto_front, ideal, nadir, asf):
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
            proj_i = fun(pareto_front, list(range(0, pareto_front.shape[0])), np.array(ref_point), ideal, nadir, bounds)

            # The projection should be the point on the Pareto front with the shortest distance to the reference point
            # (metric dictated by use ASF)
            should_be = np.argmin(asf(pareto_front, ref_point))

            assert proj_i == should_be

    def test_w_subset_i(self, fun, pareto_front, ideal, nadir, asf):
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
            proj_i = fun(pareto_front, subset, np.array(ref_point), ideal, nadir, bounds)

            # The projection should be the point on the Pareto front with the shortest distance to the reference point
            # (metric dictated by use ASF)
            should_be = np.nanargmin(asf(filtered_pf, ref_point))
            print(should_be)

            assert proj_i == should_be

    def test_w_subset_i_and_bounds(self, fun, pareto_front, ideal, nadir, asf):
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
            proj_i = fun(pareto_front, subset, np.array(ref_point), ideal, nadir, bounds)

            # The projection should be the point on the Pareto front with the shortest distance to the reference point
            # (metric dictated by use ASF)
            should_be = np.nanargmin(asf(filtered_pf, ref_point))
            print(should_be)

            assert proj_i == should_be
