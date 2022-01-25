import numpy as np
import numpy.testing as npt
import pytest
from desdeo_mcdm.interactive import ENautilus, ENautilusException


@pytest.fixture
def simple_data():
    f1 = np.linspace(1, 100, 50)
    f2 = f1[::-1] ** 2

    front = np.stack((f1, f2)).T
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)

    return front, ideal, nadir


@pytest.mark.enautilus
def test_simple_iterate(simple_data):
    """Iterates the whole method through once."""
    front, ideal, nadir = simple_data

    method = ENautilus((front), ideal, nadir)

    req = method.start()

    n_iterations = 11
    n_points = 4

    req.response = {
        "n_iterations": n_iterations,
        "n_points": n_points,
    }

    req = method.iterate(req)
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": False,
    }

    while method._n_iterations_left > 1:
        req = method.iterate(req)
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": False,
        }

    req = method.iterate(req)

    assert method._n_iterations_left == 0


@pytest.mark.enautilus
def test_step_back_response(simple_data):
    """Tests response for stepping back"""
    front, ideal, nadir = simple_data

    method = ENautilus((front), ideal, nadir)

    req = method.start()

    n_iterations = 8
    n_points = 3

    req.response = {
        "n_iterations": n_iterations,
        "n_points": n_points,
    }

    req = method.iterate(req)
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": False,
    }

    for _ in range(3):
        req = method.iterate(req)
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": False,
        }

    # save solution and bounds
    prev_solutions = req.content["points"]
    prev_l_bounds = req.content["lower_bounds"]
    prev_u_bounds = req.content["upper_bounds"]
    iter_left = req.content["n_iterations_left"]

    req = method.iterate(req)

    # prev_solutions missing
    with pytest.raises(ENautilusException) as e:
        req.response = {
            "preferred_point_index": 0,
            "step_back": True,
            "change_remaining": False,
        }

    assert "prev_solutions" in str(e)

    # previous lower bound missing
    with pytest.raises(ENautilusException) as e:
        req.response = {
            "preferred_point_index": 0,
            "step_back": True,
            "change_remaining": False,
            "prev_solutions": prev_solutions,
        }

    assert "prev_lower_bounds" in str(e)

    # previous upper bound missing
    with pytest.raises(ENautilusException) as e:
        req.response = {
            "preferred_point_index": 0,
            "step_back": True,
            "change_remaining": False,
            "prev_solutions": prev_solutions,
            "prev_lower_bounds": prev_l_bounds,
        }

    assert "prev_upper_bounds" in str(e)

    # iterations_left missing
    with pytest.raises(ENautilusException) as e:
        req.response = {
            "preferred_point_index": 0,
            "step_back": True,
            "change_remaining": False,
            "prev_solutions": prev_solutions,
            "prev_lower_bounds": prev_l_bounds,
            "prev_upper_bounds": prev_u_bounds,
        }

    assert "iterations_left" in str(e)

    # correct response
    req.response = {
        "preferred_point_index": 0,
        "step_back": True,
        "change_remaining": False,
        "prev_solutions": prev_solutions,
        "prev_lower_bounds": prev_l_bounds,
        "prev_upper_bounds": prev_u_bounds,
        "iterations_left": iter_left,
    }


@pytest.mark.enautilus
def test_change_remaining_response(simple_data):
    """Tests the response for chaging remaining iterations"""
    front, ideal, nadir = simple_data

    method = ENautilus((front), ideal, nadir)

    req = method.start()

    n_iterations = 15
    n_points = 4

    req.response = {
        "n_iterations": n_iterations,
        "n_points": n_points,
    }

    req = method.iterate(req)
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": False,
    }

    for _ in range(3):
        req = method.iterate(req)
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": False,
        }

    req = method.iterate(req)

    # iterations_left missing
    with pytest.raises(ENautilusException) as e:
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": True,
        }

    assert "iterations_left" in str(e)

    # correct
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": True,
        "iterations_left": 20,
    }


@pytest.mark.enautilus
def test_change_remaining_iterate(simple_data):
    """Tests iterations of stepping back"""
    front, ideal, nadir = simple_data

    method = ENautilus((front), ideal, nadir)

    req = method.start()

    n_iterations = 15
    n_points = 4

    req.response = {
        "n_iterations": n_iterations,
        "n_points": n_points,
    }

    req = method.iterate(req)
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": False,
    }

    for _ in range(3):
        req = method.iterate(req)
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": False,
        }

    # increment iterations notably
    iterations_increment = 20
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": True,
        "iterations_left": iterations_increment,
    }

    req = method.iterate(req)

    assert method._n_iterations_left == iterations_increment
    assert req.content["n_iterations_left"] == iterations_increment

    # iterate for n times
    n = 10
    for _ in range(n):
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": False,
        }
        req = method.iterate(req)

    assert method._n_iterations_left == iterations_increment - n

    # decrement iterations left
    iterations_decrement = 5

    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": True,
        "iterations_left": iterations_decrement,
    }

    req = method.iterate(req)

    assert method._n_iterations_left == iterations_decrement
    assert req.content["n_iterations_left"] == iterations_decrement


@pytest.mark.enautilus
def test_step_back_once(simple_data):
    """Tests stepping back once"""
    front, ideal, nadir = simple_data

    method = ENautilus((front), ideal, nadir)

    req = method.start()

    n_iterations = 8
    n_points = 3

    req.response = {
        "n_iterations": n_iterations,
        "n_points": n_points,
    }

    req = method.iterate(req)
    req.response = {
        "preferred_point_index": 0,
        "step_back": False,
        "change_remaining": False,
    }

    for _ in range(3):
        req = method.iterate(req)
        req.response = {
            "preferred_point_index": 0,
            "step_back": False,
            "change_remaining": False,
        }

    # save solution and bounds
    prev_solutions = req.content["points"]
    prev_l_bounds = req.content["lower_bounds"]
    prev_u_bounds = req.content["upper_bounds"]
    iter_left = req.content["n_iterations_left"]

    req = method.iterate(req)

    # we should have new solutions
    for i in range(n_points):
        assert np.not_equal(req.content["points"][i], prev_solutions[i]).all()
        assert np.not_equal(req.content["lower_bounds"][i], prev_l_bounds[i]).all()
    # upper bound does not change in this case

    # go back to the previous iteration
    req.response = {
        "preferred_point_index": 0,
        "step_back": True,
        "change_remaining": False,
        "prev_solutions": prev_solutions,
        "prev_lower_bounds": prev_l_bounds,
        "prev_upper_bounds": prev_u_bounds,
        "iterations_left": iter_left,
    }
    req = method.iterate(req)

    # we should have the same values as previously
    npt.assert_almost_equal(req.content["points"], prev_solutions)
    npt.assert_almost_equal(req.content["lower_bounds"], prev_l_bounds)

