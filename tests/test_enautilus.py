import numpy as np
import numpy.testing as npt
import pytest
from desdeo_mcdm.interactive import ENautilus


@pytest.mark.enautilus
def test_simple_iterate():
    """Iterates the whole method through once."""
    f1 = np.linspace(1, 100, 50)
    f2 = f1[::-1] ** 2

    front = np.stack((f1, f2)).T
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)

    method = ENautilus((front), ideal, nadir)

    req = method.start()

    n_iterations = 11
    n_points = 4

    req.response = {
        "n_iterations": n_iterations,
        "n_points": n_points,
    }

    req = method.iterate(req)
    req.response = {"preferred_point_index": 0}

    while method._n_iterations_left > 1:
        req = method.iterate(req)
        req.response = {"preferred_point_index": 0}

    req = method.iterate(req)

    assert method._n_iterations_left == 0
