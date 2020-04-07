"""Implementation of the NAUTILUS Navigator algorithm for solving
multiobjective optimization problems.

"""
import numpy as np

from typing import Tuple, List, Optional, Callable

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod


class NautilusNavigatorException(Exception):
    pass


class NautilusNavigator(InteractiveMethod):
    def __init__(
        self, pareto_front: np.ndarray, ideal: np.ndarray, nadir: np.ndarray
    ):

        if not ideal.shape == nadir.shape:
            raise NautilusNavigatorException(
                "The dimensions of the ideal and nadir point do not match."
            )

        if not ideal.shape[0] == pareto_front.shape[1]:
            raise NautilusNavigatorException(
                "The Pareto front must consist of objective vectors with the "
                "same number of objectives as defined in the ideal and nadir "
                "points."
            )
        self._ideal = ideal
        self._nadir = nadir

        self._pareto_front = pareto_front


if __name__ == "__main__":
    front = np.array([[1, 2, 3], [2, 3, 4], [2, 2, 3], [3, 2, 1]])
    ideal = np.zeros(3)
    nadir = np.ones(3) * 10

    method = NautilusNavigator(front, ideal, nadir)
