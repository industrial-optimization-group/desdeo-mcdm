"""Implementation of the NAUTILUS Navigator algorithm for solving
multiobjective optimization problems.

"""
import numpy as np

from typing import Tuple, List, Optional, Callable, Dict

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest


class NautilusNavigatorRequest(BaseRequest):
    def __init__(
        self,
        ideal: np.ndarray,
        nadir: np.ndarray,
        reachable_lb: np.ndarray,
        reachable_ub: np.ndarray,
        reachable_idx: List[int],
        step_number: int,
        steps_remaining: int,
        distance: float,
    ):
        msg = (
            "Please supply aspirations levels for each objective between "
            "the ideal and nadir values."
        )
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
            "reachable_lb": reachable_lb,
            "reachable_ub": reachable_ub,
            "reachable_idx": reachable_idx,
            "step_number": step_number,
            "steps_remaining": steps_remaining,
            "distance": distance,
            "reference_point": None,
        }

        super().__init__(
            "reference_point_preference", "required", content=content
        )

    @classmethod
    def init_with_method(cls, method):
        return cls(
            method._ideal,
            method._nadir,
            method.reachable_lb,
            method.reachable_ub,
            method.reachable_idx,
            method._step_number,
            method._steps_remaining,
            method._distance,
        )

    def validator(self, response: Dict) -> None:
        if "reference_point" not in response:
            raise NautilusNavigatorException("'reference_point' entry missing.")

        ref_point = response["reference_point"]
        try:
            if np.any(ref_point < self._content["ideal"]) or np.any(
                ref_point > self._content["nadir"]
            ):
                raise NautilusNavigatorException(
                    f"The given reference point {ref_point} "
                    "must be between the ranges imposed by the ideal and nadir points."
                )
        except Exception as e:
            raise NautilusNavigatorException(
                f"An exception rose when validating the given reference point {ref_point}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
            )

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NautilusNavigatorException(Exception):
    pass


class NautilusNavigator(InteractiveMethod):
    def __init__(
        self, pareto_front: np.ndarray, ideal: np.ndarray, nadir: np.ndarray
    ):
        if not pareto_front.ndim == 2:
            raise NautilusNavigatorException(
                "The supplied Pareto front should be a two dimensional array. Found "
                f" number of dimensions {pareto_front.ndim}."
            )

        if not ideal.shape[0] == pareto_front.shape[1]:
            raise NautilusNavigatorException(
                "The Pareto front must consist of objective vectors with the "
                "same number of objectives as defined in the ideal and nadir "
                "points."
            )

        if not ideal.shape == nadir.shape:
            raise NautilusNavigatorException(
                "The dimensions of the ideal and nadir point do not match."
            )

        self._ideal = ideal
        self._nadir = nadir

        # in objective space!
        self._pareto_front = pareto_front

        # bounds of the rechable region
        self._reachable_up = self._nadir
        self._reachable_lb = self._ideal

        # currently reachable solution as a list of indices of the Pareto front
        self._reachable_idx = list(range(0, self._pareto_front.shape[0]))

        # current iteration step number
        self._step_number = 1

        # iterations left
        self._steps_remaining = 100

        # L2 distance to the supplied Pareto front
        self._distance = 0

    def start(self):
        pass

    def iterate(self):
        pass


if __name__ == "__main__":
    front = np.array([[1, 2, 3], [2, 3, 4], [2, 2, 3], [3, 2, 1]])
    ideal = np.zeros(3)
    nadir = np.ones(3) * 10

    method = NautilusNavigator((front), ideal, nadir)
