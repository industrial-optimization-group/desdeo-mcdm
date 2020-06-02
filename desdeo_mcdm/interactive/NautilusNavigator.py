from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer
from desdeo_tools.solver.ScalarSolver import DiscreteMinimizer


class NautilusNavigatorRequest(BaseRequest):
    """Request to handle interactions with NAUTILUS Navigator. See the
    NautilusNavigator class for further details.
    """

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
        allowed_speeds: [int],
        current_speed: int,
        navigation_point: np.ndarray,
    ):
        msg = (
            # TODO: Be more specific...
            "Please supply aspirations levels for each objective between "
            "the upper and lower bounds as `reference_point`. Specify a "
            "speed between 1-5 as `speed`. If going to a previous step is "
            "desired, please set `go_to_previous` to True, otherwise it should "
            "be False. "
            "Lastly, if stopping is desired, `stop` should be True, "
            "otherweise it should be set to False."
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
            "allowed_speeds": allowed_speeds,
            "current_speed": current_speed,
            "navigation_point": navigation_point,
        }

        super().__init__(
            "reference_point_preference", "required", content=content
        )

    @classmethod
    def init_with_method(cls, method):
        return cls(
            method._ideal,
            method._nadir,
            method._reachable_lb,
            method._reachable_ub,
            method._reachable_idx,
            method._step_number,
            method._steps_remaining,
            method._distance,
            method._allowed_speeds,
            method._current_speed,
            method._navigation_point,
        )

    def validator(self, response: Dict) -> None:
        if "reference_point" not in response:
            raise NautilusNavigatorException("'reference_point' entry missing.")

        if "speed" not in response:
            raise NautilusNavigatorException("'speed' entry missing.")

        if "go_to_previous" not in response:
            raise NautilusNavigatorException("'go_to_previous' entry missing.")

        if "stop" not in response:
            raise NautilusNavigatorException("'stop' entry missing.")

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

        speed = response["speed"]
        try:
            if int(speed) not in self._content["allowed_speeds"]:
                raise NautilusNavigatorException(f"Invalid speed: {speed}.")
        except Exception as e:
            raise NautilusNavigatorException(
                f"An exception rose when validating the given speed {speed}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
            )

        if not type(response["go_to_previous"]) == bool:
            raise (
                f"Non boolean value {response['go_to_previous']} "
                f"found for 'go_to_previous' when validating the response."
            )

        if not type(response["stop"]) == bool:
            raise (
                f"Non boolean value {response['stop']} "
                f"found for 'go_to_previous' when validating the response."
            )

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NautilusNavigatorException(Exception):
    """Raised when an exception related to NAUTILUS Navigator is encountered.
    """

    pass


class NautilusNavigator(InteractiveMethod):
    """Implementations of the NAUTILUS Navigator algorithm.

    """

    def __init__(
        self,
        pareto_front: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        objective_names: Optional[List[str]] = None,
        minimize: Optional[List[int]] = None,
    ):
        """

        Args:
            pareto_front (np.ndarray): A two dimensional numpy array
            representing a Pareto front with objective vectors on each of its
            rows.
            ideal (np.ndarray): The ideal objective vector of the problem
            being represented by the Pareto front.
            nadir (np.ndarray): The nadir objective vector of the problem
            being represented by the Pareto front.
            objective_names (Optional[List[str]], optional): Names of the
            objectives. List must match the number of columns in
            pareto_front. Defaults to 'f1', 'f2', 'f3', ...
            minimize (Optional[List[int]], optional): Multipliers for each
            objective. '-1' indicates maximization and '1' minimization.
            Defaults to all objective values being minimized.

        Raises:
            NautilusNavigatorException: One or more dimension mismatches are
            encountered among the supplies arguments.
        """
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

        if objective_names:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusNavigatorException(
                    "The supplied objective names must have a leangth equal to "
                    "the numbr of objectives."
                )
            self._objective_names = objective_names
        else:
            self._objective_names = [f"f{i+1}" for i in range(ideal.shape[0])]

        if minimize:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusNavigatorException(
                    "The minimize list must have "
                    "as many elements as there are objectives."
                )
            self._minimize = minimize
        else:
            self._minimize = [1 for _ in range(ideal.shape[0])]

        self._ideal = ideal
        self._nadir = nadir

        # in objective space!
        self._pareto_front = pareto_front

        # bounds of the rechable region
        self._reachable_ub = self._nadir
        self._reachable_lb = self._ideal

        # currently reachable solution as a list of indices of the Pareto front
        self._reachable_idx = list(range(0, self._pareto_front.shape[0]))

        # current iteration step number
        self._step_number = 1

        # iterations left
        self._steps_remaining = 100

        # L2 distance to the supplied Pareto front
        self._distance = 0

        self._allowed_speeds = [1, 2, 3, 4, 5]
        self._current_speed = None

        self._reference_point = None
        self._navigation_point = self._nadir
        self._projection_index = None

    def start(self) -> NautilusNavigatorRequest:
        """Returns the first Request object to begin iterating.

        Returns:
            NautilusNavigatorRequest: The Request.
        """
        return NautilusNavigatorRequest.init_with_method(self)

    def iterate(
        self, request: NautilusNavigatorRequest
    ) -> NautilusNavigatorRequest:
        """Perform the next logical step based on the response in the
        Request.
        """
        reqs = self.handle_request(request)
        return reqs

    def handle_request(
        self, request: NautilusNavigatorRequest
    ) -> NautilusNavigatorRequest:
        """Handle the Request and its contents.

        Args:
            request (NautilusNavigatorRequest): A Request with a defined response.

        Returns:
            NautilusNavigatorRequest: Some of the contents of the response are invalid.
        """
        preference_point = request.response["reference_point"]
        speed = request.response["speed"]
        go_to_previous = request.response["go_to_previous"]
        stop = request.response["stop"]

        if go_to_previous:
            step_number = request.content["step_number"]
            nav_point = request.content["navigation_point"]
            lower_bounds = request.content["reachable_lb"]
            upper_bounds = request.content["reachable_ub"]
            reachable_idx = request.content["reachable_idx"]
            distance = request.content["distance"]
            steps_remaining = request.content["steps_remaining"]

            return self.update(
                preference_point,
                speed,
                go_to_previous,
                stop,
                step_number,
                nav_point,
                lower_bounds,
                upper_bounds,
                reachable_idx,
                distance,
                steps_remaining,
            )

        else:
            return self.update(preference_point, speed, go_to_previous, stop,)

    def update(
        self,
        ref_point: np.ndarray,
        speed: int,
        go_to_previous: bool,
        stop: bool,
        step_number: Optional[int] = None,
        nav_point: Optional[np.ndarray] = None,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None,
        reachable_idx: Optional[List[int]] = None,
        distance: Optional[float] = None,
        steps_remaining: Optional[int] = None,
    ) -> Optional[NautilusNavigatorRequest]:
        """Update the inernal state of self.

        Args:
            ref_point (np.ndarray): A reference point given by a decision maker.
            speed (int): An integer value between 1-5 indicating the navigation speed.
            go_to_previous (bool): If True, the parameters indicate the state
            of a previous state, and the request is handled accordingly.
            stop (bool): If the navigation should stop. If True, self.update return None.
            step_number (Optional[int], optional): Current step number, or
            previous step number if go_to_previous is True. Defaults to None.
            nav_point (Optional[np.ndarray], optional): The current
            navigation point. Relevant if go_to_previous is True. Defaults to
            None.
            lower_bounds (Optional[np.ndarray], optional): Lower bounds of
            the reachable objective vector valus. Relevant if go_to_previous
            is True. Defaults to None.
            upper_bounds (Optional[np.ndarray], optional): Upper bounds of
            the reachable objective vector valus. Relevant if go_to_previous
            is True. Defaults to None.
            reachable_idx (Optional[List[int]], optional): Indices of the
            reachable Pareto optimal solutions. Relevant if go_to_previous is
            True. Defaults to None.
            distance (Optional[float], optional): Distance to the Pareto
            optimal front. Relevant if go_to_previous is True. Defaults to
            None.
            steps_remaining (Optional[int], optional): Remaining steps in the
            navigation. Relevant if go_to_previous is True. Defaults to None.

        Returns:
            NautilusNavigatorRequest: Some of the given parameters are erraneous.
        """
        # go to a previous state
        if go_to_previous:
            self._step_number = step_number
            self._navigation_point = nav_point
            self._reachable_lb = lower_bounds
            self._reachable_ub = upper_bounds
            self._reachable_idx = reachable_idx
            self._distance = distance
            self._steps_remaining = steps_remaining
            return NautilusNavigatorRequest.init_with_method(self)

        # compute a new navigation point closer to the Pareto front and the
        # bounds of the reachable Pareto optimal region.
        elif self._step_number == 1 or not np.allclose(
            ref_point, self._reference_point
        ):
            if self._step_number == 1:
                self._current_speed = speed

            proj_i = self.solve_nautilus_asf_problem(
                self._pareto_front,
                self._reachable_idx,
                ref_point,
                self._ideal,
                self._nadir,
            )

            self._reference_point = ref_point
            self._projection_index = proj_i

        elif stop:
            # new reference point given, also update speed
            self._reference_point = ref_point
            self._current_speed = speed

        new_nav = self.calculate_navigation_point(
            self._pareto_front[self._projection_index],
            self._navigation_point,
            self._steps_remaining,
        )

        self._navigation_point = new_nav

        new_lb, new_ub = self.calculate_bounds(
            self._pareto_front[self._reachable_idx], self._navigation_point,
        )

        self._reachable_lb = new_lb
        self._reachable_ub = new_ub

        new_dist = self.calculate_distance(
            self._navigation_point,
            self._pareto_front[self._projection_index],
            self._nadir,
        )

        self._distance = new_dist

        new_reachable = self.calculate_reachable_point_indices(
            self._pareto_front, self._reachable_lb, self._reachable_ub,
        )

        self._reachable_idx = new_reachable

        # If stop, do not update steps
        if self._steps_remaining == 1:
            # stop
            return NautilusNavigatorRequest.init_with_method(self)

        self._step_number += 1
        self._steps_remaining -= 1

        return NautilusNavigatorRequest.init_with_method(self)

    def calculate_reachable_point_indices(
        self,
        pareto_front: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> List[int]:
        """Calculate the indices of the reachable Pareto optimal solutions
        based on lower and upper bounds.

        Returns:
            List[int]: List of the indices of the reachable solutions.
        """
        low_idx = np.all(pareto_front >= lower_bounds, axis=1)
        up_idx = np.all(pareto_front <= upper_bounds, axis=1)

        reachable_idx = np.argwhere(low_idx & up_idx).squeeze()

        return reachable_idx

    def solve_nautilus_asf_problem(
        self,
        pareto_f: np.ndarray,
        subset_indices: [int],
        ref_point: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
    ) -> int:
        """Forms and solves the achievement scalarizing function to find the
        closesto point on the Pareto optimal front to the given reference
        point.

        Args:
            pareto_f (np.ndarray): The whole Pareto optimal front.
            subset_indices ([type]): Indices of the currently reachable solutions.
            ref_point (np.ndarray): The reference point indicating a decision
            maker's preference.
            ideal (np.ndarray): Ideal point.
            nadir (np.ndarray): Nadir point.

        Returns:
            int: Index of the closest point according the minimized value of the ASF.
        """
        asf = PointMethodASF(nadir, ideal)
        scalarizer = DiscreteScalarizer(asf, {"reference_point": ref_point})
        solver = DiscreteMinimizer(scalarizer)

        tmp = np.copy(pareto_f)
        mask = np.zeros(tmp.shape[0], dtype=bool)
        mask[subset_indices] = True
        tmp[~mask] = np.nan

        res = solver.minimize(tmp)

        return res

    def calculate_navigation_point(
        self,
        projection: np.ndarray,
        nav_point: np.ndarray,
        steps_remaining: int,
    ) -> np.ndarray:
        """Calculate a new navigation point based on the projection of the
        preference point to the Pareto optimal front.

        Args:
            projection (np.ndarray): The point on the Pareto optimal front
            closest to the preference point given by a decision maker.
            nav_point (np.ndarray): The previous navigation point.
            steps_remaining (int): How many steps are remaining in the navigation.

        Returns:
            np.ndarray: The new navigation point.
        """
        new_nav_point = (
            (steps_remaining - 1) / steps_remaining
        ) * nav_point + (1 / steps_remaining) * projection
        return new_nav_point

    def calculate_bounds(
        self, pareto_front: np.ndarray, nav_point: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the new bounds of the reachable points on the Pareto
        optimal front from a navigation point.

        Args:
            pareto_front (np.ndarray): The Pareto optimal front.
            nav_point (np.ndarray): The current navigation point.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The lower and upper bounds.
        """
        _pareto_front = np.atleast_2d(pareto_front)
        new_lower_bounds = np.zeros(_pareto_front.shape[1])
        new_upper_bounds = np.zeros(_pareto_front.shape[1])

        # TODO: vectorize this loop
        for r in range(_pareto_front.shape[1]):
            mask = np.zeros(_pareto_front.shape[1], dtype=bool)
            mask[r] = True

            subject_to = _pareto_front[:, ~mask].reshape(
                (_pareto_front.shape[0], _pareto_front.shape[1] - 1)
            )

            con_mask = np.all(subject_to <= nav_point[~mask], axis=1)

            min_val = np.min(_pareto_front[con_mask, mask])
            max_val = np.max(_pareto_front[con_mask, mask])

            new_lower_bounds[r] = min_val
            new_upper_bounds[r] = max_val

        return new_lower_bounds, new_upper_bounds

    def calculate_distance(
        self, nav_point: np.ndarray, projection: np.ndarray, nadir: np.ndarray
    ) -> float:
        """Calculate the distance to the Pareto optimal front from a
        navigation point. The distance is calculated to the supplied
        projection which is assumed to lay on the front.

        Args:
            nav_point (np.ndarray): The navigation point.
            projection (np.ndarray): The point of the Pareto optimal front the distance is calculated to.
            nadir (np.ndarray): The nadir point of the Pareto optimal set.

        Returns:
            float: The distance.
        """
        nom = np.linalg.norm(nav_point - nadir)
        denom = np.linalg.norm(projection - nadir)
        dist = (nom / denom) * 100

        return dist


if __name__ == "__main__":
    # front = np.array([[1, 2, 3], [2, 3, 4], [2, 2, 3], [3, 2, 1]], dtype=float)
    # ideal = np.zeros(3)
    # nadir = np.ones(3) * 5
    f1 = np.linspace(1, 100, 50)
    f2 = f1[::-1] ** 2

    front = np.stack((f1, f2)).T
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)

    method = NautilusNavigator((front), ideal, nadir)

    req = method.start()
    print(req.content["reachable_lb"])
    print(req.content["navigation_point"])
    print(req.content["reachable_ub"])

    response = {
        "reference_point": np.array([50, 6000]),
        "speed": 5,
        "go_to_previous": False,
        "stop": False,
    }
    req.response = response
    req = method.iterate(req)
    req.response = response

    req1 = req

    import time

    while req.content["steps_remaining"] > 1:
        time.sleep(1 / req.content["current_speed"])
        req = method.iterate(req)
        req.response = response
        print(req.content["steps_remaining"])
        print(req.content["reachable_lb"])
        print(req.content["navigation_point"])
        print(req.content["reachable_ub"])

    req1.response["go_to_previous"] = True
    req = method.iterate(req1)
    req.response = response
    req.response["go_to_previous"] = False

    while req.content["steps_remaining"] > 1:
        time.sleep(1 / req.content["current_speed"])
        req = method.iterate(req)
        req.response = response
        print(req.content["steps_remaining"])
        print(req.content["reachable_lb"])
        print(req.content["navigation_point"])
        print(req.content["reachable_ub"])
    print(req)
