from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from desdeo_tools.scalarization.ASF import PointMethodASF
from desdeo_tools.scalarization.Scalarizer import DiscreteScalarizer
from desdeo_tools.solver.ScalarSolver import DiscreteMinimizer


class NautilusNavigatorStopRequest(BaseRequest):
    """Request to stop navigation and return the solution found (or the
    currently reachable solutions, if stopped before the navigation ends.
    """

    def __init__(
        self,
        reachable_idx: List[int],
        pareto_front: np.ndarray,
        decision_variables: Optional[np.ndarray] = None,
    ):
        message = (
            "These are the current objectives vectors (and decision vectors, if supplied)"
            "reachable from the current navigation point."
        )

        if decision_variables is not None:
            reachable_decision = decision_variables[reachable_idx]
        else:
            reachable_decision = None

        content = {
            "message": message,
            "objective_vectors": pareto_front[reachable_idx],
            "decision_vectors": reachable_decision,
            "reachable_idx": reachable_idx,
        }

        super().__init__("classification_preference", "no_interaction", content=content)


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
        user_bounds: List[float],
        reachable_idx: List[int],
        step_number: int,
        steps_remaining: int,
        distance: float,
        allowed_speeds: List[int],
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
            "Bounds for one or more objectives may also be specified as 'user_bounds'; when navigating,"
            "the value of the objectives present in the navigation points will not exceed the values"
            "specified in 'user_bounds'."
            "Lastly, if stopping is desired, `stop` should be True, "
            "otherwise it should be set to False."
        )
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
            "reachable_lb": reachable_lb,
            "reachable_ub": reachable_ub,
            "user_bounds": user_bounds,
            "reachable_idx": reachable_idx,
            "step_number": step_number,
            "steps_remaining": steps_remaining,
            "distance": distance,
            "allowed_speeds": allowed_speeds,
            "current_speed": current_speed,
            "navigation_point": navigation_point,
        }

        super().__init__("reference_point_preference", "required", content=content)

    @classmethod
    def init_with_method(cls, method):
        return cls(
            method._ideal,
            method._nadir,
            method._reachable_lb,
            method._reachable_ub,
            method._user_bounds,
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

        if "user_bounds" not in response:
            raise NautilusNavigatorException("'user_bounds' entry missing")

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

        # This ensures that Nones are converted to np.nan
        user_bounds = np.array(response["user_bounds"], dtype=float)
        try:
            if len(user_bounds) != self._content["ideal"].size:
                raise NautilusNavigatorException(
                    f"The given user bounds '{user_bounds}' has mismatching dimensions compared "
                    f"to the ideal point '{self._content['ideal']}'."
                )
            """
            if np.any(user_bounds < self._content["reachable_lb"]) or np.any(
                user_bounds > self._content["reachable_ub"]
            ):
                raise NautilusNavigatorException(
                    f"The given user bounds '{user_bounds}' has one or more elements that are outside the region "
                    f"bounded by the reachable upper and lower bounds of the current navigation points. "
                    f"Current lower bounds: {self._content['reachable_lb']} "
                    f"Current upper bounds: {self._content['reachable_ub']}."
                )
            """
        except Exception as e:
            raise NautilusNavigatorException(
                f"An exception rose when validating the given user bounds "
                f"{user_bounds}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
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

    Args:
        pareto_front (np.ndarray): A two dimensional numpy array
            representing a Pareto front with objective vectors on each of its
            rows.
        ideal (np.ndarray): The ideal objective vector of the problem
            being represented by the Pareto front.
        nadir (np.ndarray): The nadir objective vector of the problem
            being represented by the Pareto front.
        decision_variables (Optional[np.ndarray]): Two dimensinoal numpy 
            array of decision variables
            that can be optionally supplied. The i'th vector in 
            decision_variables should result in the i'th objective
            vector in pareto_front. Defaults to None.

    Raises:
        NautilusNavigatorException: One or more dimension mismatches are
        encountered among the supplies arguments.
    """

    def __init__(
        self,
        pareto_front: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        decision_variables: Optional[np.ndarray] = None,
    ):
        if not pareto_front.ndim == 2:
            raise NautilusNavigatorException(
                "The supplied Pareto front should be a two dimensional array. Found "
                f" number of dimensions {pareto_front.ndim}."
            )

        if decision_variables is not None:
            # check correct dimensino of decision_variables, if supplied.
            if decision_variables.shape[0] != pareto_front.shape[0]:
                raise NautilusNavigatorException(
                    "The supplied decision variables must be as many as the objective "
                    "vectors in the supplied Pareto front. Dimension of variables "
                    f"{decision_variables.shape[0]}; dimension of Pareto front "
                    f"{pareto_front.shape[0]}."
                )

        self._decision_variables = decision_variables

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

        # bounds of the reachable region
        self._reachable_ub = self._nadir
        self._reachable_lb = self._ideal

        # user given bounds, defaults to none
        self._user_bounds = np.repeat(np.nan, self._ideal.size)

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

    def iterate(self, request: NautilusNavigatorRequest) -> NautilusNavigatorRequest:
        """Perform the next logical step based on the response in the
        Request.
        """
        reqs = self.handle_request(request)
        return reqs

    def handle_request(
        self, request: NautilusNavigatorRequest
    ) -> Union[NautilusNavigatorRequest, NautilusNavigatorStopRequest]:
        """Handle the Request and its contents.

        Args:
            request (NautilusNavigatorRequest): A Request with a defined response.

        Returns:
            NautilusNavigatorRequest: Some of the contents of the response are invalid.
        """
        preference_point = request.response["reference_point"]
        speed = request.response["speed"]
        go_to_previous = request.response["go_to_previous"]
        # ensure Nones are converted to NaN
        user_bounds = np.array(request.response["user_bounds"], dtype=float)
        stop = request.response["stop"]
        reachable_idx = request.content["reachable_idx"]

        if stop:
            return NautilusNavigatorStopRequest(
                reachable_idx, self._pareto_front, self._decision_variables
            )

        if go_to_previous:
            step_number = request.content["step_number"]
            nav_point = request.content["navigation_point"]
            lower_bounds = request.content["reachable_lb"]
            upper_bounds = request.content["reachable_ub"]
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
                user_bounds,
                reachable_idx,
                distance,
                steps_remaining,
            )

        else:
            return self.update(
                preference_point, speed, go_to_previous, stop, user_bounds=user_bounds
            )

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
        user_bounds: Optional[np.ndarray] = None,
        reachable_idx: Optional[List[int]] = None,
        distance: Optional[float] = None,
        steps_remaining: Optional[int] = None,
    ) -> Optional[NautilusNavigatorRequest]:
        """Update the internal state of self.

        Args:
            ref_point (np.ndarray): A reference point given by a decision maker.
            speed (int): An integer value between 1-5 indicating the navigation speed.
            go_to_previous (bool): If True, the parameters indicate the state
                of a previous state, and the request is handled accordingly.
            stop (bool): If the navigation should stop. If True, returns a request with self's current state.
            step_number (Optional[int], optional): Current step number, or
                previous step number if go_to_previous is True. Defaults to None.
            nav_point (Optional[np.ndarray], optional): The current
                navigation point. Relevant if go_to_previous is True. Defaults to
                None.
            lower_bounds (Optional[np.ndarray], optional): Lower bounds of
                the reachable objective vector values. Relevant if go_to_previous
                is True. Defaults to None.
            upper_bounds (Optional[np.ndarray], optional): Upper bounds of
                the reachable objective vector values. Relevant if go_to_previous
                is True. Defaults to None.
            user_bounds (Optional[np.ndarray], optional): The user given bounds for each objective.
                The reachable lower limit with attempt to not exceed the given bounds for each
                objective value.
            reachable_idx (Optional[List[int]], optional): Indices of the
                reachable Pareto optimal solutions. Relevant if go_to_previous is
                True. Defaults to None.
            distance (Optional[float], optional): Distance to the Pareto
                optimal front. Relevant if go_to_previous is True. Defaults to
                None.
            steps_remaining (Optional[int], optional): Remaining steps in the
                navigation. Relevant if go_to_previous is True. Defaults to None.

        Returns:
            NautilusNavigatorRequest: Some of the given parameters are erroneous.
        """

        # if stop navigation, return request with current state
        if stop:
            return NautilusNavigatorRequest.init_with_method(self)

        # go to a previous state
        elif go_to_previous:
            self._step_number = step_number
            self._navigation_point = nav_point
            self._reachable_lb = lower_bounds
            self._reachable_ub = upper_bounds
            self._user_bounds = user_bounds
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
                self._user_bounds,
            )

            self._reference_point = ref_point
            self._projection_index = proj_i

        new_nav = self.calculate_navigation_point(
            self._pareto_front[self._projection_index],
            self._navigation_point,
            self._steps_remaining,
        )

        self._navigation_point = new_nav
        self._user_bounds = user_bounds

        new_lb, new_ub = self.calculate_bounds(
            self._pareto_front[self._reachable_idx],
            self._navigation_point,
            self._user_bounds,
            self._reachable_lb,
            self._reachable_ub,
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

    @staticmethod
    def solve_nautilus_asf_problem(
        pareto_f: np.ndarray,
        subset_indices: List[int],
        ref_point: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        user_bounds: np.ndarray,
    ) -> int:
        """Forms and solves the achievement scalarizing function to find the
        closest point on the Pareto optimal front to the given reference
        point.

        Args:
            pareto_f (np.ndarray): The whole Pareto optimal front.
            subset_indices ([type]): Indices of the currently reachable solutions.
            ref_point (np.ndarray): The reference point indicating a decision
                maker's preference.
            ideal (np.ndarray): Ideal point.
            nadir (np.ndarray): Nadir point.
            user_bounds (np.ndarray): Bounds given by the user (the DM) for each objective,which should not be
                exceeded. A 1D array where NaN's indicate 'no bound is given' for the respective objective value.

        Returns:
            int: Index of the closest point according the minimized value of the ASF.
        """
        asf = PointMethodASF(nadir, ideal)
        scalarizer = DiscreteScalarizer(asf, {"reference_point": ref_point})
        solver = DiscreteMinimizer(scalarizer)

        # Copy the front and filter out the reachable solutions.
        # If user bounds are given, filter out solutions outside the those bounds.
        # Infeasible solutions on the pareto font are set to be NaNs.
        tmp = np.copy(pareto_f)
        mask = np.zeros(tmp.shape[0], dtype=bool)
        mask[subset_indices] = True
        tmp[~mask] = np.nan

        # indices of solutions with one or more objective value exceeding the user bounds.
        bound_mask = np.any(tmp > user_bounds, axis=1)
        tmp[bound_mask] = np.nan

        res = solver.minimize(tmp)

        return res["x"]

    def calculate_navigation_point(
        self, projection: np.ndarray, nav_point: np.ndarray, steps_remaining: int,
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
        new_nav_point = ((steps_remaining - 1) / steps_remaining) * nav_point + (
            1 / steps_remaining
        ) * projection
        return new_nav_point

    @staticmethod
    def calculate_bounds(
        pareto_front: np.ndarray,
        nav_point: np.ndarray,
        user_bounds: np.ndarray,
        previous_lb: np.ndarray,
        previous_ub: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the new bounds of the reachable points on the Pareto
        optimal front from a navigation point.

        Args:
            pareto_front (np.ndarray): The Pareto optimal front.
            nav_point (np.ndarray): The current navigation point.
            user_bounds (np.ndarray): Bounds given by the user (the DM) for each objective,which should not be
                exceeded. A 1D array where NaN's indicate 'no bound is given' for the respective objective value.
            previous_lb (np.ndarray): If no new lower bound can be found for an objective, this value is used.
            previous_ub (np.ndarray): If no new upper bound can be found for an objective, this value is used.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The lower and upper bounds.
        """
        # make sure the front is at least 2D
        _pareto_front = np.atleast_2d(pareto_front)

        new_lower_bounds = np.zeros(_pareto_front.shape[1])
        new_upper_bounds = np.zeros(_pareto_front.shape[1])

        # discard solutions that breach the given user bounds by setting them to NaN
        user_bounds_mask = np.any(_pareto_front > user_bounds, axis=1)
        _pareto_front[user_bounds_mask] = np.nan

        # TODO: vectorize this loop
        for r in range(_pareto_front.shape[1]):
            mask = np.zeros(_pareto_front.shape[1], dtype=bool)
            mask[r] = True

            subject_to = _pareto_front[:, ~mask].reshape(
                (_pareto_front.shape[0], _pareto_front.shape[1] - 1)
            )

            con_mask = np.all(subject_to <= nav_point[~mask], axis=1)

            if _pareto_front[con_mask, mask].size != 0:
                min_val = np.nanmin(_pareto_front[con_mask, mask])
                max_val = np.nanmax(_pareto_front[con_mask, mask])
            else:
                min_val = previous_lb[r]
                max_val = previous_ub[r]

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
        "user_bounds": [None, None],
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
        print(req.content["user_bounds"])

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
