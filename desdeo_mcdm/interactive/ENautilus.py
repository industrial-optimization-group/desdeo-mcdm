from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


class ENautilusException(Exception):
    """Raised when an exception related to ENautilus is encountered.

    """

    pass


class ENautilusInitialRequest(BaseRequest):
    """ A request class to handle the initial preferences.

    """

    def __init__(self, ideal: np.ndarray, nadir: np.ndarray):
        msg = (
            "Please specify the number of iterations as 'n_iterations' to be carried out, and how many intermediate "
            "points to show as 'n_points'."
        )
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
        }

        super().__init__("reference_point_preference", "required", content=content)

    def validator(self, response: Dict) -> None:
        if "n_iterations" not in response:
            raise ENautilusException("'n_iterations' entry missing")

        if "n_points" not in response:
            raise ENautilusException("'n_points' entry missing")

        n_iterations = response["n_iterations"]

        if not isinstance(n_iterations, int) or int(n_iterations) < 1:
            raise ENautilusException(
                "'n_iterations' must be a positive integer greater than zero"
            )

        n_points = response["n_points"]

        if not isinstance(n_points, int) or int(n_points) < 1:
            raise ENautilusException(
                "'n_points' must be a positive integer greater than zero"
            )

    @classmethod
    def init_with_method(cls, method):
        return cls(method._ideal, method._nadir)

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class ENautilusRequest(BaseRequest):
    """A request class to handle the intermediate requests.

    """

    def __init__(
        self,
        ideal: np.ndarray,
        nadir: np.ndarray,
        points: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        n_iterations_left: int,
        distances: np.ndarray,
    ):
        self._max_index = len(np.squeeze(points))
        msg = """Please select the most preferred solution by index as 'preferred_point_index'.
            The number of remaining iterations may also be changed by 
            setting 'change_remaining' to True and
            supplying the desried number of remaining iterations as
            'new_iterations_left'.
            If you wish to step back, then set 'step_back' to True. When stepping back to a 
            previous iteration, that iteration's intermediate solutions should be supplied alongside
            the associated distances and upper and lower bounds as 'prev_distances',
            'prev_solutions', 'prev_lower_bounds', and
            'prev_upper_bounds'. The number of remaining iterations should be supplied as 
            well when stepping back as 'iterations_left'.
            When 'step_back' is true, 'preferred_point_index' and 'change_remaining' are ignored.
            """
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
            "points": points,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "n_iterations_left": n_iterations_left,
            "distances": distances,
        }

        super().__init__("reference_point_preference", "required", content=content)

    def validator(self, response: Dict) -> None:
        if (
            "preferred_point_index" not in response
            or "step_back" not in response
            or "change_remaining" not in response
        ):
            raise ENautilusException(
                "'preferred_point_index', 'step_back', and 'change_remaining' must be specified."
            )

        if not response["step_back"]:
            pref_point_index = response["preferred_point_index"]

            if pref_point_index < 0 or pref_point_index > self._max_index:
                raise ENautilusException("The given index is out of bounds.")

            if response["change_remaining"]:
                if "iterations_left" not in response:
                    raise ENautilusException(
                        "When 'change_remaining' is True, 'iterations_left' must be specified."
                    )
        else:
            # stepping back
            # check that the previous solution is given alongside its bounds.
            if "prev_solutions" not in response:
                raise ENautilusException("'prev_solutions' entry missing.")
            if "prev_lower_bounds" not in response:
                raise ENautilusException("'prev_lower_bounds' entry missing.")
            if "prev_upper_bounds" not in response:
                raise ENautilusException("'prev_upper_bounds' entry missing.")
            if "iterations_left" not in response:
                raise ENautilusException(
                    "When stepping back 'iterations_left' must be specified."
                )
            if "prev_distances" not in response:
                raise ENautilusException("'prev_distances' entry missing.")
            # TODO: if issues arise, checking the dimensions of the prev_* entries can
            # be beneficial.

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class ENautilusStopRequest(BaseRequest):
    """A request class to handle termination.

    """

    def __init__(
        self, preferred_point: np.ndarray, solution: Optional[np.ndarray] = None
    ):
        msg = "Most preferred solution found."
        content = {"message": msg, "objective": preferred_point, "solution": solution}

        super().__init__("print", "no_interaction", content=content)


class ENautilus(InteractiveMethod):
    def __init__(
        self,
        pareto_front: np.ndarray,
        ideal: np.ndarray,
        nadir: np.ndarray,
        objective_names: Optional[List[str]] = None,
        variables: Optional[np.ndarray] = None,
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
            variables (Optional[np.ndarray], optional): The decision variables
                of the objective vectors in pareto_front. The i'th variable vector
                in variables corresponds to the i'th objective vector in
                pareto_front. Defaults to None.

        Raises:
            ENavigatorException: One or more dimension mismatches are
                encountered among the supplies arguments.
        """
        if not pareto_front.ndim == 2:
            raise ENautilusException(
                "The supplied Pareto front should be a two dimensional array. Found "
                f" number of dimensions {pareto_front.ndim}."
            )

        if not ideal.shape[0] == pareto_front.shape[1]:
            raise ENautilusException(
                "The Pareto front must consist of objective vectors with the "
                "same number of objectives as defined in the ideal and nadir "
                "points."
            )

        if not ideal.shape == nadir.shape:
            raise ENautilusException(
                "The dimensions of the ideal and nadir point do not match."
            )

        if objective_names:
            if not len(objective_names) == ideal.shape[0]:
                raise ENautilusException(
                    "The supplied objective names must have a length equal to "
                    "the number of objectives."
                )
            self._objective_names = objective_names
        else:
            self._objective_names = [f"f{i+1}" for i in range(ideal.shape[0])]

        self._ideal = ideal
        self._nadir = nadir

        self._variables = variables

        # in objective space!
        self._pareto_front = pareto_front

        # bounds of the reachable region
        self._reachable_ub = self._nadir
        self._reachable_lb = self._ideal

        # currently reachable solution as a list of indices of the Pareto front
        self._reachable_idx = list(range(0, self._pareto_front.shape[0]))

        self._preferred_point = None
        self._projection_index = None

        self._n_points = None
        self._n_iterations_left = None

    def start(self) -> ENautilusInitialRequest:
        return ENautilusInitialRequest.init_with_method(self)

    def iterate(
        self, request: Union[ENautilusInitialRequest, ENautilusRequest]
    ) -> Union[ENautilusRequest, ENautilusStopRequest]:
        """Perform the next logical iteration step based on the given request type.

        """
        if type(request) is ENautilusInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is ENautilusRequest:
            return self.handle_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(
        self, request: ENautilusInitialRequest
    ) -> ENautilusRequest:
        """Handles the initial request by parsing the response appropriately.

        """
        self._n_iterations_left = request.response["n_iterations"]
        self._n_points = request.response["n_points"]
        # self._intermediate_points = np.repeat(np.atleast_2d(self._nadir), self._n_points, axis=0)
        self._preferred_point = self._nadir

        zbars = self.calculate_representative_points(
            self._pareto_front, self._reachable_idx, self._n_points
        )
        zs = self.calculate_intermediate_points(
            self._preferred_point, zbars, self._n_iterations_left
        )
        new_lower_bounds, new_upper_bounds = self.calculate_bounds(
            self._pareto_front, zs
        )
        distances = self.calculate_distances(zs, zbars, self._nadir)

        return ENautilusRequest(
            self._ideal,
            self._nadir,
            zs,
            new_lower_bounds,
            new_upper_bounds,
            self._n_iterations_left,
            distances,
        )

    def handle_request(
        self, request: ENautilusRequest
    ) -> Union[ENautilusRequest, ENautilusStopRequest]:
        """Handles the intermediate requests.

        """
        if not request.response["step_back"]:
            preferred_point_index = request.response["preferred_point_index"]
            self._preferred_point = request.content["points"][preferred_point_index]

            if self._n_iterations_left <= 1:
                self._n_iterations_left = 0
                # if self._variables is defined: first, find the index of
                # self._preferred point in self._pareto_front. Second,
                # return the variable vector at the found position. Otherwise,
                # do not return any variable vectors.
                if self._variables is not None:
                    idx = np.linalg.norm(
                        np.abs(self._pareto_front - self._preferred_point), axis=1
                    ).argmin()
                    solution = self._variables[idx]
                else:
                    solution = None
                return ENautilusStopRequest(self._preferred_point, solution)

            self._reachable_lb = request.content["lower_bounds"][preferred_point_index]
            self._reachable_ub = request.content["upper_bounds"][preferred_point_index]

            self._reachable_idx = self.calculate_reachable_point_indices(
                self._pareto_front, self._reachable_lb, self._reachable_ub
            )

            if not request.response["change_remaining"]:
                # decrement iterations left
                self._n_iterations_left -= 1
            else:
                self._n_iterations_left = request.response["iterations_left"]

            # Start again
            zbars = self.calculate_representative_points(
                self._pareto_front, self._reachable_idx, self._n_points
            )
            zs = self.calculate_intermediate_points(
                self._preferred_point, zbars, self._n_iterations_left
            )
            new_lower_bounds, new_upper_bounds = self.calculate_bounds(
                self._pareto_front, zs
            )
            distances = self.calculate_distances(zs, zbars, self._nadir)

        # stepping back
        else:
            zs = request.response["prev_solutions"]
            new_lower_bounds = request.response["prev_lower_bounds"]
            new_upper_bounds = request.response["prev_upper_bounds"]
            self._n_iterations_left = request.response["iterations_left"]
            distances = request.response["prev_distances"]

        return ENautilusRequest(
            self._ideal,
            self._nadir,
            zs,
            new_lower_bounds,
            new_upper_bounds,
            self._n_iterations_left,
            distances,
        )

    def calculate_representative_points(
        self, pareto_front: np.ndarray, subset_indices: List[int], n_points: int
    ) -> np.ndarray:
        """Calculates the most representative points on the Pareto front. The points are clustered using k-means.

        Args:
            pareto_front (np.ndarray): The Pareto front.
            subset_indices (List[int]): A list of indices representing the
                subset of the points on the Pareto front for which the
                representative points should be calculated.
            n_points (int): The number of representative points to be calculated.

        Returns:
            np.ndarray: A 2D array of the most representative points. If the
                subset of Pareto efficient points is less than n_points, returns
                the subset of the Pareto front.
        """
        if len(np.atleast_1d(subset_indices)) > n_points:
            kmeans = KMeans(n_clusters=n_points)
            kmeans.fit(pareto_front[subset_indices])

            closest, _ = pairwise_distances_argmin_min(
                kmeans.cluster_centers_, pareto_front[subset_indices]
            )

            zbars = pareto_front[subset_indices][closest]

        else:
            zbars = pareto_front[subset_indices]

        return zbars

    def calculate_intermediate_points(
        self, preferred_point: np.ndarray, zbars: np.ndarray, n_iterations_left: int,
    ) -> np.ndarray:
        """Calculates the intermediate points between representative points an a preferred point.

        Args:
            preferred_point (np.ndarray): The preferred point, 1D array.
            zbars (np.ndarray): The representative points, 2D array.
            n_iterations_left (int): The number of iterations left.

        Returns:
            np.ndarray: The intermediate points as a 2D array.
        """
        zs = ((n_iterations_left - 1) / n_iterations_left) * preferred_point + (
            1 / n_iterations_left
        ) * zbars
        return np.atleast_2d(zs)

    def calculate_bounds(
        self, pareto_front: np.ndarray, intermediate_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the new bounds of the reachable points on the Pareto
        optimal front from each of the intermediate points.

        Args:
            pareto_front (np.ndarray): The Pareto optimal front.
            intermediate_points (np.ndarray): The current intermediate points as a 2D array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The lower and upper bounds for each of the intermediate points.
        """
        _pareto_front = np.atleast_2d(pareto_front)
        n_points = np.atleast_2d(intermediate_points).shape[0]
        new_lower_bounds = np.zeros((n_points, _pareto_front.shape[1]))
        new_upper_bounds = np.zeros((n_points, _pareto_front.shape[1]))

        for i, point in enumerate(np.atleast_2d(intermediate_points)):
            # TODO: vectorize this loop
            for r in range(_pareto_front.shape[1]):
                mask = np.zeros(_pareto_front.shape[1], dtype=bool)
                mask[r] = True

                subject_to = _pareto_front[:, ~mask].reshape(
                    (_pareto_front.shape[0], _pareto_front.shape[1] - 1)
                )

                con_mask = np.all(subject_to <= point[~mask], axis=1)

                min_val = np.min(_pareto_front[con_mask, mask])
                max_val = np.max(_pareto_front[con_mask, mask])

                new_lower_bounds[i, r] = min_val
                new_upper_bounds[i, r] = max_val

        return new_lower_bounds, new_upper_bounds

    def calculate_distances(
        self, intermediate_points: np.ndarray, zbars: np.ndarray, nadir: np.ndarray
    ) -> np.ndarray:
        distances = np.linalg.norm(
            np.atleast_2d(intermediate_points) - nadir, axis=1
        ) / np.linalg.norm(np.atleast_2d(zbars) - nadir, axis=1)
        """Calculates the distance to the Pareto front for each intermediate
        point given utilizing representative points representing the
        intermediate points.

        Args:
            intermediate_points (np.ndarray): The intermediate points, 2D array.
            zbars (np.ndarray): The representative points corresponding to the intermediate points, 2D array.
            nadir (np.ndarray): The nadir point, 1D array.

        Returns:
            np.ndarray: The distances calculated for each intermediate point to the Pareto front.
        """
        return distances * 100

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
