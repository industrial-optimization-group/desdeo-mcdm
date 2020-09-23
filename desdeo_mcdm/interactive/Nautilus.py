from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from desdeo_tools.interaction.request import BaseRequest
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod


class NautilusException(Exception):
    """Raised when an exception related to Nautilus is encountered.

    """

    pass


class NautilusInitialRequest(BaseRequest):
    """ A request class to handle the initial preferences.

    """

    def __init__(self, ideal: np.ndarray, nadir: np.ndarray):
        msg = (
            "Please specify the number of iterations as 'n_iterations' to be carried out.\n"
            "Please specify as 'preference_method' whether to \n"
            "1. Rank the objectives in increasing order according to the importance of improving their value.\n"
            "2. Specify percentages reflecting how much would you like to improve each of the current objective "
            "values."
            "Depending on your selection on 'preference_method', please specify either the ranks or percentages for "
            "each objective as 'preference_info'."
        )
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
        }

        super().__init__("reference_point_preference", "required", content=content)

    def validator(self, response: Dict) -> None:
        """
        Validate decision maker's response.
        """
        if "n_iterations" not in response:
            raise NautilusException("'n_iterations' entry missing")
        self.validate_preferences()
        n_iterations = response["n_iterations"]

        if not isinstance(n_iterations, int) or int(n_iterations) < 1:
            raise NautilusException("'n_iterations' must be a positive integer greater than zero")
    # TODO: Continue here
    def validate_preferences(self, response: Dict) -> None:
        """
        Validate decision maker's preferences.
        """
        if "preference_method" not in response:
            raise NautilusException("'preference_method entry missing")
        if response["preference_method"] not in [1, 2]:
            raise NautilusException("please specify either preference method 1 (rank) or 2 (percentages).")
        if "preference_info" not in response:
            raise NautilusException("'preference_info entry missing")
        if response["preference_method"] == 1 and response["preference_info"]:
            pass

    @classmethod
    def init_with_method(cls, method):
        return cls(method._ideal, method._nadir)

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NautilusRequest(BaseRequest):
    """A request class to handle the intermediate requests.

    """

    def __init__(
            self,
            ideal: np.ndarray,
            nadir: np.ndarray,
            points: np.ndarray,
            lower_bounds: np.ndarray,
            upper_bounds: np.ndarray,
            minimize: List[int],
    ):

        msg = "Please select whether to \n" \
              "1. Rank the objectives according importance of improving their value (press 1).\n" \
              "2. Specify percentages reflecting how much would you like to improve each of the current objective " \
              "values (press 2)."
        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
            "lower_bounds": lower_bounds,
            "upper_bounds": upper_bounds,
            "distances": distances,
            "minimize": minimize,
        }

        super().__init__("reference_point_preference", "required", content=content)

    def validator(self, response: Dict) -> None:
        if "preference_method" not in response:
            raise NautilusException("'preference_method' not specified")

        pref_method = response["preference_method"]

        if pref_method not in [1, 2]:
            raise NautilusException("Please select either method 1 or 2.")

    @BaseRequest.response.setter
    def response(self, response: Dict):
        self.validator(response)
        self._response = response


class NautilusStopRequest(BaseRequest):
    """A request class to handle termination.

    """

    def __init__(self, preferred_point: np.ndarray):
        msg = "Most preferred solution found."
        content = {"message": msg, "solution": preferred_point}

        super().__init__("print", "no_interaction", content=content)


class Nautilus(InteractiveMethod):
    def __init__(
            self,
            # pareto_front: np.ndarray,
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
            NautilusException: One or more dimension mismatches are
            encountered among the supplies arguments.
        """
        # NOTE: no pareto front as input(?) - pareto optimal points are calculated at every iteration- not beforehand
        """
        if not pareto_front.ndim == 2:
            raise NautilusException(
                "The supplied Pareto front should be a two dimensional array. Found "
                f" number of dimensions {pareto_front.ndim}."
            )

        if not ideal.shape[0] == pareto_front.shape[1]:
            raise NautilusException(
                "The Pareto front must consist of objective vectors with the "
                "same number of objectives as defined in the ideal and nadir "
                "points."
            )
            
        """

        if not ideal.shape == nadir.shape:
            raise NautilusException("The dimensions of the ideal and nadir point do not match.")

        if objective_names:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusException(
                    "The supplied objective names must have a leangth equal to " "the numbr of objectives."
                )
            self._objective_names = objective_names
        else:
            self._objective_names = [f"f{i + 1}" for i in range(ideal.shape[0])]

        if minimize:
            if not len(objective_names) == ideal.shape[0]:
                raise NautilusException("The minimize list must have " "as many elements as there are objectives.")
            self._minimize = minimize
        else:
            self._minimize = [1 for _ in range(ideal.shape[0])]

        self._ideal = ideal
        self._nadir = nadir

        # in objective space!
        # self._pareto_front = pareto_front

        # bounds of the rechable region
        self._reachable_ub = self._nadir
        self._reachable_lb = self._ideal

        # currently reachable solution as a list of indices of the Pareto front
        # self._reachable_idx = list(range(0, self._pareto_front.shape[0]))

        # current iteration step number
        self._step_number = 1

        self._distance = None

        self._preferred_point = None
        self._projection_index = None

        self._n_iterations = None
        self._n_iterations_left = None

    def start(self) -> NautilusInitialRequest:
        return NautilusInitialRequest.init_with_method(self)

    def iterate(
            self, request: Union[NautilusInitialRequest, NautilusRequest]
    ) -> Union[NautilusRequest, NautilusStopRequest]:
        """Perform the next logical iteration step based on the given request type.

        """
        if type(request) is NautilusInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is NautilusRequest:
            return self.handle_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(self, request: NautilusInitialRequest) -> NautilusRequest:
        """Handles the initial request by parsing the response appropiately.

        """
        self._n_iterations = request.response["n_iterations"]
        self._n_iterations_left = self._n_iterations

        return NautilusRequest(
            self._ideal, self._nadir, self._minimize
        )

    def handle_request(self, request: ENautilusRequest) -> Union[ENautilusRequest, ENautilusStopRequest]:
        """Handles the intermediate requests.

        """
        preferred_point_index = request.response["preferred_point_index"]
        self._preferred_point = request.content["points"][preferred_point_index]

        if self._n_iterations_left <= 1:
            self._n_iterations_left = 0
            return ENautilusStopRequest(self._preferred_point)

        self._reachable_lb = request.content["lower_bounds"][preferred_point_index]
        self._reachable_ub = request.content["upper_bounds"][preferred_point_index]
        self._distance = request.content["distances"][preferred_point_index]

        self._reachable_idx = self.calculate_reachable_point_indices(
            self._pareto_front, self._reachable_lb, self._reachable_ub
        )

        # increment and decrement iterations
        self._n_iterations_left -= 1
        self._step_number += 1

        # Start again
        zbars = self.calculate_representative_points(self._pareto_front, self._reachable_idx, self._n_points)
        zs = self.calculate_intermediate_points(self._preferred_point, zbars, self._n_iterations_left)
        new_lower_bounds, new_upper_bounds = self.calculate_bounds(self._pareto_front, zs)
        distances = self.calculate_distances(zs, zbars, self._nadir)

        return ENautilusRequest(
            self._ideal, self._nadir, zs, new_lower_bounds, new_upper_bounds, distances, self._minimize
        )

    def calculate_representative_points(
            self, pareto_front: np.ndarray, subset_indices: List[int], n_points: int
    ) -> np.ndarray:
        """Calcualtes the most representative points on the Pareto front. The points are clustered using k-means.

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

            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, pareto_front[subset_indices])

            zbars = pareto_front[subset_indices][closest]

        else:
            zbars = pareto_front[subset_indices]

        return zbars

    def calculate_intermediate_points(
            self, preferred_point: np.ndarray, zbars: np.ndarray, n_iterations_left: int,
    ) -> np.ndarray:
        """Calcualtes the intermediate points between representative points an a preferred point.

        Args:
            preferred_point (np.ndarray): The preferred point, 1D array.
            zbars (np.ndarray): The representative points, 2D array.
            n_iterations_left (int): The number of iterations left.

        Returns:
            np.ndarray: The intermediate points as a 2D array.
        """
        zs = ((n_iterations_left - 1) / n_iterations_left) * preferred_point + (1 / n_iterations_left) * zbars
        return np.atleast_2d(zs)

    def calculate_bounds(
            self, pareto_front: np.ndarray, intermediate_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the new bounds of the reachable points on the Pareto
        optimal front from each of the intermediate points.

        Args:
            pareto_front (np.ndarray): The Pareto optimal front.
            intermediate_points (np.ndarray): The current intermedaite points as a 2D array.

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

                subject_to = _pareto_front[:, ~mask].reshape((_pareto_front.shape[0], _pareto_front.shape[1] - 1))

                con_mask = np.all(subject_to <= point[~mask], axis=1)

                min_val = np.min(_pareto_front[con_mask, mask])
                max_val = np.max(_pareto_front[con_mask, mask])

                new_lower_bounds[i, r] = min_val
                new_upper_bounds[i, r] = max_val

        return new_lower_bounds, new_upper_bounds

    def calculate_distances(self, intermediate_points: np.ndarray, zbars: np.ndarray, nadir: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(np.atleast_2d(intermediate_points) - nadir, axis=1) / np.linalg.norm(
            np.atleast_2d(zbars) - nadir, axis=1
        )
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
            self, pareto_front: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
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


if __name__ == "__main__":
    # front = np.array([[1, 2, 3], [2, 3, 4], [2, 2, 3], [3, 2, 1]], dtype=float)
    # ideal = np.zeros(3)
    # nadir = np.ones(3) * 5
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
        print(method._n_iterations_left)
        req = method.iterate(req)
        print(req.content["points"])
        req.response = {"preferred_point_index": 0}

    print(method._n_iterations_left)
    req = method.iterate(req)
    print(method._n_iterations_left)
    print(method._distance)
    print(req.content["solution"])
