from desdeo_problem import Constraint
from desdeo_mcdm.interactive.Nautilus import NautilusException
from desdeo_problem.Problem import MOProblem
from desdeo_tools.scalarization.ASF import PointMethodASF, ReferencePointASF, SimpleASF
from desdeo_tools.scalarization.Scalarizer import Scalarizer
from desdeo_tools.solver.ScalarSolver import ScalarMethod, ScalarMinimizer
from desdeo_mcdm.interactive.ReferencePointMethod import validate_reference_point
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from desdeo_mcdm.interactive.InteractiveMethod import InteractiveMethod
from desdeo_tools.interaction.request import BaseRequest, SimplePlotRequest
from scipy.spatial import ConvexHull

from scipy.optimize import linprog, differential_evolution

#TODO
# Scipy linprog failing :/
# Request validations
# Discrete case
# Plot requests
# A lot of checking and validation
# Remove new direction from request
# Remove pref_method from request

# Questions
# adaptive approximation method
# Assuming in requests
class ParetoNavigatorException(Exception):
    """Raised when an exception related to Pareto Navigator is encountered.
    """

    pass


class ParetoNavigatorInitialRequest(BaseRequest):
    """
    A request class to handle the Decision Maker's initial preferences for the first iteration round.

    In what follows, the DM is involved. First, the DM is asked to select a starting
    point for the navigation phase.
    """

    def __init__(self, ideal: np.ndarray, nadir: np.ndarray, allowed_speeds: np.ndarray) -> None:
        """
        Initialize with ideal and nadir vectors.

        Args:
            ideal (np.ndarray): Ideal vector
            nadir (np.ndarray): Nadir vector
            allowed_speeds (np.ndarray): Allowed movement speeds
        """

        self._ideal = ideal
        self._nadir = nadir
        self._allowed_speeds = allowed_speeds

        min_speed = np.min(self._allowed_speeds)
        max_speed = np.max(self._allowed_speeds)

        msg = "Please specify a starting point as 'preferred_solution'."
        "Or specify a reference point as 'reference_point'."
        "Please specify speed as 'speed' to be an integer value between"
        f"{min_speed} and {max_speed} "
        f"where {min_speed} is the slowest speed and {max_speed} the fastest."

        content = {
            "message": msg,
            "ideal": ideal,
            "nadir": nadir,
        }

        # Could also be a ref point
        super().__init__("preferred_solution_preference", "required", content=content)

    @classmethod
    def init_with_method(cls, method: InteractiveMethod):
        """
        Initialize request with given instance of ReferencePointMethod.

        Args:
            method (ReferencePointMethod): Instance of ReferencePointMethod-class.
        Returns:
            RPMInitialRequest: Initial request.
        """

        return cls(method._ideal, method._nadir, method._allowed_speeds)

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for initial request.

        Args:
            response (Dict): The Decision Maker's response.

        Raises:
            ParetoNavigatorException: In case reference point 
            or preferred solution is missing.
        """

        if "reference_point" in response:
            validate_reference_point(response["reference_point"], self._ideal, self._nadir)
        elif "preferred_solution" in response:
            # Validate
            pass
        else:
            msg = "Please specify either a starting point as 'preferred_solution'."
            "or a reference point as 'reference_point."
            raise ParetoNavigatorException(msg)
        
        if 'speed' not in response:
            msg = "Please specify a speed as 'speed'"
            raise ParetoNavigatorException(msg)
        
        speed = response['speed']
        try:
            if int(speed) not in self._allowed_speeds:
                raise ParetoNavigatorException(f"Invalid speed: {speed}.")
        except Exception as e:
            raise ParetoNavigatorException(
                f"An exception rose when validating the given speed {speed}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
            )
            
        self._response = response


class ParetoNavigatorRequest(BaseRequest):
    """
    A request class to handle the Decision Maker's preferences after the first iteration round.
    """

    def __init__(
        self, 
        current_solution: np.ndarray, 
        ideal: np.ndarray, 
        nadir: np.ndarray,
        allowed_speeds: np.ndarray,
        valid_classifications: np.ndarray,
    ) -> None:
        """
        Initialize request with current iterations's solution process information.

        Args:
            current_solution (np.ndarray): Current solution.
            ideal (np.ndarray): Ideal vector.
            nadir (np.ndarray): Nadir vector.
            allowed_speeds (np.ndarray): Allowed movement speeds
            valid_classifications (np.ndarray): Valid classifications
        """

        self._current_solution = current_solution
        self._ideal = ideal
        self._nadir = nadir
        self._allowed_speeds = allowed_speeds
        self._valid_classifications = valid_classifications

        min_speed = np.min(self._allowed_speeds)
        max_speed = np.max(self._allowed_speeds)

        msg = ( # TODO more understandable 
            "If you are satisfied with the current solution, please state:"
            "'satisfied' as 'True'."
            "If you wish to change the direction, please state:"
            "'new_direction' as 'True' AND"
            "specify a 'preference_method' as either"
            "1 = 'reference_point' OR 2 = 'preference_info'"
            "Depending on your selection on 'preference_method', please specify either"
            "'reference_point' as a new reference point OR"
            "classification' as a list of strings"
            "If you wish to step back specify 'step_back' as 'True'"
            "If you wish to change the speed, please specify speed"
            "as 'speed' to be an integer value between"
            f"{min_speed} and {max_speed}"
            f"where {min_speed} is the slowest speed and {max_speed} the fastest."

        )

        content = {"message": msg, "current_solution": current_solution}

        super().__init__("reference_point_preference", "required", content=content)
    
    @classmethod
    def init_with_method(cls, method: InteractiveMethod):
        """
        Initialize request with given instance of ParetoNavigator.

        Args:
            method (ParetoNavigator): Instance of ParetoNavigator-class.
        Returns:
            ParetoNavigatorRequest: Initial request.
        """

        return cls(
            method._current_solution,
            method._ideal,
            method._nadir,
            method._allowed_speeds,
            method._valid_classifications,
        )

    @BaseRequest.response.setter
    def response(self, response: Dict) -> None:
        """
        Set the Decision Maker's response information for request.

        Args:
            response (Dict): The Decision Maker's response.

        Raises:
            ParetoNavigatorException: In case response is invalid.
        """

        if 'satisfied' in response and response['satisfied']:
            self._response = response
            return

        if 'speed' not in response:
            raise ParetoNavigatorException("Please specify a speed")
        speed = response["speed"]
        try:
            if int(speed) not in self._allowed_speeds:
                raise ParetoNavigatorException(f"Invalid speed: {speed}.")
        except Exception as e:
            raise ParetoNavigatorException(
                f"An exception rose when validating the given speed {speed}.\n"
                f"Previous exception: {type(e)}: {str(e)}."
            )
        
        if 'new_direction' in response and response['new_direction']:
            if 'preference_method' not in response:
                msg = "Specify a preference as 'preference_method' method when changing direction"
                raise ParetoNavigatorException(msg)

            
            try:
                pref_method = int(response['preference_method'])
                if pref_method == 1:
                    if 'reference_point' not in response:
                        msg = "Specify a reference point when using reference point preference"
                        raise ParetoNavigatorException(msg)
                    else: # Validate reference point
                        validate_reference_point(response['reference_point'], self._ideal, self._nadir)
                elif pref_method == 2:
                    if 'classification' not in response:
                        msg = "Specify classifications when using classification prefence"
                        raise ParetoNavigatorException(msg)
                    else: # Validate classifications
                        classifications = np.unique(response['classification'])
                        if not np.all(np.isin(classifications, self._valid_classifications)):
                            msg = "Invalid classifications"
                            raise ParetoNavigatorException(msg)
                else: # Not 1 or 2
                    msg = "Preference method should be an integer value either 1 or 2"
                    raise ParetoNavigatorException(msg)
                response['preference_method'] = pref_method # Make sure is integer either 1 or 2
            except Exception as e:
                raise ParetoNavigatorException(f"Previous exception: {type(e)}: {str(e)}.")
        else: # New direction false or not specified
            response['new_direction'] = False # Is setting responses fine?


        self._response = response


class ParetoNavigatorStopRequest(BaseRequest):
    """
    A request class to handle termination.
    """

    def __init__(self, final_solution: np.ndarray, objective_values: np.ndarray = None) -> None:
        """
        Initialize termination request with final solution and objective vector.

        Args:
            final_solution (np.ndarray): Solution (decision variables).
            objective_values (np.ndarray): Objective vector.
        """
        msg = "Final solution found."

        content = {
            "message": msg,
            "final_solution": final_solution,
            "objective_values": objective_values
        }

        super().__init__("print", "no_interaction", content=content)


class ParetoNavigator(InteractiveMethod):
    """
    Paretonavigator as described in 'Pareto navigator for interactive nonlinear
    multiobjective optimization' (2008) [Petri Eskelinen · Kaisa Miettinen ·
    Kathrin Klamroth · Jussi Hakanen]. 

    Args:
        problem (MOProblem): The problem to be solved.
        pareto_optimal_solutions (np.ndarray): Some pareto optimal solutions to construct the polyhedral set
    """
    def __init__(
        self,
        problem: MOProblem,
        pareto_optimal_solutions: np.ndarray, # Initial pareto optimal solutions
        epsilon: float = 1e-6, # No need?
        scalar_method: Optional[ScalarMethod] = None
    ):
        if problem.nadir is None or problem.ideal is None:
            pass # adaptive approximation method -> Az<b, nadir, ideal

        self._scalar_method = scalar_method

        self._problem = problem

        self._ideal = problem.ideal
        self._nadir = problem.nadir
        self._utopian = self._ideal - epsilon # No need?
        self._n_objectives = self._ideal.shape[0]

        self._weights = self.calculate_weights(self._ideal, self._nadir)
        A, self.b =  self.polyhedral_set_eq(pareto_optimal_solutions)
        self.lppp_A= self.construct_lppp_A(self._weights, A) # Used in (3), Doesn't change

        self._pareto_optimal_solutions = pareto_optimal_solutions

        # initialize method with MOProblem
        # TODO discrete
        # self._objectives: Callable = lambda x: self._problem.evaluate(x).objectives
        # self._variable_bounds: Union[np.ndarray, None] = problem.get_variable_bounds()
        # self._variable_vectors = None
        # self._constraints: Optional[Callable] = lambda x: self._problem.evaluate(x).constraints

        self._allowed_speeds = [1, 2, 3, 4, 5]

        # Improve, degrade, maintain, 
        self._valid_classifications = ["<", ">", "="]

        self._current_speed = None
        self._reference_point = None
        self._current_solution = None
        self._direction = None
    
    def start(self):
        """
        Start the solving process

        Returns:
            ParetoNavigatorInitialRequest: Initial request
        """
        return ParetoNavigatorInitialRequest.init_with_method(self)
    
    def iterate(
        self, 
        request: Union[ParetoNavigatorInitialRequest, ParetoNavigatorRequest, ParetoNavigatorStopRequest]
    ) -> Union[ParetoNavigatorRequest, ParetoNavigatorStopRequest]:
        """
        Perform the next logical iteration step based on the given request type.

        Args:
            request (Union[ParetoNavigatorInitialRequest, ParetoNavigatorRequest,ParetoNavigatorStopRequest]):
            A ParetoNavigatorRequest

        Returns:
            Union[RPMRequest, RPMStopRequest]: A new request with content depending on the Decision Maker's
            preferences.
        """
        
        if type(request) is ParetoNavigatorInitialRequest:
            return self.handle_initial_request(request)
        elif type(request) is ParetoNavigatorRequest:
            return self.handle_request(request)
        else:
            # if stop request, do nothing
            return request

    def handle_initial_request(self, request: ParetoNavigatorInitialRequest) -> ParetoNavigatorRequest:
        if "reference_point" in request.response:
            self._reference_point = request.response["reference_point"]
            starting_point = self.solve_asf(self._problem, self._reference_point)
        else: # Preferred po solution
            starting_point = self._pareto_optimal_solutions[request.response["preferred_solution"]]

        self._current_solution = starting_point
        self._current_speed = request.response['speed']

        return ParetoNavigatorRequest.init_with_method(self)

    def handle_request(
        self,
        request: ParetoNavigatorRequest
    ) -> Union[ParetoNavigatorRequest, ParetoNavigatorStopRequest]:
        
        resp: dict = request.response
        if "satisfied" in resp and resp["satisfied"]:
            final_solution = self.solve_asf(
                self._problem,
                self._current_solution,
            )
            objectives = self._problem.evaluate(final_solution).objectives.squeeze()
            return ParetoNavigatorStopRequest(final_solution, objectives)

        # First iteration after initial, make sure preference is given
        if self._direction is None and ('new_direction' not in resp or not resp['new_direction']):
            if 'reference_point' not in resp: # Or other preference
                raise ParetoNavigatorException("One must specify preference information after starting the method")


        max_speed = np.max(self._allowed_speeds)
        if 'speed' in resp:
            self._current_speed = resp['speed'] / max_speed
        
        if 'step_back' in resp and resp['step_back']:
            self._current_speed *= -1
        else: # Make sure speed is positive
            self._current_speed = np.abs(self._current_speed) 
        
        if resp['new_direction']:
            if resp['preference_method'] == 1:
                self._reference_point = resp['reference_point']
            else:
                ref_point = self.classification_to_ref_point(resp["classification"])
                self._reference_point = ref_point

        self._direction = self.calculate_direction(self._current_solution, self._reference_point)
    
        # Get the new solution by solving the linear parametric problem
        self._current_solution = self.solve_linear_parametric_problem(
            self._current_solution,
            self._direction,
            self._current_speed,
            self.lppp_A,
            self.b
        )

        return ParetoNavigatorRequest.init_with_method(self)

    def calculate_weights(self, ideal: np.ndarray, nadir: np.ndarray):
        return 1 / (nadir - ideal)

    # HELP
    def polyhedral_set_eq(self, po_solutions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct a polyhedral set as convex hull satisfying Az < b 
        from the set of pareto optimal solutions

        Args:
            po_solutions (np.ndarray): Some pareto optimal solutions

        Returns:
            (np.ndarray, np.ndarray): Matrix A and vector b from the convex hull equation Az < b
        """
        convex_hull = ConvexHull(po_solutions)
        A = convex_hull.equations[:,0:-1]
        b = convex_hull.equations[:,-1]
        return A, b
    
    def construct_lppp_A(self, weights, A):
        """
        The matrix A used in the linear parametric programming problem
        """
        k = len(weights)
        diag = np.zeros((k,k))

        np.fill_diagonal(diag, 1)
        weights_inv = np.reshape(np.vectorize(lambda w: -1/w)(weights), (k,1))
        upper_A = np.hstack((weights_inv, diag))

        fill_zeros = np.zeros((len(A), 1))
        filled_A = np.hstack((fill_zeros, A))

        lppp_A = np.concatenate((upper_A, filled_A))
        return lppp_A
    
    def calculate_direction(self, current_solution: np.ndarray, ref_point: np.ndarray):
        return ref_point - current_solution
    
    def classification_to_ref_point(self, classifications):
        """
        Transform a classification to a reference point

        Args:
            classifications (np.ndarray): Classification for each objective

        Returns:
            np.ndarray: A reference point which is constructed from the classifications
        """
        def mapper(c: str, i: int):
            if c == "<": return self._ideal[i]
            elif c == ">": return self._nadir[i]
            else: return self._current_solution[i] #  c == "=". Request handles invalid classifications
        k = len(classifications)
        ref_point = [mapper(c, i) for i, c in (list(enumerate(classifications)))]
        return np.array(ref_point)
    
    # HELP
    def solve_linear_parametric_problem(
        self,
        current_sol: np.ndarray, # z^c
        direction: np.ndarray, # d
        a: float, # alpha
        A: np.ndarray, # Az < b
        b: np.ndarray,
    ) -> np.ndarray:
        """
        The linear parametric programming problem  (3)
        """
        k = len(current_sol)
        c = np.array([1] + k*[0])

        moved_ref_point = current_sol + (a * direction) # Z^-
        moved_ref_point = np.reshape(moved_ref_point, ((k,1)))
        b_new = np.append(moved_ref_point, b) # b'

        obj_bounds = np.stack((self._ideal, self._nadir))
        bounds = [(None, None)] + [(x,y) for x,y in obj_bounds.T] # sequence of pairs
        sol = linprog(c = c,A_ub = A, b_ub = b_new, bounds=bounds)
        if sol["success"]:
            return sol["x"][1:] # zeta in index 0.
        else:
            print("failed")
            return sol["x"][1:] 
            # raise ParetoNavigatorException("Couldn't calculate new solution")

    def solve_asf(self, problem: MOProblem, ref_point: np.ndarray):
        """
        Solve achievement scalarizing function

        Args:
            problem (MOProblem): The problem 
            ref_point: A reference point
        
        Returns: 
            np.ndarray: The decision vector which solves the achievement scalarizing function
        """
        asf = SimpleASF(np.ones(ref_point.shape))     
        scalarizer = Scalarizer(
            lambda x: problem.evaluate(x).objectives,
            asf,
            scalarizer_args={"reference_point": np.atleast_2d(ref_point)}
        )   

        if problem.n_of_constraints > 0:
            _con_eval = lambda x: problem.evaluate(x).constraints.squeeze()
        else:
            _con_eval = None
        
        solver = ScalarMinimizer(
            scalarizer, problem.get_variable_bounds(), constraint_evaluator=_con_eval, method=self._scalar_method,
        )

        res = solver.minimize(problem.get_variable_upper_bounds() / 2)

        if res["success"]:
            return res["x"]
        
        else:
            raise ParetoNavigatorException("Could solve achievement scalarazing function")

# Testing
if __name__ == "__main__":
    from desdeo_problem.Objective import _ScalarObjective
    from desdeo_problem import variable_builder

    # Objectives
    def f1(xs):
        xs = np.atleast_2d(xs)
        return -xs[:, 0] - xs[:, 1] + 5

    def f2(xs):
        xs = np.atleast_2d(xs)
        return (
            (1/5) *
            (
                np.square(xs[:, 0]) -
                10 * xs[:, 0] +
                np.square(xs[:, 1]) -
                4 * xs[:, 1] + 
                11
            )
        )

    def f3(xs):
        xs = np.atleast_2d(xs)
        return (5 - xs[:, 0])*(xs[:, 1] - 11)

    obj1 = _ScalarObjective("obj1", f1)
    obj2 = _ScalarObjective("obj2", f2)
    obj3 = _ScalarObjective("obj3", f3)
    objectives = [obj1, obj2, obj3]
    objectives_n = len(objectives)

    # TODO other constraints

    # variables
    var_names = ["x1", "x2"]  # Make sure that the variable names are meaningful to you.
    variables_n = len(var_names)

    initial_values = np.array([2, 3])
    lower_bounds = [0, 0]
    upper_bounds = [4, 6]
    bounds = np.stack((lower_bounds, upper_bounds))
    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

    # Constraints
    def c1(xs, ys):
        xs = np.atleast_2d(xs)
        return np.negative((3 * xs[:,0] + xs[:,1] - 12))
    
    def c2(xs, ys):
        xs = np.atleast_2d(xs)
        return np.negative((2 * xs[:,0] + xs[:,1] -9))

    def c3(xs, ys):
        xs = np.atleast_2d(xs)
        return np.negative((xs[:,0] + 2 * xs[:,1] - 12))
    
    con1 = Constraint.ScalarConstraint("c1", variables_n, objectives_n, c1)
    con2 = Constraint.ScalarConstraint("c2", variables_n, objectives_n, c2)
    con3 = Constraint.ScalarConstraint("c3", variables_n, objectives_n, c3)
    constraints = [con1, con2, con3]

    # problem
    problem = MOProblem(objectives=objectives, variables=variables, constraints=constraints)  # objectives "seperately"


    from desdeo_mcdm.utilities.solvers import payoff_table_method
    ideal, nadir = np.array([[-2, -3.1, -55], [5.0, 4.6, -14.25]]) # payoff_table_method(problem)
    problem.ideal = ideal
    problem.nadir = nadir

    po_sols = np.array([
        [-2, 0, -18],
        [-1, 4.6, -25],
        [0, -3.1, -14.25],
        [1.38, 0.62, -35.33],
        [1.73, 1.72, -38.64],
        [2.48, 1.45, -42.41],
        [5.00, 2.20, -55.00],
    ])

    method = ParetoNavigator(problem, po_sols)
    
    request = method.start()
    print(request.content['message'])

    request.response = {
        'preferred_solution': 3,
        'speed': 1,
    }

    request = method.iterate(request)
    print(request.content['message'])
    print(request.content['current_solution'])

    request.response = {
        # 'reference_point': np.array([ideal[0], ideal[1], nadir[2]]),
        'speed': 3,
        'new_direction': True,
        'preference_method': 2,
        'classification': ['<', '<', '>']
    }

    for i in range(15):
        request = method.iterate(request)
        print(request.content["current_solution"])

        request.response = {
            'satisfied': False,
            'speed': 3,
        }
    
    cur_sol = request.content["current_solution"]

    request.response = {
        'preference_method': 2,
        'classification': ['<', '>', '='],
        # 'reference_point': np.array([ideal[0], nadir[1], cur_sol[2]]),
        'new_direction': True,
        'speed': 5,
        'satisfied': False,
    }

    for i in range(15):
        request = method.iterate(request)
        print(request.content["current_solution"])

        request.response = {
            'satisfied': False,
            'speed': 3,
        }
    
    request.response = {
        'preference_method': 2,
        'classification': ['>', '<', '<'],
        'new_direction': True,
        'speed': 3,
        'satisfied': False,
    }
    
    for i in range(3):
        request = method.iterate(request)
        print(request.content["current_solution"])

        request.response = {
            'satisfied': False,
            'speed': 3,
        }

    request.response = {
        'satisfied': True,
    }

    request = method.iterate(request)
    sol = request.content["final_solution"]
    print(sol)
    print(request.content["objective_values"])