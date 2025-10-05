"""
    Linear Knapsack problem.
"""

import torch
import numpy as np
import gurobipy as gp

from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import LINEAR_OBJ, TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST

from typing import Tuple, Union, Dict

########################################################################################################################

PROBLEM_ID = 'knapsack'

########################################################################################################################


class KnapsackProblem(OptimizationProblem):
    """
    A class representing a knapsack problem in which the item values are parameterized.
    """

    def __init__(self, dim: int):
        """

        :param dim: int; number of items.
        """

        self._obj_type = LINEAR_OBJ
        self._is_minimization_problem = False
        self._name = PROBLEM_ID
        self._dim = dim

    @property
    def obj_type(self) -> str:
        """
        :return: str; a string identifier of the objective function type.
        """
        return self._obj_type

    @property
    def is_minimization_problem(self) -> bool:
        """
        :return: bool; the optimization problem direction, True for min and False for max.
        """
        return self._is_minimization_problem

    @property
    def name(self) -> str:
        """
        :return: str; the name of the optimization problem.
        """
        return self._name

    @property
    def dim(self) -> int:
        """
        :return: int; the problem dimension (number of items).
        """

        return self._dim

    def solve(self,
              y: np.ndarray,
              opt_prob_params: Dict,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the knapsack problem for a given vector of item weights
        :param y: numpy.ndarray; a given vector of item values.
        :param opt_prob_params: dict; the keys are the parameter names and the values are the corresponding torch.Tensor.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        weights = opt_prob_params['weights'].numpy()
        capacity = opt_prob_params['capacity'].numpy()

        # Create the Gurobi model
        model = gp.Model()

        # Suppress Gurobi output
        model.setParam('OutputFlag', 0)

        # Eventually, solve the LP relaxation rather than the integer version
        if "use_LP_relaxation" in kwargs and kwargs["use_LP_relaxation"]:
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.CONTINUOUS, name="x", lb=0, ub=1)
        else:
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="x")

        # Define the model
        model.setObjective(y @ x, gp.GRB.MAXIMIZE)
        model.addConstr(weights @ x <= capacity, name="eq")

        # Solve the model
        model.optimize()

        # Sanity check
        assert model.status == gp.GRB.OPTIMAL, "Problem was not solved to optimality"

        """solver = pywraplp.Solver.CreateSolver('SCIP')

        variables = [solver.IntVar(lb=0, ub=1, name='v' + str(i)) for i in range(self._dim)]
        objective = solver.Objective()
        for c, var in zip(y, variables):
            objective.SetCoefficient(var, float(c))
        objective.SetMaximization()

        constraint = solver.Constraint(0, float(capacity))
        for w, var in zip(weights, variables):
            constraint.SetCoefficient(var, float(w))

        status = solver.Solve()
        status == pywraplp.Solver.OPTIMAL
        sol = np.asarray([var.solution_value() for var in variables])"""

        return x.x, model.Runtime
        # return sol, 0

    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params) -> Dict:
        """
        Compute the objective value given the predictions and the solution.
        :param y: torch.Tensor; the predictions.
        :param sols: torch.Tensor; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: torch.Tensor; the objective value.
        """

        # FIXME: this is too hard-coded; we need to find a more extensible way
        # Unpack the optimization instance-specific parameters
        # y, weights, capacity = self._unpack_prob_params(y=y, opt_prob_params=opt_prob_params)

        # Compute the total cost
        total_cost = torch.matmul(sols, y)

        # Return the cost as a dictionary with information about the total cost, the cost of the feasible solutions
        # and the ration of feasible solutions
        cost = dict()
        cost[TOTAL_COST] = total_cost
        cost[SUBOPTIMALITY_COST] = total_cost
        # Since predictions are the item values, we can not violate any constraints
        cost[PENALTY_COST] = 0

        return cost, True

