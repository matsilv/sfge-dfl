"""
    Quadratic Knapsack.
"""

import torch
import numpy as np
import gurobipy as gp
from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import QUADRATIC_OBJ, TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST

from typing import Tuple, Union, Dict

########################################################################################################################

PROBLEM_ID = 'quadratic_kp'

########################################################################################################################


class QuadraticKP(OptimizationProblem):
    """
    A class representing a quadratic knapsack problem in which the item values are parameterized.
    """

    def __init__(self, dim: int = 48):

        self._obj_type = QUADRATIC_OBJ
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
        :param opt_prob_params: numpy.ndarray; the instance-specific optimization problem parameters.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        weights = opt_prob_params['weights'].numpy()
        capacity = opt_prob_params['capacity'].numpy()
        values = y.reshape(self._dim, self._dim)

        # Create the Gurobi model
        model = gp.Model()

        # Suppress the output
        model.setParam('OutputFlag', 0)

        # Solve the relaxed problem if required
        if "use_LP_relaxation" in kwargs and kwargs["use_LP_relaxation"]:
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.CONTINUOUS, name="x", lb=0, ub=1)
        else:
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="x")

        # Add the capacity constraint
        model.addConstr(weights @ x <= capacity, name="capacity")

        # Compute the quadratic cost
        cost = x @ values @ x

        # Set the objective function
        model.setObjective(cost, gp.GRB.MAXIMIZE)

        # Solve the problem
        model.optimize()

        # Sanity check
        assert model.status == gp.GRB.OPTIMAL, "Problem was not solved to optimality"

        # If the problem is not solved to optimality, raise an exception
        return x.x, model.Runtime

    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params) -> dict:
        """
        Compute the objective value given the predictions and the solution.
        :param y: torch.Tensor; the predictions.
        :param sols: torch.Tensor; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: torch.Tensor; the objective value.
        """

        # Example: we have 3 decision variables x1, x2, x3, x4
        # We have to compute the following dot product:
        # | x1 x1 x1 x1 |   | x1 x2 x3 x4 |   | x1*x1 x1*x2 x1*x3 x1*x4 |
        # | x2 x2 x2 x2 | * | x1 x2 x3 x4 | = | x2*x1 x2*x2 x2*x3 x2*x4 |
        # | x3 x3 x3 x3 |   | x1 x2 x3 x4 |   | x3*x1 x3*x2 x3*x3 x3*x4 |
        # | x4 x4 x4 x4 |   | x1 x2 x3 x4 |   | x4*x1 x4*x2 x4*x3 x4*x4 |

        # We compute the above operation in vectorized form (matrix are unrolled)

        # The length corresponds to the number of items
        sol_dim = len(sols)
        assert sol_dim**2 == len(y)

        # This is the computation described above
        first_term = sols.repeat_interleave(sol_dim)
        second_term = sols.repeat(sol_dim)
        dot_prod = first_term * second_term
        total_cost = torch.matmul(dot_prod, y)

        # Return the cost as a dictionary with information about the total cost, the cost of the feasible solutions
        # and the ration of feasible solutions
        cost = dict()
        cost[TOTAL_COST] = total_cost
        cost[SUBOPTIMALITY_COST] = total_cost
        cost[PENALTY_COST] = 0

        return cost, True
