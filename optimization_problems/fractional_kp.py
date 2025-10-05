"""
    Fractional Knapsack problem.
"""

import torch
import numpy as np
from optimization_problems.optimization_problem import OptimizationProblem
import gurobipy as gp
from optimization_problems import LINEAR_OBJ, TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST

from typing import Tuple, Union, Dict

########################################################################################################################

PROBLEM_ID = 'fractional_kp'

########################################################################################################################


class FractionalKPProblem(OptimizationProblem):
    """
    A class representing a fractional knapsack problem in which both the items values and weights are parameterized.
    """

    def __init__(self, dim: int):
        """
        :param dim: int; number of items.
        """

        self._obj_type = LINEAR_OBJ
        self._dim = dim
        self._is_minimization_problem = False
        self._name = PROBLEM_ID

    @property
    def obj_type(self) -> str:
        """
        :return: str; the type of objective function of the optimization problem (linear, quadratic, etc...).
        """
        return self._obj_type

    @property
    def dim(self) -> int:
        """
        :return: int; the problem dimension (the number of items).
        """
        return self._dim

    @property
    def is_minimization_problem(self) -> bool:
        """
        :return: bool; the optimization problem direction, True for min and False for max.
        """
        return self._is_minimization_problem

    def solve(self,
              y: np.ndarray,
              opt_prob_params: np.ndarray,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the knapsack problem for a given vector of item weights
        :param y: numpy.ndarray; a given vector of item values.
        :param opt_prob_params: dict; the keys are the parameter names and the values are the corresponding torch.Tensor.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        # Unpack the optimization problem parameters from the predictions and the instance-specific array of parameters
        item_values = y[:self._dim]
        item_weights = y[self._dim:]
        capacity = opt_prob_params['capacity']

        # Create the Gurobi model
        model = gp.Model()
        # Suppress Gurobi output
        model.setParam('OutputFlag', 0)

        # Define the decision variables
        x = model.addMVar(shape=self._dim, vtype=gp.GRB.CONTINUOUS, name="x", lb=0, ub=1)

        # Define the model
        model.setObjective(item_values @ x, gp.GRB.MAXIMIZE)
        model.addConstr(item_weights @ x <= capacity, name="eq")

        # Solve the model
        model.optimize()

        # Sanity check
        assert model.status == gp.GRB.OPTIMAL, "Problem was not solved to optimality"

        return x.x, model.Runtime

    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params: np.ndarray) -> Dict:
        """
        Compute the objective value given the predictions and the solution. In this case the predictions are not needed
        to compute the objective value but we want to be consistent with the function signature.
        :param y: numpy.ndarray; the predictions.
        :param sols: numpy.ndarray; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: dict; we keep track of the total, suboptimality and violation.
        """

        # Unpack the optimization problem parameters from the predictions and the instance-specific array of parameters
        item_values = y[:self._dim]
        item_weights = y[self._dim:]
        capacity = opt_prob_params['capacity']
        item_penalties = opt_prob_params['penalty']
        capacity = torch.as_tensor(capacity)
        capacity = torch.squeeze(capacity)
        item_penalties = torch.as_tensor(item_penalties)

        # Here we compute the post-hoc regret and the correction function as implemented in
        # https://github.com/dadahxy/AAAI_PostHocRegret/blob/main/frational%20knapsack/runModel.py
        if min(item_weights) >= 0:
            sol_cost = sols @ item_values

            # Correction action
            lmbd = 1
            selected_total_weight = sols @ item_weights

            if selected_total_weight > capacity:
                lmbd = capacity / selected_total_weight

            discarded_items = sols * (1 - lmbd)
            sol_cost *= lmbd
            penalty_cost = item_values * item_penalties @ discarded_items

            total_cost = sol_cost - penalty_cost

            # Check whether the predicted solution is feasible
            if (discarded_items > 0).any():
                feasible = False
            else:
                feasible = True

        else:
            total_cost = 0
            penalty_cost = np.inf
            sol_cost = 0
            feasible = False

        # We keep track of the total, suboptimality and violation
        cost = dict()
        cost[TOTAL_COST] = total_cost
        cost[SUBOPTIMALITY_COST] = sol_cost
        cost[PENALTY_COST] = penalty_cost

        return cost, feasible
