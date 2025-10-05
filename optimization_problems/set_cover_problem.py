"""
    Weighted Set Multi-cover with stochastic coverage requirements.
"""

import numpy as np
import gurobipy as gp
import torch

from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import NONLINEAR_OBJ, TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST

from typing import Union, Tuple, Dict

########################################################################################################################

PROBLEM_ID = 'wsmc'

########################################################################################################################


# FIXME: shall we wrap optimization model parameters in a class?
class StochasticWeightedSetMultiCover(OptimizationProblem):
    """
    An abstract class representing for WSMC problem.
    """

    def __init__(self,
                 num_sets: int,
                 num_products: int):

        self._is_minimization_problem = True
        self._name = 'weighted_set_multi_cover'
        self._obj_type = NONLINEAR_OBJ

        # Problems parameters that are shared across all the instances; the only parameter that changes is the demands
        self._num_sets = num_sets
        self._num_products = num_products

    @property
    def num_sets(self) -> int:
        """
        :return: int; the number of sets.
        """
        return self._num_sets

    @property
    def num_products(self) -> int:
        """
        :return: int; the number of products.
        """
        return self._num_products

    @property
    def name(self) -> str:
        """
        :return: str; the problem identifier.
        """
        assert self._name is not None, "The problem name is not initialized"
        return self._name

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

    def solve(self,
              y: np.ndarray,
              opt_prob_params: Dict,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the stochastic WSMC problem for a given vector of predicted demands.
        :param y: np.ndarray; a given vector of item values.
        :param opt_prob_params: dict; the keys are the parameter names and the values are the corresponding torch.Tensor.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        set_costs = opt_prob_params['set_costs'].numpy()
        prod_costs = opt_prob_params['prod_costs'].numpy()
        availability = opt_prob_params['availability'].numpy()
        availability = availability.reshape(self._num_products, self._num_sets)

        # Create the Gurobi model
        env = gp.Env(empty=True)
        # Suppress Gurobi output
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)

        penalty_cost = 0

        # If a single demands vector is given then we solve the problem in a deterministic fashion and add a fake
        # scenarios dimension...
        if len(y.shape) == 1:
            num_scenarios = 1
            demands = np.expand_dims(y, axis=0)
        # ...otherwise solve the problem with the Sample Average Approximation algorithm by sampling multiple scenarios
        else:
            # Sanity check: first dimension is the number of scenarios whereas the second on is the number of items
            # (products)
            assert len(y.shape) == 2, "demands must be a 2-dimensional array"
            demands = y
            num_scenarios = y.shape[0]

        # This is the set of decision variables
        decision_vars = [model.addVar(vtype=gp.GRB.INTEGER,
                                      lb=0,
                                      name=f'x_{j}') for j in range(self._num_sets)]
        decision_vars = np.asarray(decision_vars)

        # This is the coverage requirements constraint matrix
        cover_constraint = availability @ decision_vars

        # Iterate for each scenario (1 if we solve the deterministic version of the problem)
        for omega in range(num_scenarios):
            current_demands = demands[omega]

            # Initialize the sets of indicator and slack variables
            all_indicator_vars = list()
            all_slack_vars = list()

            # Add the indicator constraints
            for i in range(0, self._num_products):
                indicator_var = model.addVar(vtype=gp.GRB.BINARY, name=f'z_{omega},{i}')
                slack_var = model.addVar(vtype=gp.GRB.INTEGER, name=f's_{omega},{i}')

                # Add the indicator and slack variables
                all_indicator_vars.append(indicator_var)
                all_slack_vars.append(slack_var)

                # LHS of the indicator constraint
                lhs_constraint = slack_var + cover_constraint[i]

                # Indicator constraint
                model.addGenConstrIndicator(binvar=indicator_var,
                                            binval=True,
                                            lhs=lhs_constraint,
                                            sense=gp.GRB.GREATER_EQUAL,
                                            rhs=current_demands[i],
                                            name=f'Indicator_constraint_{omega},{i}')

                # Add demands satisfaction constraint
                model.addConstr(cover_constraint[i] >= current_demands[i] * (1 - indicator_var))

            # Add the penalty when the indicator constraint is violated
            all_slack_vars = gp.MVar(all_slack_vars)
            penalty_cost += prod_costs @ all_slack_vars

        # Convert the list of decision variables in matrix form
        decision_vars = gp.MVar(decision_vars)

        # Objective function
        obj = set_costs @ decision_vars + penalty_cost / num_scenarios
        model.setObjective(obj, gp.GRB.MINIMIZE)

        # Solve the model
        model.optimize()
        status = model.status
        assert status == gp.GRB.Status.OPTIMAL, "Solution is not optimal"

        # Get the solution
        solution = [s.x for s in decision_vars]
        solution = np.asarray(solution, dtype=np.float32)

        return solution, model.Runtime

    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params: Dict) -> Tuple[Dict, bool]:
        """
        Compute the objective value of the solution given the costs.
        :param y: numpy.ndarray; the costs vector.
        :param sols: numpy.ndarray; the solution.
        :param opt_prob_params: dict of torch.Tensor; instance-specific optimization problem params.
        :return: float; the objective function value.
        """

        set_costs = opt_prob_params['set_costs']
        prod_costs = opt_prob_params['prod_costs']
        availability = opt_prob_params['availability']
        availability = availability.reshape(self._num_products, self._num_sets)

        # Convert model parameters from numpy to torch.Tensor
        set_costs = torch.as_tensor(set_costs, dtype=torch.float32)
        penalties = torch.as_tensor(prod_costs, dtype=torch.float32)
        availability = torch.as_tensor(availability, dtype=torch.float32)
        demands = torch.as_tensor(y, dtype=torch.float32)

        # The first term is the cost of the solution whereas the second one is the penalties for not satisfied demands
        tot_sets_cost = torch.matmul(set_costs, sols)
        covered_demands = torch.matmul(availability, sols)
        not_covered_demands = torch.clip(demands - covered_demands, min=0)
        penalty_cost = torch.matmul(penalties, not_covered_demands)

        total_cost = tot_sets_cost + penalty_cost

        # Return the cost as a dictionary with information about the total cost, the cost of the feasible solutions
        # and the ration of feasible solutions
        cost = dict()
        cost[TOTAL_COST] = total_cost
        cost[SUBOPTIMALITY_COST] = tot_sets_cost
        cost[PENALTY_COST] = torch.sum(not_covered_demands)

        # Check whether the soft constraints was violated
        if (not_covered_demands > 0).any():
            feasible = False
        else:
            feasible = True

        return cost, feasible

