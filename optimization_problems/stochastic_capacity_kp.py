"""
    Linear Knapsack problem with unknown capacity.
"""

import torch
import numpy as np
import gurobipy as gp

from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import NONLINEAR_OBJ, TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST

from typing import Tuple, Union, Dict

########################################################################################################################

PROBLEM_ID = 'stochastic_capacity_kp'

########################################################################################################################


class StochasticCapacityKnapsackProblem(OptimizationProblem):
    """
    A class representing a knapsack problem in which the item weights are parameterized.
    """

    def __init__(self, dim: int, penalty: float):
        """
        :param dim: int; the KP dimension.
        :param penalty: float; penalty for the recourse action.
        """

        assert penalty > 1, "Penalty must be greater than 1, otherwise there is no benefit in preventing the " \
                            "recourse action"

        self._obj_type = NONLINEAR_OBJ
        self._is_minimization_problem = False
        self._name = PROBLEM_ID
        self._dim = dim
        self._penalty = penalty

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

    @staticmethod
    def _convert_opt_prob_params(opt_prob_params: Dict) -> Tuple[np.ndarray]:
        """
        The optimization problem parameters are expected as a dictionary of torch.Tensor. They are converted to
        numpy.ndarray and returned.
        :param opt_prob_params: dict; the keys and values and respectively the names and values of the parameters.
        :return: tuple of np.ndarray; the parameters are returned as a tuple so ordering matters.
        """
        weights = opt_prob_params['weights']
        assert isinstance(weights, torch.Tensor)
        values = opt_prob_params['values']
        assert isinstance(values, torch.Tensor)

        weights = weights.numpy()
        values = values.numpy()

        # FIXME: ordering matters
        return weights, values

    # FIXME: shared code with StochasticWeightsKnapsackProblem
    @staticmethod
    def _first_stage_value(values: Union[np.ndarray, torch.Tensor],
                           first_stage_sol: Union[np.ndarray, torch.Tensor, gp.MVar]) -> Union[gp.LinExpr, float]:
        """
        Compute the first stage value. It is simply the matrix multiplication between the item values and the
        first-stage solution.
        :param values: np.ndarray or torch.Tensor; the item values.
        :param first_stage_sol: np.ndarray or torch.Tensor or gp.MVar; the first-stage solution.
        :return: gp.LinExpr or float; the first-stage value as either a Gurobi expression or a numeric value.
        """

        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()

        return values @ first_stage_sol

    def _second_stage_value(self,
                            values: Union[np.ndarray, torch.Tensor],
                            snd_stage_selected: Union[np.ndarray, torch.Tensor, gp.MVar],
                            snd_stage_removed: Union[np.ndarray, torch.Tensor, gp.MVar]) -> Union[gp.LinExpr, float]:
        """
        Compute the second-stage value. Items selected during second-stage have a lower value. Removing items during
        second-stage incurs in a penalty.
        :param values: np.ndarray or torch.Tensor; the item values.
        :param snd_stage_selected: np.ndarray or torch.Tensor or gp.MVar; the second-stage selected items.
        :param snd_stage_removed: np.ndarray or torch.Tensor or gp.MVar; the second-stage removed items.
        :return: gp.LinExpr or float; the first-stage value as either a Gurobi expression or a numeric value.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()

        return 1 / self._penalty * values @ snd_stage_selected - self._penalty * values @ snd_stage_removed

    def solve(self,
              y: np.ndarray,
              opt_prob_params: Dict,
              solve_to_optimality: bool = False,
              time_limit: int = 30,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the knapsack problem for a given capacity value.
        :param y: numpy.ndarray; the capacity value.
        :param opt_prob_params:
        :param solve_to_optimality: bool; if true, we check whether the problem was solved to optimality; otherwise we
               check whether the solver reached timeout as well.
        :param time_limit: int; force a timeout for complex instances.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        assert isinstance(y, np.ndarray)

        if len(y) == 0:
            y = np.expand_dims(y, 0)

        weights, values = self._convert_opt_prob_params(opt_prob_params)

        y = np.round(y, decimals=0)
        y = np.clip(y, a_min=0, a_max=np.inf)

        # Create the Gurobi model
        model = gp.Model()

        # Suppress Gurobi output
        model.setParam('OutputFlag', 0)
        model.setParam('Timelimit', time_limit)

        # If a single capacity value is given then we solve the problem in a deterministic fashion and add a fake
        # scenarios dimension...
        if len(y.shape) == 1:
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="x")
            model.addConstr(weights @ x <= y, name='Packing constraint')
            model.setObjective(self._first_stage_value(values, x), gp.GRB.MAXIMIZE)

        # ...otherwise solve the problem with the Sample Average Approximation algorithm by sampling multiple scenarios
        else:
            # Sanity check: first dimension is the number of scenarios whereas the second on is the number of items
            # (products)
            assert len(y.shape) == 2, "A 2-dimensional array is expected"
            num_scenarios = y.shape[0]
            capacity = y

            # First-stage decision variables
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="x")

            # Second-stage decisions
            # Selected items
            u_plus = [model.addMVar(shape=self._dim,
                                    vtype=gp.GRB.BINARY,
                                    name=f"u_plus_{omega}") for omega in range(num_scenarios)]
            # Removed items
            u_minus = [model.addMVar(shape=self._dim,
                                     vtype=gp.GRB.BINARY,
                                     name=f"u_minus_{omega}") for omega in range(num_scenarios)]

            # Initialize the second-stage value
            second_stage_value = 0

            # Add a packing constraints for each scenario
            for omega in range(num_scenarios):
                # Packing constraints
                selected_items_capacity = weights @ x + \
                                          weights @ u_plus[omega] - \
                                          weights @ u_minus[omega]
                model.addConstr(selected_items_capacity <= capacity[omega], name=f"packing_constraints_{omega}")

                # We can only remove already selected items
                model.addConstr(x >= u_minus[omega])

                # We can only add items that have not been selected during first-stage
                model.addConstr(x + u_plus[omega] <= 1)

                # Update the second-stage value
                second_stage_value += self._second_stage_value(values, u_plus[omega], u_minus[omega])

            second_stage_value = 1 / num_scenarios * second_stage_value

            # Define the objective function
            model.setObjective(self._first_stage_value(values, x) + second_stage_value, gp.GRB.MAXIMIZE)

        # Solve the model
        model.optimize()

        # Sanity check
        if solve_to_optimality:
            assert model.status == gp.GRB.OPTIMAL
        else:
            assert model.status in [gp.GRB.TIME_LIMIT, gp.GRB.OPTIMAL]

        # print(model.Runtime)

        return x.x, model.Runtime

    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params) -> Dict:
        """
        Compute the objective value given the predictions and the solution.
        :param y: torch.Tensor; the item weights.
        :param sols: torch.Tensor; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: torch.Tensor; the objective value.
        """

        assert isinstance(y, torch.Tensor)

        weights, values = self._convert_opt_prob_params(opt_prob_params)

        y = y.numpy()
        y = np.round(y, decimals=0)
        y = np.clip(y, a_min=0, a_max=np.inf)

        sols = sols.numpy()

        # Create the Gurobi model
        model = gp.Model()

        # Suppress Gurobi output
        model.setParam('OutputFlag', 0)

        # FIXME: repeating the second-stage optimization problem definition is error-prone: we must ensure it is the
        #  same as the one defined in the "solve" method.
        # Second-stage decisions
        # Selected items
        u_plus = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="u_plus")
        # Removed items
        u_minus = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="u_minus")

        # Second-stage constraints
        model.addConstr(weights @ sols + weights @ u_plus - weights @ u_minus <= y, name=f"packing constraints")

        # We can only remove already selected items
        model.addConstr(sols >= u_minus)

        # We can only add items that have not been selected during first-stage
        model.addConstr(sols + u_plus <= 1)

        # Define the objective function
        tot_value = self._first_stage_value(values, sols) + self._second_stage_value(values, u_plus, u_minus)
        model.setObjective(tot_value, gp.GRB.MAXIMIZE)

        # Solve the model
        model.optimize()

        # Sanity check
        assert model.status == gp.GRB.OPTIMAL

        suboptimality_cost = self._first_stage_value(values, sols)
        penalty_cost = self._second_stage_value(values, u_plus.x, u_minus.x)

        if penalty_cost != 0:
            feasible = False
        else:
            feasible = True

        # Return the cost as a dictionary with information about the total cost, the cost of the feasible solutions
        # and the ration of feasible solutions
        cost = dict()
        cost[TOTAL_COST] = suboptimality_cost + penalty_cost
        cost[SUBOPTIMALITY_COST] = suboptimality_cost
        cost[PENALTY_COST] = penalty_cost

        return cost, feasible
