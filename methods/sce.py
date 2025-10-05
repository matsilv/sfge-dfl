"""
    Self-contrastive estimation (SCE) as described in Mulamba, Maxime, et al. "Contrastive Losses and Solution Caching
    for Predict-and-Optimize." 30th International Joint Conference on Artificial Intelligence (IJCAI-21): IJCAI-21.
    International Joint Conferences on Artificial Intelligence, 2021.
"""

import torch
from torch import nn
from dfl_module import DFLModule

from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import TOTAL_COST

from typing import Tuple, Union, Dict

########################################################################################################################


class SCEModule(DFLModule):
    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 # FIXME: annealer argument is introduced to allow compatibility with the DFL module
                 annealer=None,
                 lr: float = 1e-1,
                 monitor: str = None,
                 min_delta: float = 0):
        """
        :param net: torch.nn.Module; the neural model.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the optimization
               problem to solve.
        :param annealer; not used by this class.
        :param lr: float; the learning for the Adam optimizer.
        :param monitor: str; the metric to monitor for early stopping.
        :param min_delta: float; the min improvement required to prevent early stopping.
        """

        super().__init__(net=net,
                         optimization_problem=optimization_problem,
                         annealer=annealer,
                         lr=lr,
                         monitor=monitor,
                         min_delta=min_delta)

    def training_step(self,
                      batch: Tuple,
                      batch_idx: int) -> torch.Tensor:
        """
        A training step on a single batch.
        :param batch: tuple; input, target, scaled target, optimal solution, solver parameters and instance-specific
                     optimization problem parameters.
        :param batch_idx: int; the index of the batch in the training set.
        :return: torch.Tensor; the value of the loss function.
        """

        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_prob_params = \
            super(SCEModule, self).training_step(batch, batch_idx)

        # Make predictions (NB: we keep track of the gradient)
        y_hat = self(x)

        # Keep track of the loss on the whole batch
        loss = 0

        # For each element of the batch
        for idx in range(len(y)):

            # Get solver parameters and instance-specific optimization problem parameters
            solve_params_i = {k: v[idx] for k, v in solve_params.items()}
            opt_prob_params_i = opt_prob_params[idx]

            # Compute the optimization direction
            mm = 1 if self._optimization_problem.is_minimization_problem else -1

            # Compute the optimal solution with the predicted parameters (\hat{v}^{\star} in the paper)
            # When computing \hat{v}, we are not keeping track of the gradient since solving the optimization problem is
            # non-differentiable (inside the 'solve_from_torch' method we detach the tensor from the graph)
            sol_hat = \
                self._optimization_problem.solve_from_torch(y_hat[idx],
                                                            opt_prob_params=opt_prob_params_i,
                                                            **solve_params_i)

            # Compute the cost of the optimal solution with the predicted parameters w.r.t. the predicted parameters
            # (f( \hat{v}^{\star}, m(\omega, x) ) in the paper)
            f_sol_true_y_hat, _ = \
                self._optimization_problem.get_objective_values(y=y_hat[idx],
                                                                sols=sol_true[idx],
                                                                opt_prob_params=opt_prob_params_i)
            f_sol_true_y_hat = f_sol_true_y_hat[TOTAL_COST]

            # Compute the cost of the true optimal solution w.r.t. the predicted parameters (f( v^{\star}, m(\omega, x)
            # ) in the paper)
            f_sol_hat_y_hat, _ = \
                self._optimization_problem.get_objective_values(y=y_hat[idx],
                                                                sols=sol_hat,
                                                                opt_prob_params=opt_prob_params_i)
            f_sol_hat_y_hat = f_sol_hat_y_hat[TOTAL_COST]

            # L_{MAP} loss as defined in equations (6) of the paper
            loss += mm * (f_sol_true_y_hat - f_sol_hat_y_hat)

        return loss / len(batch)
