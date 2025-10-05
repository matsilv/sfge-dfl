"""
    Score-function gradient estimation for regret minimization.
"""

import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

from dfl_module import DFLModule
from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import TOTAL_COST, LINEAR_OBJ, QUADRATIC_OBJ

from typing import Tuple, Union, Dict

########################################################################################################################


def function_to_estimate(optimization_problem: OptimizationProblem,
                         y_i: torch.Tensor,
                         y_hat: torch.Tensor,
                         sol_true_i: torch.Tensor,
                         sol_hat: torch.Tensor,
                         opt_prob_params: torch.Tensor,
                         sce: bool = False) -> torch.Tensor:
    """
    The function to estimate with SFGE, namely the regret and eventually the SCE term.
    :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem;
                                 the regret is computed for this optimization problem.
    :param y_i: torch.Tensor; true parameters for the i-th element of the batch..
    :param y_hat: torch.Tensor; predicted parameters for the i-th element of the batch..
    :param sol_true_i: torch.Tensor; the true optimal solution for the i-th element of the batch.
    :param opt_prob_params: torch.Tensor; instance-specific optimization problem parameters.
    :param sol_hat: torch.Tensor; the optimal solution computed with the predictions, for the i-th element of the batch.
    :param sce: bool; if True, the SCE term is added to the function to estimate.
    :return: torch.Tensor; the estimated function.
    """

    # Compute the optimization direction
    mm = 1 if optimization_problem.is_minimization_problem else -1

    # Remove fake batch dimension to the torch tensors
    y_i = torch.squeeze(y_i)
    y_hat = torch.squeeze(y_hat)
    sol_true_i = torch.squeeze(sol_true_i)
    sol_hat = torch.squeeze(sol_hat)

    # True optimal cost (c v^{\star})
    optimal_cost, _ = \
        optimization_problem.get_objective_values(y=y_i,
                                                  sols=sol_true_i,
                                                  opt_prob_params=opt_prob_params)
    optimal_cost = optimal_cost[TOTAL_COST]

    # True cost of the optimal solution with the predicted costs (c \hat{v}^{\star})
    pred_sol_true_cost, _ = \
        optimization_problem.get_objective_values(y=y_i,
                                                  sols=sol_hat,
                                                  opt_prob_params=opt_prob_params)
    pred_sol_true_cost = pred_sol_true_cost[TOTAL_COST]

    # Compute the regret and keep track of it
    regret = mm * (pred_sol_true_cost - optimal_cost)

    value_to_estimate = regret

    # In the following, we compute the MAP version of the Noise Contrastive Estimation (NCE) of
    # Mulamba, Maxime, et al. "Contrastive Losses and Solution Caching for Predict-and-Optimize." 30th International
    # Joint Conference on Artificial Intelligence (IJCAI-21): IJCAI-21. International Joint Conferences on Artificial
    # Intelligence, 2021.

    if sce:
        # Optimal solution cost with the predicted params (\hat{c} v^{\star})
        opt_sol_pred_cost, _ = \
            optimization_problem.get_objective_values(y=y_hat,
                                                      sols=sol_true_i,
                                                      opt_prob_params=opt_prob_params)
        opt_sol_pred_cost = opt_sol_pred_cost[TOTAL_COST]

        # Predicted cost of the optimal solution with the predicted costs (\hat{c} \hat{v}^{\star})
        pred_sol_pred_cost, _ = \
            optimization_problem.get_objective_values(y=y_hat,
                                                      sols=sol_hat,
                                                      opt_prob_params=opt_prob_params)

        pred_sol_pred_cost = pred_sol_pred_cost[TOTAL_COST]

        # Compute the sceterm
        sce_term = mm * (opt_sol_pred_cost - pred_sol_pred_cost)

        # For linear and quadratic problems we compute of (\hat{c} - c) version of the MAP loss...
        if optimization_problem.obj_type in [LINEAR_OBJ, QUADRATIC_OBJ]:
            value_to_estimate = value_to_estimate + sce_term
        else:
            # ... whereas for non-linear problems we compute the original version
            value_to_estimate = sce_term

    return value_to_estimate

########################################################################################################################


def standarize_batch_values(batch_values: np.ndarray) -> np.ndarray:
    """
    Standarize a batch of values the same way authors of Garage library do:
    # https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/tf/_functions.py#L446.
    :param batch: the batch of values.
    :type batch: np.ndarray
    """

    # Compute the variance
    variance = np.var(batch_values)
    # Compute the reciprocal of the standard deviation
    var_rec_sqrt = 1 / np.sqrt(variance + 1e-8)
    # Compute the mean
    mean_val = np.mean(batch_values)
    # Compute standardization
    batch_values = batch_values * var_rec_sqrt - mean_val * var_rec_sqrt

    return batch_values

########################################################################################################################


class SFGEModule(DFLModule):
    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 annealer,
                 # FIXME: parameters have default just to allow compatibility of the loading checkpoint method with
                 #  DFLModule
                 sce_loss: bool = False,
                 lr: float = 1e-1,
                 monitor: str = None,
                 min_delta: float = 0,
                 std_batch_vals: bool = True):
        """
        :param net: torch.nn.Module; the neural model.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the optimization
               problem to solve.
        :param annealer: annealing of some parameters (e.g. the std dev of the Gaussian distribution).
        :param sce_loss: bool; if True, add the SCE term to the estimated function.
        :param lr: float; the learning rate.
        :param monitor: str; the metric to monitor for early stopping.
        :param min_delta: float; the min improvement required to prevent early stopping.
        :param std_batch_vals: bool; if True, the values of the function whose gradient is estimated (e.g. the regret)
               are standardized; this is a good practise to reduce variance and speed-up training.
        """

        super().__init__(net=net,
                         optimization_problem=optimization_problem,
                         annealer=annealer,
                         lr=lr,
                         monitor=monitor,
                         min_delta=min_delta)

        self._sce_loss = sce_loss
        self._std_batch_vals = std_batch_vals

    @property
    def sce_loss(self) -> bool:
        """
        :return: bool; whether the SCE term is added to the regret in the function estimation.
        """
        return self._sce_loss

    def distr(self, input_features: torch.Tensor) -> torch.distributions.Distribution:
        """
        Compute the parametrized distribution.
        :param input_features: torch.Tensor;
        :return: torch.distributions.Distribution; the parametrized distribution.
        """

        return self.net.distr(input_features)

    def forward(self,
                input_features: torch.Tensor,
                # FIXME: we set the default to False to inherit 'validation_step' from 'DFLModule'
                sample: bool = False,
                # FIXME: when using a Gaussian distribution, to have a scale-independent std dev, we need both the
                #        scaled and unscaled predictions
                return_scaled: bool = False):
        """
        Override the forward method of the PyTorch Module.
        :param input_features: torch.Tensor; the input tensor.
        :param sample: bool; True if you want to sample the predictions, False if you want to act greedily.
        :param return_scaled: bool; if True, return both the scaled and unscaled predictions; else return only the
                                    unscaled predictions.
        :return:
        """
        return self.net(x=input_features, sample=sample, return_scaled=return_scaled)
    
    def _batch_predictions(self,
                           batch: Tuple,
                           batch_idx: int) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        """
        Make predictions and compute the regret on a batch of samples.
        :param batch: the batch is a tuple with the input and target features, the scaled target features, 
        the optimal solution, the solver parameters and the optimization problem parameters.
        :type batch: Tuple.
        :param batch_idx: the index of the batch in the dataset.
        :type batch_idx: int.
        """
        
        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_prob_params = \
            super(SFGEModule, self).training_step(batch, batch_idx)

        # Keep track of the batch regrets and predictions
        batch_regrets = list()
        batch_y_hat = list()

        # First of all, we compute the regret for a given predictions; we do not compute the gradient of this operations
        # but we need the regret since it is the estimated function
        with torch.no_grad():

            # For each element of the batch...
            for idx in range(len(y)):

                # Get current predictions and true target values, and solver parameters
                solve_params_i = {k: v[idx] for k, v in solve_params.items()}
                opt_prob_params_i = opt_prob_params[idx]

                # FIXME: 'return_scaled' is useful only for the 'MultiVariateGaussianModule'
                # Sample a predictions; we need both the scaled and unscaled prediction
                y_hat, scaled_y_hat = self(x[idx], sample=True, return_scaled=True)

                # FIXME: these instructions are probably useless
                y_hat = y_hat.detach()
                scaled_y_hat = scaled_y_hat.detach()

                # Keep track the unscaled prediction
                batch_y_hat.append(y_hat.numpy())

                # Optimal solution with the predicted costs
                sol_hat = \
                    self._optimization_problem.solve_from_torch(y_torch=scaled_y_hat,
                                                                opt_prob_params=opt_prob_params_i,
                                                                **solve_params_i)

                # The regret is the function we want to minimize via SFGE
                regret = \
                    function_to_estimate(optimization_problem=self._optimization_problem,
                                         y_i=y[idx],
                                         y_hat=scaled_y_hat,
                                         sol_true_i=sol_true[idx],
                                         sol_hat=sol_hat,
                                         opt_prob_params=opt_prob_params_i,
                                         sce=self._sce_loss)

                batch_regrets.append(regret)

        # From list to numpy.array
        batch_regrets = np.asarray(batch_regrets)
        batch_y_hat = np.asarray(batch_y_hat)
        batch_y_hat = torch.from_numpy(batch_y_hat)

        # Get the distribution
        distr = self._net.distr(x)

        # Compute the log-probability of the predictions
        log_prob = distr.log_prob(batch_y_hat)

        return batch_y_hat, batch_regrets, log_prob

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

        batch_y_hat, batch_regrets, log_prob = self._batch_predictions(batch=batch, batch_idx=batch_idx)

        if self._std_batch_vals:
            std_batch_regret = standarize_batch_values(batch_regrets)
        else:
            std_batch_regret = batch_regrets.copy()

        # Convert to torch.Tensor
        std_batch_regret = torch.from_numpy(std_batch_regret)

        # The score function is \nabla_{\theta} \log p(\hat{y}; \theta)
        # Here we compute a loss function whose gradient (computed with automatic differentation of PyTorch) is the
        # score function gradient estimators (but we are not explicitly computing the SFGE!)
        # entropy = torch.mean(-log_prob)
        loss = torch.multiply(-log_prob, -std_batch_regret)
        loss = torch.mean(loss)

        return loss

########################################################################################################################


