"""
    DFL main module.
"""
import logging
import os.path
import warnings

import torch
from torch import nn
import pytorch_lightning as pl
from abc import abstractmethod
import numpy as np

from optimization_problems import TOTAL_COST, PENALTY_COST
from optimization_problems.optimization_problem import OptimizationProblem

from typing import Tuple, Dict, Union

from utils.probabilistic_models import MultivariateGaussianModule


########################################################################################################################


class DFLModule(pl.LightningModule):
    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 annealer,
                 monitor: str = None,
                 min_delta: float = 0,
                 lr: float = 1e-1):
        """
        :param net: torch.nn.Module; the predictive model.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the optimization
               problem to solve.
        :param annealer; not used by this class.
        :param monitor: str; the metric to monitor for early stopping.
        :param min_delta: float; the min improvement required to prevent early stopping.
        :param lr: float; the learning rate.
        """
        super().__init__()
        self._hyperparams = dict()

        self._net = net
        self._lr = lr
        self._hyperparams['lr'] = lr
        self._optimization_problem = optimization_problem
        self._monitor = monitor
        self._hyperparams['monitor'] = monitor
        self._min_delta = min_delta
        self._hyperparams['min_delta'] = min_delta
        self._annealer = annealer
        self._anneal_params = list()
        self._exp_filepath = None

    @property
    def net(self) -> nn.Module:
        return self._net

    @property
    def learning_rate(self) -> float:
        return self._lr

    @property
    def optimization_problem(self) -> OptimizationProblem:
        return self._optimization_problem

    @property
    def monitor(self) -> str:
        return self._monitor

    @property
    def min_delta(self) -> float:
        return self._min_delta

    @property
    def exp_filepath(self) -> str:
        return self._exp_filepath

    @exp_filepath.setter
    def exp_filepath(self, filepath: str):
        self._exp_filepath = filepath
    
    def get_hyperparams(self) -> Dict:
        return self._hyperparams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        An inference step of the neural model.
        :param x: torch.Tensor; the input features.
        :return: torch.Tensor; the prediction.
        """
        return self.net(x)

    def training_step(self, batch: Tuple, batch_idx: int):
        """
        A training step on a single batch.
        :param batch: tuple; input, target, scaled target, optimal solution and solver parameters.
        :param batch_idx: int; the index of the batch in the training set.
        :return:
        """

        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_model_params = batch
        opt_model_params = [{key: opt_model_params[key][j] for key in opt_model_params.keys()} for j in range(len(y))]

        return x, y, scaled_y, sol_true, solve_params, opt_model_params

    def on_train_epoch_end(self) -> None:

        if isinstance(self.net, MultivariateGaussianModule):
            log_std_dev_savepath = os.path.join(self.exp_filepath, 'log-std-dev')
            if not os.path.exists(log_std_dev_savepath):
                os.makedirs(log_std_dev_savepath)

            if isinstance(self.net._log_std_dev, torch.nn.Parameter):
                np.save(f'{log_std_dev_savepath}/epoch-{self.current_epoch}', self.net._log_std_dev.detach().numpy())

        if self._annealer is not None:
            annealing_factor = self._annealer.get_annealing_factor(self.current_epoch)

            # FIXME: bad code here; maybe a wrapper for the neural architecture?
            for param in self._net.anneal_params:
                param.data = annealing_factor * param.data

    def validation_step(self,
                        batch: Tuple,
                        batch_idx: int,
                        log: bool = True,
                        testing: bool = False) -> Dict:
        """
        A validation step on a single batch.
        :param batch: tuple; input, target, scaled target, optimal solution and solver parameters.
        :param batch_idx: int; the index of the batch in the training set.
        :param log: bool; True to log the results, False otherwise.
        :param testing: bool; True if in testing mode, False otherwise.
        :return: dict; the validation metrics.
        """

        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_model_params = batch

        opt_model_params = [{key: opt_model_params[key][j] for key in opt_model_params.keys()} for j in range(len(y))]

        # Sanity check
        assert len(x) == len(y) and len(x) == len(scaled_y) and len(x) == len(sol_true), \
            "The unpacked values from the batch must have the same length"

        # Compute the direction of the optimization problem
        mm = 1 if self._optimization_problem.is_minimization_problem else -1

        # Make predictions
        y_hat = self(x).squeeze()

        # Compute the MSE w.r.t. the true target values
        mse = nn.MSELoss(reduction='mean')(y_hat.view(y.shape), y)

        # Keep track of the regret, relative regret, relative regret of the feasible solutions and number of infeasible
        # solutions
        regret_list = list()
        relative_regret_list = list()
        feas_sol_rel_regret_list = list()
        num_infeas_sol = 0

        # For element of the batch
        for i in range(len(y)):

            # Solver arguments
            kwargs = {
                'allow_cache_sampling': False,
                'allow_cache_adding': False,
                **solve_params
            }

            # Select prediction for the current batch and reshape it according to the target values
            y_hat = y_hat.view(y.shape)[i]

            # Compute the optimal solution w.r.t. the predicted parameters
            sol_hat, runtime = \
                self.optimization_problem.solve_from_torch(y_torch=y_hat,
                                                           opt_prob_params=opt_model_params[i],
                                                           return_runtime=True,
                                                           **kwargs)

            # Compute the regret as the difference between the objective functions of the optimal and predicted
            # solutions
            optimal_cost, feasibility_opt_sol = \
                self.optimization_problem.get_objective_values(y=torch.squeeze(y[i]),
                                                               sols=torch.squeeze(sol_true),
                                                               opt_prob_params=opt_model_params[i])

            sol_cost, feasibility_pred_sol = \
                self.optimization_problem.get_objective_values(y=torch.squeeze(y),
                                                               sols=torch.squeeze(sol_hat),
                                                               opt_prob_params=opt_model_params[i])

            # We distinguish between suboptimality and penalty cost
            tot_optimal_cost = optimal_cost[TOTAL_COST]
            tot_sol_cost = sol_cost[TOTAL_COST]

            # Compute regret and relative regret
            sol_regret = mm * (tot_sol_cost - tot_optimal_cost)
            sol_relative_regret = sol_regret / tot_optimal_cost

            # if sol_relative_regret < 0:
            #     print()

            # Sanity check
            assert optimal_cost[PENALTY_COST] == 0, "Penalty of the optimal solution must be 0"
            assert feasibility_opt_sol, "Optimal solution must be feasible"
            # assert sol_relative_regret >= 0

            if sol_relative_regret < 0:
                warnings.warn(f'Relative regret < 0: {sol_relative_regret}')

            # Keep track of the regret and relative regret in a list
            regret_list.append(sol_regret)
            relative_regret_list.append(sol_relative_regret)

            assert feasibility_pred_sol >= 0

            # If the solution is feasible, keep track of the relative regret...
            if feasibility_pred_sol:
                feas_sol_rel_regret_list.append(sol_relative_regret)
            # ...otherwise keep track of the number of infeasible solutions
            else:
                num_infeas_sol += 1

        # Compute average of the metrics across the batch
        regret = torch.mean(torch.tensor(regret_list))
        relative_regret = torch.mean(torch.tensor(relative_regret_list))

        if relative_regret == np.inf:
            print()

        # FIXME: sometimes no feasible solution is found
        if len(feas_sol_rel_regret_list) == 0:
            feas_sol_rel_regret = torch.tensor([0])
        else:
            feas_sol_rel_regret = torch.mean(torch.tensor(feas_sol_rel_regret_list))

        num_infeas_sol = torch.tensor(num_infeas_sol)

        # FIXME: the logged string is too long
        # Log the values if required
        if log:
            prefix = 'test' if testing else 'val'
            regret_str = prefix + '_regret'
            relative_regret_str = prefix + '_relative_regret'
            mse_str = prefix + '_mse'
            cost_str = prefix + '_cost'
            optimal_cost_str = prefix + '_optimal_cost'
            feas_sol_rel_reg_str = prefix + '_feasible_solutions_relative_regret'
            num_infeas_sol_str = prefix + '_num_infeasible_solutions'
            runtime_str = prefix + '_runtime'

            self.log(mse_str, mse, prog_bar=True, on_step=False, on_epoch=True)
            self.log(regret_str, regret, prog_bar=True, on_step=False, on_epoch=True)
            self.log(relative_regret_str, relative_regret, prog_bar=True, on_step=False, on_epoch=True)
            self.log(cost_str, tot_sol_cost, prog_bar=False, on_step=False, on_epoch=True)
            self.log(optimal_cost_str, tot_optimal_cost, prog_bar=False, on_step=False, on_epoch=True)
            self.log(feas_sol_rel_reg_str, feas_sol_rel_regret, prog_bar=False, on_step=False, on_epoch=True)
            self.log(num_infeas_sol_str, num_infeas_sol, prog_bar=False, on_step=False, on_epoch=True)
            self.log(runtime_str, runtime, prog_bar=False, on_step=False, on_epoch=True)

        # Save results in a dictionary
        metrics = dict()
        metrics[mse_str] = mse
        metrics[regret_str] = regret

        return metrics

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict:
        """
        A test step on single batch. It is the same as 'validation_step'.
        :param batch: tuple; input, target, scaled target, optimal solution and solver parameters.
        :param batch_idx: int; the index of the batch in the training set.
        :return: dict; the test metrics.
        """

        # Reuse validation_step for testing
        return self.validation_step(batch, batch_idx, log=True, testing=True)

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Create Adam optimizer.
        :return: torch.optim.Adam; an instance of Adam optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return optimizer
