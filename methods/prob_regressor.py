"""
    Predict-then-optimize approach that maximizes the likelihood between prediction and target values.
"""

import torch
from torch import nn

from dfl_module import DFLModule
from optimization_problems.optimization_problem import OptimizationProblem

from typing import Tuple, Union, Dict

from utils.probabilistic_models import PoissonModule


########################################################################################################################


# FIXME: ProbRegressorModule and SFGE are somehow similar; can we find a way to find a unique design?
class ProbRegressorModule(DFLModule):
    """
        Probabilistic regressor trained to minimize the negative log-likelihood.
    """

    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 annealer=None,
                 # FIXME: parameters have default to allow compatibility of the loading checkpoint method with
                 #  DFLModule
                 lr: float = 1e-1,
                 monitor: str = None,
                 min_delta: float = 0):
        """
        :param net: torch.nn.Module; the neural model.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the optimization
               problem to solve.
        :param lr: float; the learning rate.
        """

        super().__init__(net=net,
                         optimization_problem=optimization_problem,
                         lr=lr,
                         monitor=monitor,
                         min_delta=min_delta,
                         annealer=annealer)

    def distr(self, input_features: torch.Tensor) -> torch.distributions.Distribution:
        """
        Compute the parametrized distribution.
        :param input_features: torch.Tensor; the input features on which the distribution parameters depend.
        :return: torch.distributions.Distribution; the parametrized distribution.
        """

        return self.net.distr(input_features)

    def forward(self,
                input_features: torch.Tensor,
                # FIXME: we set the default to False to inherit 'validation_step' from 'DFLModule'
                sample: bool = False,
                # FIXME: when using a Gaussian distribution, to have a scale-independent std dev, we need both the
                #        scaled and unscaled predictions
                return_scaled: bool = False) -> torch.Tensor:
        """
        Override the forward method of the PyTorch Module.
        :param input_features: torch.Tensor; the input tensor.
        :param sample: bool; True if you want to sample the predictions, False if you want to act greedily.
        :param return_scaled: bool; if True, return both the scaled and unscaled predictions; else return only the
                              unscaled predictions.
        :return:
        """

        return self._net(x=input_features, sample=sample, return_scaled=return_scaled)

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
        x, y, scaled_y, sol_true, solve_params, opt_model_params = batch

        # The maximum likelihood estimation problem is equivalent to minimize the negative log likelihood
        # if isinstance(distr, torch.distributions.Poisson):
        if isinstance(self._net, PoissonModule):
            # Get from the model the distribution object
            distr = self._net.distr(x)
            # neg_log_likelihood = torch.sum(-distr.log_prob(y), dim=1)
            loss = -distr.log_prob(y).mean()
            # loss = torch.mean(neg_log_likelihood, dim=0)
            # rate = self._net.forward(x, return_scaled=False, sample=False)
            # loss = torch.nn.PoissonNLLLoss(log_input=False)(rate, y)
        else:
            # Get from the model the distribution object
            distr = self._net.distr(x)
            neg_log_likelihood = -distr.log_prob(scaled_y)
            loss = torch.mean(neg_log_likelihood, dim=0)

        return loss
