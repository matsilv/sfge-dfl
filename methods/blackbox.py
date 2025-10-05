"""
Implementation of Blackbox [1].

[1] Vlastelica, M., Paulus, A., Musil, V., Martius, G., Rolínek, M. (2019).
Differentiation of Blackbox Combinatorial Solvers.
"""

import torch
from torch import nn

from dfl_module import DFLModule
from losses import LOSS_CLASSES
from optimization_problems.optimization_problem import OptimizationProblem

from typing import Tuple, Any, Dict, Union

########################################################################################################################


class BlackboxModule(DFLModule):

    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 annealer=None,
                 lr: float = 1e-1,
                 lmbd: float = 0.001,
                 loss: str = None,
                 monitor: str = None,
                 min_delta: float = 0):
        """
        :param net: torch.nn.Module; the predictive model.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the optimization
               problem to solve.
        :param annealer; not used by this class.
        :param lmbd: float; hyperparameter controlling the trade-off between the “informativeness of the gradient” and
                            the “faithfulness to the original function”.
        :param lr: float; the learning rate.
        :param loss: str; string identifier of the loss function to be used; it can be None when the class is
                          instantiated to load a model.
        """

        super().__init__(net=net,
                         optimization_problem=optimization_problem,
                         lr=lr,
                         monitor=monitor,
                         min_delta=min_delta,
                         annealer=annealer)

        self._method = Blackbox.apply
        self._lmbd = lmbd
        self._optimization_direction = self._optimization_problem.is_minimization_problem

        # FIXME: the loss can not be None if the class is instantiated for training
        if loss is not None:
            loss_class = LOSS_CLASSES[loss]
            self._loss_function = loss_class()

    @property
    def lmbd(self) -> float:
        """
        :return: float; the lambda coefficient as described in the Blackbox paper.
        """
        return self._lmbd

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        A training step on a single batch.
        :param batch: tuple; input, target, scaled target, optimal solution, solver parameters and instance-specific
                             optimization problem parameters.
        :param batch_idx: int; the index of the batch in the training set.
        :return: torch.Tensor; the value of the loss function.
        """

        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_prob_params = \
            super(BlackboxModule, self).training_step(batch, batch_idx)

        # Make predictions and remove the fake batch dimension
        y_hat = self(x).squeeze()

        # Initialize the loss value
        loss = 0

        # For each element of the batch compute the loss function
        for idx in range(len(y)):

            # Get the values for each batch element
            y_hat_i = y_hat[idx]
            y_i = y[idx]
            solve_params_i = {k: v[idx] for k, v in solve_params.items()}
            opt_prob_params_i = opt_prob_params[idx]

            # The forward step of the Blackbox method compute the optimal solution w.r.t. the predictions
            sol_hat = \
                self._method(y_hat_i,
                             self._optimization_problem,
                             opt_prob_params_i,
                             solve_params_i,
                             self._lmbd)

            # Compute the loss function
            loss += self._loss_function(sol_hat=sol_hat,
                                        sol_true=sol_true[idx],
                                        y=y_i,
                                        is_minimization_problem=self._optimization_direction)

        return loss/len(y)

########################################################################################################################


class Blackbox(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                predictions: torch.Tensor,
                optimization_problem: OptimizationProblem,
                opt_prob_params: Dict,
                solve_params: Dict,
                lmbd: float = 0.001) -> torch.Tensor:
        """
        The forward pass computes and stores the solution for the predicted cost vector.
        :param ctx: any type; the context object of the Autograd function.
        :param predictions: torch.Tensor; the predictions.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem;
               the parameterized optimization problem to solve.
        :param opt_prob_params: dict; the current optimization problem instance parameters.
        :param solve_params: dict; optional parameters for the solver call.
        :param lmbd: float; hyperparameter controlling the trade-off between the “informativeness of the gradient” and
                     the “faithfulness to the original function”.
        :return: torch.Tensor; the solution corresponding to the predicted cost vector.
        """

        # The optimization problem direction
        mm = 1 if optimization_problem.is_minimization_problem else -1

        # Optimal solution w.r.t. the predicted parameters
        sol_hat = \
            optimization_problem.solve_from_torch(y_torch=predictions,
                                                  opt_prob_params=opt_prob_params,
                                                  **solve_params)

        # Save information for the backward step
        mm = torch.tensor(mm)
        lmbd = torch.tensor(lmbd)

        ctx.save_for_backward(predictions)
        ctx.optimization_problem = optimization_problem
        ctx.opt_prob_params = opt_prob_params
        ctx.lmbd = lmbd
        ctx.mm = mm
        ctx.sol_hat = sol_hat

        return sol_hat

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        The backward step of the algorithm.
        :param ctx: any type; the context object of the Autograd function.
        :param grad_output: torch.Tensor; each argument is the gradient w.r.t the given output.
        :return: tuple of torch.Tensor; each returned value should be the gradient w.r.t. the corresponding input. If an
                 input is not a Tensor or is a Tensor not requiring grads, you can just pass None as a gradient for that
                 input.
        """

        # Unpack values saved in the context
        input_features, *_ = ctx.saved_tensors
        mm = ctx.mm
        lmbd = ctx.lmbd
        opt_prob_params = ctx.opt_prob_params
        sol_hat = ctx.sol_hat

        # FIXME: I am not sure about this fix but it seems to work; we should take a closer look at the paper
        # Compute the perturbation as described in Algorithm 1 of the paper
        input_perturbed = input_features + int(mm) * float(lmbd) * grad_output

        # Compute the perturbated solution as described in Algorithm 1 of the paper
        sol_perturbed = \
            ctx.optimization_problem.solve_from_torch(y_torch=input_perturbed,
                                                      opt_prob_params=opt_prob_params)

        # Compute the gradient of the pertubation operation as described in Algorithm 1 of the paper
        grad_perturbed = -int(mm) * (sol_hat - sol_perturbed) / float(lmbd)

        return grad_perturbed, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any):
        raise NotImplementedError()
