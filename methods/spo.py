"""
    Implementation of SPO [1].
    [1] Elmachtoub, A. N., & Grigas, P. (2021). Smart “predict, then optimize”. Management Science.
"""

import torch
from torch import nn
import warnings

from dfl_module import DFLModule
from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import LINEAR_OBJ, QUADRATIC_OBJ, NONLINEAR_OBJ

from typing import Any, Dict, Tuple, Union

########################################################################################################################


class SPOModule(DFLModule):
    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 annealer=None,
                 obj_type: str = None,
                 lr: float = 1e-1,
                 alpha: int = 2,
                 monitor: str = None,
                 min_delta: float = 0):
        """
        :param net: torch.nn.Module; the predictive model.
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the optimization
               problem to solve.
        :param annealer; not used by this class.
        :param obj_type: str; the type of objective function of the optimization problem (linear, quadratic, etc...).
        :param lr: float; the learning rate.
        :param alpha: int; the alpha parameter as described in [1].
        :param monitor: str; the metric to monitor for early stopping.
        :param min_delta: float; the min improvement required to prevent early stopping.
        """

        super().__init__(net=net,
                         optimization_problem=optimization_problem,
                         annealer=annealer,
                         lr=lr,
                         monitor=monitor,
                         min_delta=min_delta)

        # FIXME: remove if/else statements with dict
        if obj_type == LINEAR_OBJ:
            self._method = SPO.apply
        elif obj_type == QUADRATIC_OBJ:
            self._method = QuadraticSPO.apply
        elif obj_type is None:
            warnings.warn("You haven't specified any objective function type")
        else:
            raise Exception("SPO supports only linear and quadratic  objective functions")

        self._alpha = alpha
        self._obj_type = obj_type

    @property
    def obj_type(self):
        """
        :return: str; the type of objective function of the optimization problem (linear, quadratic, etc...).
        """
        return self._obj_type

    @property
    def alpha(self) -> int:
        return self._alpha

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
            super(SPOModule, self).training_step(batch, batch_idx)

        # Make predictions and remove the fake batch dimension
        y_hat = self(x).squeeze()

        # Initialize the loss value
        loss = 0

        # For each element of the batch compute the loss function
        for idx in range(len(y)):

            # Convert it to tensor so as we give it as input to the autograd function
            alpha = torch.tensor(self._alpha)

            # There is no true loss for SPO
            loss += \
                self._method(y_hat[idx],
                             y[idx],
                             sol_true[idx],
                             self._optimization_problem,
                             alpha,
                             solve_params,
                             opt_prob_params[idx])

        return loss/len(y)

########################################################################################################################


class SPO(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                input_features: torch.Tensor,
                y_true: torch.Tensor,
                sol_true: torch.Tensor,
                optimization_problem: OptimizationProblem,
                alpha: int,
                solve_params: Dict,
                opt_prob_params: Dict):

        """
        The forward pass computes and stores the solution for the SPO-perturbed cost vector (for the backward
        pass); in SPO there is no true forward step.

        :param ctx: any type; the context object of the Autograd function.
        :param input_features: torch.Tensor; the predictions.
        :param y_true: torch.Tensor; the true cost vector
        :param sol_true: torch.Tensor; the true solution
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem;
               the parameterized optimization problem to solve.
        :param alpha: int; the alpha parameter as described in [1].
        :param solve_params: dict; optional parameters for the solver call.
        :param opt_prob_params: dict; the current optimization problem instance parameters.
        :return: torch.Tensor; the regret of the predicted cost vector with respect to the ground-truth cost vector.
        """

        # Compute the optimization direction
        mm = 1 if optimization_problem.is_minimization_problem else -1

        # Compute the SPO perturbated solution
        sol_spo = \
            optimization_problem.solve_from_torch(alpha * input_features - y_true,
                                                  opt_prob_params=opt_prob_params,
                                                  **solve_params)

        # Save information for the backward step
        ctx.save_for_backward(sol_spo, sol_true, torch.tensor(mm), alpha)

        # The SPO has no true loss since we directly compute the sub-gradient; we return a dummy value of 1
        return torch.tensor(1.0)

    @staticmethod
    def backward(ctx: Any, grad_output: Any):
        """
        The backward step of the algorithm.
        :param ctx: any type; the context object of the Autograd function.
        :param grad_output: torch.Tensor; each argument is the gradient w.r.t the given output.
        :return: tuple of torch.Tensor; each returned value should be the gradient w.r.t. the corresponding input. If an
                 input is not a Tensor or is a Tensor not requiring grads, you can just pass None as a gradient for that
                 input.
        """

        # Unpack values saved in the context
        sol_spo, sol_true, mm, alpha = ctx.saved_tensors

        # Compute the SPO sub-gradient as described in [1]
        spo_sub_grad = int(mm) * alpha * (sol_true - sol_spo)

        # FIXME: find an elegant way to return None gradient for inputs that do not require it
        return spo_sub_grad, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any):
        raise NotImplementedError()

########################################################################################################################


class QuadraticSPO(torch.autograd.Function):
    """
    Since SPO can work for optimization problems with convex objective functions, here we propose an extension that
    can deal with quadratic objective functions.
    """

    # FIXME: this code is repeated; I tried to make QuadraticSPO a parent class but it can not be instantiated
    #  (all its methods are static)

    @staticmethod
    def forward(ctx: Any,
                input_features: torch.Tensor,
                y_true: torch.Tensor,
                sol_true: torch.Tensor,
                optimization_problem: OptimizationProblem,
                alpha: int,
                solve_params: Dict,
                opt_prob_params: Dict):
        """
        The forward pass computes and stores the solution for the SPO-perturbed cost vector (for the backward
        pass); in SPO there is no true forward step.

        :param ctx: any type; the context object of the Autograd function.
        :param input_features: torch.Tensor; the predictions.
        :param y_true: torch.Tensor; the true cost vector
        :param sol_true: torch.Tensor; the true solution
        :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem;
               the parameterized optimization problem to solve.
        :param alpha: int; the alpha parameter as described in [1].
        :param solve_params: dict; optional parameters for the solver call.
        :param opt_prob_params: dict; the current optimization problem instance parameters.
        :return: torch.Tensor; the regret of the predicted cost vector with respect to the ground-truth cost vector.
        """

        # Compute the optimization direction
        mm = 1 if optimization_problem.is_minimization_problem else -1

        # Compute the SPO perturbated solution
        sol_spo = \
            optimization_problem.solve_from_torch(alpha * input_features - y_true,
                                                  opt_prob_params=opt_prob_params,
                                                  **solve_params)

        # Save information for the backward step
        ctx.save_for_backward(sol_spo, sol_true, torch.tensor(mm), alpha)

        return torch.tensor(1.0)

    @staticmethod
    def backward(ctx: Any, grad_output: Any):
        """
        The backward step of the algorithm.
        :param ctx: any type; the context object of the Autograd function.
        :param grad_output: torch.Tensor; each argument is the gradient w.r.t the given output.
        :return: tuple of torch.Tensor; each returned value should be the gradient w.r.t. the corresponding input. If an
                 input is not a Tensor or is a Tensor not requiring grads, you can just pass None as a gradient for that
                 input.
        """

        sol_spo, sol_true, mm, alpha = ctx.saved_tensors

        # Here we compute the sub-gradient for quadratic objective functions
        # We compute the difference of the SPO and true solution, similarly to linear objective function, but we also
        # consider quadratic terms

        # Example: we have 3 decision variables x1, x2, x3, x4
        # We have to compute the following dot product:
        # | x1 x1 x1 x1 |   | x1 x2 x3 x4 |   | x1*x1 x1*x2 x1*x3 x1*x4 |
        # | x2 x2 x2 x2 | * | x1 x2 x3 x4 | = | x2*x1 x2*x2 x2*x3 x2*x4 |
        # | x3 x3 x3 x3 |   | x1 x2 x3 x4 |   | x3*x1 x3*x2 x3*x3 x3*x4 |
        # | x4 x4 x4 x4 |   | x1 x2 x3 x4 |   | x4*x1 x4*x2 x4*x3 x4*x4 |

        # We compute the above operation in vectorized form (matrix are unrolled) for both the SPO and true solution
        # and then compute the difference; in the code we call them first and second term

        sol_dim = len(sol_true)

        # Sanity check
        assert len(sol_spo) == sol_dim, "True and SPO solution must have the same dimension"

        # Here we compute the computation above for the SPO solution
        first_term_sol_spo = sol_spo.repeat_interleave(sol_dim)
        second_term_sol_spo = sol_spo.repeat(sol_dim)
        dot_prod_sol_spo = first_term_sol_spo * second_term_sol_spo

        # Here we compute the computation above for the true solution
        first_term_sol_true = sol_true.repeat_interleave(sol_dim)
        second_term_sol_true = sol_true.repeat(sol_dim)
        dot_prod_sol_true = first_term_sol_true * second_term_sol_true

        # Compute the SPO sub-gradient
        spo_sub_grad = int(mm) * alpha * (dot_prod_sol_true - dot_prod_sol_spo)

        # The SPO has no true loss since we directly compute the sub-gradient; we return a dummy value of 1
        return spo_sub_grad, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any):
        raise NotImplementedError()
