from torch import nn
from dfl_module import DFLModule
from optimization_problems.optimization_problem import OptimizationProblem
import functools
from typing import Tuple
import torch
from torch.distributions.gumbel import Gumbel
from torch.distributions.normal import Normal

_GUMBEL = 'gumbel'
_NORMAL = 'normal'
SUPPORTED_NOISES = (_GUMBEL, _NORMAL)

########################################################################################################################


def sample_noise_with_gradients(noise, shape):
    """Samples a noise tensor according to a distribution with its gradient.

    Args:
    noise: (str) a type of supported noise distribution.
    shape: torch.tensor<int>, the shape of the tensor to sample.

    Returns:
    A tuple Tensor<float>[shape], Tensor<float>[shape] that corresponds to the
    sampled noise and the gradient of log the underlying probability
    distribution function. For instance, for a gaussian noise (normal), the
    gradient is equal to the noise itself.

    Raises:
    ValueError in case the requested noise distribution is not supported.
    See perturbations.SUPPORTED_NOISES for the list of supported distributions.
    """
    if noise not in SUPPORTED_NOISES:
        raise ValueError('{} noise is not supported. Use one of [{}]'.format(
            noise, SUPPORTED_NOISES))

    if noise == _GUMBEL:
        sampler = Gumbel(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = 1 - torch.exp(-samples)
    elif noise == _NORMAL:
        sampler = Normal(0.0, 1.0)
        samples = sampler.sample(shape)
        gradients = samples

    return samples, gradients


def perturbed(func=None,
              num_samples = 10,
              sigma = 0.05,
              noise = _NORMAL,
              batched = False,
              device=None):
    """Turns a function into a differentiable one via perturbations.

    The input function has to be the solution to a linear program for the trick
    to work. For instance the maximum function, the logical operators or the ranks
    can be expressed as solutions to some linear programs on some polytopes.
    If this condition is violated though, the result would not hold and there is
    no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    Args:
    func: the function to be turned into a perturbed and differentiable one.
    Four I/O signatures for func are currently supported:
        If batched is True,
        (1) input [B, D1, ..., Dk], output [B, D1, ..., Dk], k >= 1
        (2) input [B, D1, ..., Dk], output [B], k >= 1
        If batched is False,
        (3) input [D1, ..., Dk], output [D1, ..., Dk], k >= 1
        (4) input [D1, ..., Dk], output [], k >= 1.
    num_samples: the number of samples to use for the expectation computation.
    sigma: the scale of the perturbation.
    noise: a string representing the noise distribution to be used to sample
    perturbations.
    batched: whether inputs to the perturbed function will have a leading batch
    dimension (True) or consist of a single example (False). Defaults to True.
    device: The device to create tensors on (cpu/gpu). If None given, it will
    default to gpu:0 if available, cpu otherwise.

    Returns:
    a function has the same signature as func but that can be back propagated.
    """
    # If device not supplied, auto detect
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # This is a trick to have the decorator work both with and without arguments.
    if func is None:
        return functools.partial(
            perturbed, num_samples=num_samples, sigma=sigma, noise=noise,
            batched=batched, device=device)

    @functools.wraps(func)
    def wrapper(input_tensor, *args):
        class PerturbedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input_tensor, *args):
                original_input_shape = input_tensor.shape
                # print("Input shape",original_input_shape)
                if batched:
                    if not input_tensor.dim() >= 2:
                        raise ValueError('Batched inputs must have at least rank two')
                else:  # Adds dummy batch dimension internally.
                    input_tensor = input_tensor.unsqueeze(0)
                input_shape = input_tensor.shape  # [B, D1, ... Dk], k >= 1
                # print("Input shape",input_shape)
                perturbed_input_shape = [num_samples] + list(input_shape)
                noises = sample_noise_with_gradients(noise, perturbed_input_shape)
                additive_noise, noise_gradient = tuple(
                    [noise.type(input_tensor.dtype) for noise in noises])
                additive_noise = additive_noise.to(device)
                noise_gradient = noise_gradient.to(device)
                perturbed_input = input_tensor.unsqueeze(0) + sigma * additive_noise

                # [N, B, D1, ..., Dk] -> [NB, D1, ..., Dk].
                flat_batch_dim_shape = [-1] + list(input_shape)[1:]
                perturbed_input = torch.reshape(perturbed_input, flat_batch_dim_shape)
                # Calls user-defined function in a perturbation agnostic manner.
                # print("perturbed input shape", perturbed_input.shape)
                perturbed_output = func(perturbed_input, *args)
                # [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk].
                perturbed_input = torch.reshape(perturbed_input, perturbed_input_shape)
                # Either
                #   (Default case): [NB, D1, ..., Dk] ->  [N, B, D1, ..., Dk]
                # or
                #   (Full-reduce case) [NB] -> [N, B]
                perturbed_output_shape = [num_samples, -1] + list(perturbed_output.shape)[1:]
                perturbed_output = torch.reshape(perturbed_output, perturbed_output_shape)

                forward_output = torch.mean(perturbed_output, dim=0)
                if not batched:  # Removes dummy batch dimension.
                    forward_output = forward_output[0]

                # Save context for backward pass
                ctx.save_for_backward(perturbed_input, perturbed_output, noise_gradient)
                ctx.original_input_shape = original_input_shape

                return forward_output

            @staticmethod
            def backward(ctx, dy):
                # Pull saved tensors
                original_input_shape = ctx.original_input_shape
                perturbed_input, perturbed_output, noise_gradient = ctx.saved_tensors
                output, noise_grad = perturbed_output, noise_gradient
                # Adds dummy feature/channel dimension internally.
                if perturbed_input.dim() > output.dim():
                    dy = dy.unsqueeze(-1)
                    output = output.unsqueeze(-1)
                # Adds dummy batch dimension internally.
                if not batched:
                    dy = dy.unsqueeze(0)
                # Flattens [D1, ..., Dk] to a single feat dim [D].
                flatten = lambda t: torch.reshape(t, (list(t.shape)[0], list(t.shape)[1], -1))
                dy = torch.reshape(dy, (list(dy.shape)[0], -1))  # (B, D)
                output = flatten(output)  # (N, B, D)
                noise_grad = flatten(noise_grad)  # (N, B, D)

                g = torch.einsum('nbd,nb->bd', noise_grad, torch.einsum('nbd,bd->nb', output, dy))
                g /= sigma * num_samples
                return torch.reshape(g, original_input_shape), None, None

        return PerturbedFunc.apply(input_tensor, *args)

    return wrapper


class PerturbedFunc(torch.autograd.Function):
    """Implementation of a Fenchel Young loss."""
    @staticmethod
    def forward(ctx, input_tensor, y_true, perturbed, batched, maximize, *args):
        diff = perturbed(input_tensor, *args) - y_true.type(input_tensor.dtype)
        if not maximize:
            diff = -diff
        # Computes per-example loss for batched inputs.
        if batched:
            loss = torch.sum(torch.reshape(diff, [list(diff.shape)[0], -1]) ** 2, dim=-1)
        else:  # Computes loss for unbatched inputs.
            loss = torch.sum(diff ** 2)
        ctx.save_for_backward(diff)
        ctx.batched = batched
        return loss

    @staticmethod
    def backward(ctx, dy):
        diff,  = ctx.saved_tensors
        batched = ctx.batched
        if batched:  # dy has shape (batch_size,) in this case.
            dy = torch.reshape(dy, [list(dy.shape)[0]] + (diff.dim() - 1) * [1])

        # print(dy)
        # return dy * diff, None, None, None, None, None, None # original
        return  diff, None, None, None, None, None, None


class FenchelYoungLoss(nn.Module):
    def __init__(self,
                 func = None,
                 num_samples = 1000,
                 sigma = 0.01,
                 noise = _GUMBEL,
                 batched = True,
                 maximize = True,
                 device=None):
        """Initializes the Fenchel-Young loss.

        Args:
            func: the function whose argmax is to be differentiated by perturbation.
            num_samples: (int) the number of perturbed inputs.
            sigma: (float) the amount of noise to be considered
            noise: (str) the noise distribution to be used to sample perturbations.
            batched: whether inputs to the func will have a leading batch dimension
            (True) or consist of a single example (False). Defaults to True.
            maximize: (bool) whether to maximize or to minimize the input function.
            device: The device to create tensors on (cpu/gpu). If None given, it will
            default to gpu:0 if available, cpu otherwise.
        """
        super().__init__()
        self._batched = batched
        self._maximize = maximize
        self.func = func
        self.perturbed = perturbed(func=func,
                                                num_samples=num_samples,
                                                sigma=sigma,
                                                noise=noise,
                                                batched=batched,
                                                device=device)

    def forward(self, input_tensor, y_true, opt_prob_params, solve_params, *args):
        return PerturbedFunc.apply(input_tensor, y_true, self.perturbed, self._batched, self._maximize, opt_prob_params, solve_params, *args)


def batch_solve(optimization_problem, batch_y_hat, opt_prob_params, solve_params):

    sols = []

    for _y_hat in batch_y_hat:
        sol_hat = (
            optimization_problem.solve_from_torch(y_torch=_y_hat,
                                                  opt_prob_params=opt_prob_params,
                                                  **solve_params)
        )
        sols.append(sol_hat)

    return torch.stack(sols,dim=0)


class DPOModule(DFLModule):
    def __init__(self,
                 net: nn.Module,
                 optimization_problem: OptimizationProblem,
                 # FIXME: parameters have default just to allow compatibility of the loading checkpoint method with
                 #  DFLModule
                 annealer=None,
                 sce_loss: bool = False,
                 lr: float = 1e-1,
                 monitor: str = None,
                 min_delta: float = 0):

        super().__init__(net=net,
                         optimization_problem=optimization_problem,
                         annealer=annealer,
                         lr=lr,
                         monitor=monitor,
                         min_delta=min_delta)

        self._sce_loss = sce_loss

        # TODO: See https://github.com/PredOpt/predopt-benchmarks/blob/main/Knapsack/Trainer/PO_models.py.
        # @perturbed(num_samples=10, sigma=0.05, noise='gumbel', batched=True)
        # def dpo_layer(batch_y, batch_opt_prob_params, solve_params):
        #     return batch_solve(
        #         optimization_problem=self._optimization_problem,
        #         batch_y_hat=batch_y,
        #         opt_prob_params=batch_opt_prob_params[0],
        #         solve_params=solve_params
        #     )
        #
        # self.layer = dpo_layer

        fy_solver = lambda y_, opt_prob_params_, solve_parmas: batch_solve(self.optimization_problem, y_, opt_prob_params_, solve_parmas)
        self.criterion = FenchelYoungLoss(fy_solver, num_samples=1,
                                             sigma=0.1, maximize=True, batched=True)

    @property
    def sce_loss(self) -> bool:
        """
        :return: bool; whether the SCE term is added to the regret in the function estimation.
        """
        return self._sce_loss
    #
    # def forward(self, input_features: torch.Tensor):
    #     return self.net(x=input_features)

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

        # batch_y_hat, batch_regrets, log_prob = self._batch_predictions(batch=batch, batch_idx=batch_idx)

        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_prob_params = super(DPOModule, self).training_step(batch, batch_idx)
        y_hat = self(x).squeeze()
        # sol_hat = self.layer(y_hat, opt_prob_params, solve_params)
        # loss = ((sol_true - sol_hat) * y).sum(-1).mean()

        # FIXME: Since opt_prob_params[0], we assume the optimization problem parameters are the same for all the
        #  elements in the batch. This is true in the experimental setup of the JAIR paper but in principle we could
        #  have different parameters for each element of the batch.
        loss = self.criterion(y_hat, sol_true, opt_prob_params[0], solve_params).mean()

        return loss
