"""
    Probabilistic models that compute P(Y|X) where X and Y are respectively the input and target features.
"""

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Poisson
from torch.nn import Linear
import numpy as np

from utils.scale_layer import ScaleLayer

from typing import Tuple


########################################################################################################################

COVARIANCE_TYPE = ['static', 'trainable', 'contextual', 'linear_annealing']

########################################################################################################################


class ProbabilisticModelModule(Module):
    """
        Base class for the probabilistic models.
    """

    def __init__(self,
                 net: Module,
                 scale_layer: ScaleLayer):

        super(ProbabilisticModelModule, self).__init__()

        self._net = net
        self._scale_layer = scale_layer
        self._anneal_params = list()

    @property
    def anneal_params(self):
        return self._anneal_params

    def forward(self,
                x: torch.Tensor,
                sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overriding of 'torch.nn.Module.forward' method.
        :param x: torch.Tensor; input features.
        :param sample: bool; if False, we compute the greedy predictions.
        :return: tuples of 2 torch.Tensor; standardized and non predictions.
        """

        raise NotImplementedError()

    def distr(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the parametrized distribution.
        :param x: torch.Tensor; the input features.
        :return: torch.distributions.Distribution; the parametrized distribution.
        """

        raise NotImplementedError()

########################################################################################################################


class MultivariateGaussianModule(ProbabilisticModelModule):
    """
        PyTorch module that compute a multivariate gaussian distribution, namely P(Y|X) = N(mu, Sigma | X) where mu is
        the vector of means and Sigma is a diagonal covariance matrix.
    """
    def __init__(self,
                 net: Module,
                 scale_layer: ScaleLayer,
                 input_shape: int,
                 output_shape: int,
                 init_std_dev_val: float,
                 covariance_type: str):
        """

        :param net: torch.Module; the neural model that computes the mean of the distribution.
        :param scale_layer: utils.ScaleLayer; PyTorch custom layer that computes the inverse standardization.
        :param output_shape: int; number of input features.
        :param output_shape: int; number of output features.
        :param init_std_dev_val: float; initial value of the logarithm of the covariance matrix (same for all the
                                        elements of the matrix).
        :param covariance_type: str; whether the covariance matrix is static, trainable or it is contextual.
        """

        super(MultivariateGaussianModule, self).__init__(net, scale_layer)

        assert covariance_type in COVARIANCE_TYPE, "Covariance type is not supported"
        self._covariance_type = covariance_type

        # Create the (logarithm of the) standard deviation of the distribution; make it PyTorch Parameter
        if self._covariance_type in ['static', 'linear_annealing']:
            init_std_dev = np.ones(shape=(output_shape,), dtype=np.float32) * init_std_dev_val
            self._log_std_dev = Parameter(torch.tensor(init_std_dev), requires_grad=False)

            self._anneal_params.append(self._log_std_dev)

        elif self._covariance_type == 'trainable':
            init_std_dev = np.ones(shape=(output_shape,), dtype=np.float32) * init_std_dev_val
            self._log_std_dev = Parameter(torch.tensor(init_std_dev), requires_grad=True)

        elif self._covariance_type == 'contextual':
            self._log_std_dev = Linear(in_features=input_shape, out_features=output_shape)

    def forward(self,
                x: torch.Tensor,
                sample: bool = False,
                return_scaled: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overriding of 'torch.nn.Module.forward' method.
        :param x: torch.Tensor; input features.
        :param sample: bool; if False, we compute the greedy (the mean) predictions.
        :param return_scaled: bool; if True, return both the scaled and unscaled predictions; else return only the
                                    unscaled predictions.
        :return: tuples of 2 torch.Tensor; standardized and non predictions.
        """

        # Compute the mean and std dev
        mean = self._net(x)

        if self._covariance_type in ['static', 'trainable', 'linear_annealing']:
            log_std_dev = self._log_std_dev
        else:
            log_std_dev = self._log_std_dev(x)

        # Sample the prediction from the distribution if required
        if sample:
            # To prevent negative value, learn the logarithm of the standard deviation
            std_dev = torch.exp(log_std_dev)

            # We assume that each sample is independent from each other: we do not learn the upper and lower triangular
            # matrices
            covariance = torch.diag(std_dev)
            distr = MultivariateNormal(loc=mean, covariance_matrix=covariance)
            output = distr.sample()
        else:
            output = mean

        scaled_output = output
        unscaled_output = self._scale_layer(output)

        if return_scaled:
            return scaled_output, unscaled_output
        else:
            return unscaled_output

    # FIXME: repeated code from 'forward'.
    def distr(self,
              x: torch.Tensor,
              # FIXME: when using a Gaussian distribution, to have a scale-independent std dev, we need both the
              #        scaled and unscaled predictions
              return_scaled: bool = False) -> torch.distributions.MultivariateNormal:
        """
        Get the parametrized distribution.
        :param x: torch.Tensor; the input features.
        :param return_scaled: bool; if True, return both the scaled and unscaled predictions; else return only the
                                    unscaled predictions.
        :return: torch.distributions.MultivariateNormal; the parametrized multi-variate gaussian distribution.
        """

        mean = self._net(x)

        if self._covariance_type in ['static', 'trainable', 'linear_annealing']:
            log_std_dev = self._log_std_dev
            std_dev = torch.exp(log_std_dev)
            covariance = torch.diag(std_dev)
        else:
            log_std_dev = self._log_std_dev(x)
            std_dev = torch.exp(log_std_dev)
            covariance = torch.diag_embed(std_dev)

        distr = MultivariateNormal(loc=mean, covariance_matrix=covariance)

        return distr


########################################################################################################################


class PoissonModule(ProbabilisticModelModule):
    def __init__(self, net: Module, scale_layer: ScaleLayer):
        """
        :param net: torch.Module; neural model that computes the log rate of the Poisson distribution.
        :param scale_layer: utils.ScaleLayer (not used in Poisson, but kept for interface compatibility).
        """
        super(PoissonModule, self).__init__(net, scale_layer)

    def _compute_rate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute strictly positive Poisson rate from network output."""
        # return torch.exp(self._net(x))
        # return torch.nn.functional.softplus(self._net(x))
        return torch.nn.Softplus()(self._net(x))

    def forward(self, x: torch.Tensor,
                sample: bool = False,
                return_scaled: bool = False) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:

        rate = self._compute_rate(x)

        if sample:
            output = Poisson(rate).sample()
        else:
            output = rate

        if return_scaled:
            return rate, output
        else:
            return output

    def distr(self, x: torch.Tensor) -> torch.distributions.Poisson:
        rate = self._compute_rate(x)
        return Poisson(rate)
