"""
    Abstract class that envelops an optimization problem.
"""

from abc import abstractmethod
import numpy as np
import torch

from typing import Tuple, Dict

########################################################################################################################


class OptimizationProblem:
    """
    Abstract class representing a parameterized optimization problem, as well as a way to solve it.
    """

    def solve_from_torch(self,
                         y_torch: torch.Tensor,
                         opt_prob_params: torch.Tensor,
                         return_runtime: bool = False,
                         **kwargs) -> Tuple[torch.Tensor]:
        """
        Solve the problem for a given prediction vector.

        :param y_torch: torch.Tensor; prediction vector.
        :param return_runtime: bool; if True, return also the runtime.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: torch.Tensor, float; a vector of decision variable values as a PyTorch Float tensor and the runtime to
                 compute the solution.
        """

        # Convert torch tensors to numpy arrays
        y_numpy = y_torch.detach().numpy()
        opt_prob_params = {key: opt_prob_params[key] for key in opt_prob_params.keys()}

        # Compute the optimal solution and convert to torch tensor
        solution, runtime = self.solve(y_numpy, opt_prob_params, return_runtime=True, **kwargs)
        solution = torch.from_numpy(solution).float()

        if return_runtime:
            return solution, runtime
        else:
            return solution

    @property
    @abstractmethod
    def obj_type(self):
        """
        :return: str; the type of objective function of the optimization problem (linear, quadratic, etc...).
        """

        raise NotImplementedError()

    @abstractmethod
    def solve(self,
              y: np.ndarray,
              opt_prob_params: Dict,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the problem for a given prediction vector.
        :param y: numpy.ndarray; the prediction vector.
        :param opt_prob_params: dict; the keys are the parameter names and the values are the corresponding torch.Tensor.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        raise NotImplementedError()

    # FIXME: shall this function work for batches or single vector or both? If both y and sols are 2-dimensional
    #  torch.matmul computes the matrix product
    @abstractmethod
    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params: Dict) -> Dict:
        """
        Compute the objective value given the predictions and the solution. In this case the predictions are not needed
        to compute the objective value but we want to be consistent with the function signature.
        :param y: torch.Tensor; the predictions.
        :param sols: torch.Tensor; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: dict; we keep track of the total, suboptimality and violation.
        """

        raise NotImplementedError()
