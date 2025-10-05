"""
    Hamming distance loss function.
"""

import torch

########################################################################################################################


class HammingLoss(torch.nn.Module):

    @staticmethod
    def forward(sol_hat: torch.Tensor, sol_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamming distance between a predicted solution and a true solution corresponding
        to a training instance.
        :param sol_hat: torch.Tensor; the predicted solution.
        :param sol_true: torch.Tensor; the true solution.
        :return: torch.Tensor; the Hamming distance between the predicted solution and the true solution.
        """

        errors = sol_hat * (1.0 - sol_true) + (1.0 - sol_hat) * sol_true

        return errors.mean(dim=0)
