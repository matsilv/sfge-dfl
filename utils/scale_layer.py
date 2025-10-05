"""
    Scale layer as PyTorch module.
"""

import torch
from torch import nn

########################################################################################################################


class ScaleLayer(nn.Module):
    """
        Base scale layer class.
    """

    def __init__(self, translation_term: float, multiplication_factor: float):

        super().__init__()

        self._translation_term = translation_term
        self._multiplication_factor = multiplication_factor

    @property
    def translation_term(self) -> float:
        """
        :return: float; the translation coefficient of the scaling operation.
        """
        return self._translation_term

    @property
    def multiplication_factor(self) -> float:
        """
        :return: float; the multiplication factor of the scaling operation.
        """
        return self._multiplication_factor

    def forward(self, input: torch.Tensor, transform: bool = True) -> torch.Tensor:
        """
        Override the forward method of PyTorch module.
        :param input: torch.Tensor; input tensor to the layer.
        :param transform: bool; true if you want to apply the scaling.
        :return: torch.Tensor; the transformed input tensor.
        """

        if transform:
            return (input * self._multiplication_factor) + self._translation_term
        else:
            return input

########################################################################################################################


class MockScaleLayer(nn.Module):
    def __init__(self):
        """
            Mock scale layer that does not apply any transformation.
        """
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Override the forward method of PyTorch module.
        :param input: torch.Tensor; input tensor to the layer.
        """
        return input
