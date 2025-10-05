"""
    Predict-then-optimize approach that minimizes the MSE between prediction and target values.
"""


import torch
from torch import nn
from dfl_module import DFLModule

from typing import Tuple, Union, Dict

########################################################################################################################


class MSEModule(DFLModule):

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        A training step on a single batch.
       :param batch: tuple; input, target, scaled target, optimal solution, solver parameters and instance-specific
                     optimization problem parameters.
        :param batch_idx: int; the index of the batch in the training set.
        :return: torch.Tensor; the value of the loss function.
        """

        # Unpack the batch
        x, y, scaled_y, sol_true, solve_params, opt_model_params = batch

        # Make predictions and remove the fake batch dimension
        y_hat = self(x).squeeze()

        # The loss is the MSE between true and predicted target values
        loss = nn.MSELoss(reduction='mean')(y_hat.view(y.shape), y)

        return loss/len(y)
