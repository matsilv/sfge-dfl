"""
    Regret for linear problems.
"""

import torch

########################################################################################################################


class Regret(torch.nn.Module):

    @staticmethod
    def forward(sol_hat: torch.Tensor,
                sol_true: torch.Tensor,
                y: torch.Tensor,
                is_minimization_problem: bool,
                has_subcoefficients: bool = False,
                **kwargs) -> torch.Tensor:

        """
        Computes the regret of a predicted solution w.r.t. a true cost vector for a linear objective function.
        :param sol_hat: torch.Tensor; the predicted solution.
        :param sol_true: torch.Tensor; the true optimal solution.
        :param y: torch.Tensor; the cost vector.
        :param is_minimization_problem: bool; a Boolean denoting whether the optimization problem is a minimization
                                        problem.
        :param has_subcoefficients: bool; boolean that denotes whether the elements of y are coefficients in the
                                    objective, or 'subcoefficients', where subcoefficients still have to be pairwise
                                    multiplied to become the coefficients in the objective, i.e.
                                    y_new = [y[0]*y[1], y[2]*y[3], ...]
        :return: torch.Tensor; the regret of the predicted solution w.r.t. the true cost vector.
        """

        mm = 1 if is_minimization_problem else -1

        return mm * (sol_hat - sol_true).dot(y)
