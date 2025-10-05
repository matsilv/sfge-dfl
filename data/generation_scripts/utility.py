"""
    Utility methods shared across all the data generation process.
"""

import random
import numpy as np

########################################################################################################################


def bernoulli(p: float) -> int:
    """
    Bernoulli probability distribution.
    :param p: float; probability of success.
    :return: int; 1 if success, 0 otherwise
    """

    assert 0 < p < 1, "The probability of success must be in ]0, 1["

    if random.random() <= p:
        return 1
    else:
        return 0

########################################################################################################################


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

########################################################################################################################


def set_seeds(seed: int):
    """
    Set random seeds to ensure reproducibility.
    :param seed:
    :return:
    """

    random.seed(seed)
    np.random.seed(seed)