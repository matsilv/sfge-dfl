"""
    Data wrapper used by the PyTorch DataLoaders.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from abc import abstractmethod

from optimization_problems.optimization_problem import OptimizationProblem

from typing import Dict, Iterable, Union, List, Tuple, Any

########################################################################################################################


def check_data_type(data: Any):
    """
    Check if the input variable is torch.Tensor or a numpy.array.
    :param data: any type; the input data to check.
    :return:
    """

    assert isinstance(data, (torch.Tensor, np.ndarray)), "data must be a torch.Tensor or a numpy.array"

########################################################################################################################


def from_numpy_to_torch(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Convert input data to torch.Tensor if it is a numpy.ndarray.
    :param data: numpy.ndarray or torch.Tensor; the input data to convert.
    :return: torch.Tensor; the converted data.
    """

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    return data

########################################################################################################################


class DataWrapper(Dataset):
    def __init__(self,
                 input_features: Union[torch.Tensor, np.ndarray],
                 target_features: Union[torch.Tensor, np.ndarray],
                 scaled_target_features: Union[torch.Tensor, np.ndarray],
                 solutions: Union[torch.Tensor, np.ndarray],
                 solve_params: Iterable[Dict] = None):
        """
        Abstract class for torch datasets wrapper.
        :param input_features: torch.Tensor or numpy.ndarray; the input features of the dataset.
        :param target_features: torch.Tensor or numpy.ndarray; the target features of the dataset.
        :param scaled_target_features: torch.Tensor or numpy.ndarray; the scaled target features of the dataset.
        :param solve_params: list of dict; the solver parameters for each instance of the dataset.
        """

        # Sanity checks
        check_data_type(input_features)
        check_data_type(target_features)
        check_data_type(scaled_target_features)
        check_data_type(solutions)

        assert len(input_features) == len(target_features) and len(input_features) == len(scaled_target_features) and\
               len(input_features) == len(solutions), \
               "input_features, target_features and scaled_target_features must have the same length"

        # If input data are numpy arrays then convert them to torch Tensor
        self._x = from_numpy_to_torch(input_features)
        self._y = from_numpy_to_torch(target_features)
        self._scaled_y = from_numpy_to_torch(scaled_target_features)
        self._solutions = from_numpy_to_torch(solutions)

        # If the solver does not require parameters, we create dummy parameters
        self._solve_params = solve_params if solve_params else [{'params': 0} for _ in self._x]

        # These are the optimization problem parameters that may change among instances
        self._opt_prob_params = None

    def __len__(self) -> int:
        return len(self.y)

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple:
        """
        Built-in method to get a single entry of the dataset.
        :param index: int; index of the entry.
        :return: tuple with input features, target features, scaled target features, optimal solution, solver parameters
                 and the index of the entry. The index is useful when the optimization problem has different parameters
                 for each entry (e.g. Knapsack with different item weights).
        """

        raise NotImplementedError()

    @property
    def x(self) -> torch.Tensor:
        """
        :return: torch.Tensor; the input features.
        """
        return self._x

    @property
    def y(self) -> torch.Tensor:
        """
        :return: torch.Tensor; the target features.
        """
        return self._y

    @property
    def scaled_y(self) -> torch.Tensor:
        """
        :return: torch.Tensor; scaled target features.
        """
        return self._scaled_y

    @property
    def solutions(self) -> torch.Tensor:
        """
        :return: torch.Tensor; the optimal solutions.
        """
        return self._solutions

    @property
    def opt_prob_params(self) -> torch.Tensor:
        """
        :return: torch.Tensor; the instance-specific optimization problem params
        """
        return self._opt_prob_params

    @property
    def solve_params(self) -> List[Dict]:
        """
        :return: list of dict; the solver parameters.
        """
        return self._solve_params

########################################################################################################################


class KnapsackDataWrapper(DataWrapper):

    def __init__(self,
                 input_features: Union[torch.Tensor, np.ndarray],
                 target_features: Union[torch.Tensor, np.ndarray],
                 scaled_target_features: Union[torch.Tensor, np.ndarray],
                 weights: Union[torch.Tensor, np.ndarray],
                 capacities: Union[torch.Tensor, np.ndarray],
                 solutions: Union[torch.Tensor, np.ndarray],
                 solve_params: Iterable[Dict] = None):
        """
        Wrapper of torch dataset.
        :param input_features: torch.Tensor or numpy.ndarray; the input features of the dataset.
        :param target_features: torch.Tensor or numpy.ndarray; the target features of the dataset.
        :param scaled_target_features: torch.Tensor or numpy.ndarray; the scaled target features of the dataset.
        """

        super().__init__(input_features=input_features,
                         target_features=target_features,
                         scaled_target_features=scaled_target_features,
                         solutions=solutions,
                         solve_params=solve_params)

        # Sanity checks
        check_data_type(weights)
        check_data_type(capacities)

        assert len(input_features) == len(weights) and len(input_features) == len(capacities), \
            "input_features, target_features, scaled_target_features, weights and capacities must have the same length"

        # If input data are numpy arrays then convert them of torch Tensor
        self._weights = from_numpy_to_torch(weights)
        self._capacities = from_numpy_to_torch(capacities)

        self._opt_prob_params = [{"weights": self._weights[index], "capacity": self._capacities[index]}
                                 for index in range(len(self))]

    def __getitem__(self, index: int) -> Tuple:
        """
        Built-in method to get a single entry of the dataset.
        :param index: int; index of the entry.
        :return: tuple with input features, target features, scaled target features, weights, capacity optimal solution, solver parameters
                 and the index of the entry. The index is useful when the optimization problem has different parameters
                 for each entry (e.g. Knapsack with different item weights).
        """

        x_i = self._x[index]
        y_i = self._y[index]
        scaled_y_i = self._scaled_y[index]
        sol_i = self._solutions[index]
        solve_params_i = self._solve_params[index]
        opt_prob_params_i = self._opt_prob_params[index]

        return x_i, y_i, scaled_y_i, sol_i, solve_params_i, opt_prob_params_i
########################################################################################################################


class SetCoverDataWrapper(DataWrapper):
    def __init__(self,
                 input_features: Union[torch.Tensor, np.ndarray],
                 target_features: Union[torch.Tensor, np.ndarray],
                 scaled_target_features: Union[torch.Tensor, np.ndarray],
                 set_costs: Union[torch.Tensor, np.ndarray],
                 prod_costs: Union[torch.Tensor, np.ndarray],
                 availabilities: Union[torch.Tensor, np.ndarray],
                 solutions: Union[torch.Tensor, np.ndarray],
                 solve_params: Iterable[Dict] = None):
        """
        Wrapper of torch dataset.
        :param input_features: torch.Tensor or numpy.ndarray; the input features of the dataset.
        :param target_features: torch.Tensor or numpy.ndarray; the target features of the dataset.
        :param scaled_target_features: torch.Tensor or numpy.ndarray; the scaled target features of the dataset.
        :param solver; optimization_problems.optimization_problem.OptimizationProblem; the instance that wraps the
                       optimization problem and its solver.
        :param solve_params: list of dict; the solver parameters for each instance of the dataset.
        """

        super().__init__(input_features=input_features,
                         target_features=target_features,
                         scaled_target_features=scaled_target_features,
                         solutions=solutions,
                         solve_params=solve_params)

        # Sanity checks
        check_data_type(set_costs)
        check_data_type(prod_costs)
        check_data_type(availabilities)

        assert len(input_features) == len(set_costs) and len(input_features) == len(prod_costs) and \
               len(input_features) == len(availabilities), \
            "input_features, target_features, scaled_target_features, weights and capacities must have the same length"

        # If input data are numpy arrays then convert them of torch Tensor
        self._set_costs = from_numpy_to_torch(set_costs)
        self._prod_costs = from_numpy_to_torch(prod_costs)
        availabilities = availabilities.reshape(availabilities.shape[0], -1)
        self._availabilities = from_numpy_to_torch(availabilities)

        self._opt_prob_params = [{'set_costs': self._set_costs[index],
                                  'prod_costs': self._prod_costs[index],
                                  'availability': self._availabilities[index]}
                                 for index in range(len(self))]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple:
        """
        Built-in method to get a single entry of the dataset.
        :param index: int; index of the entry.
        :return: tuple with input features, target features, scaled target features, weights, capacity optimal solution, solver parameters
                 and the index of the entry. The index is useful when the optimization problem has different parameters
                 for each entry (e.g. Knapsack with different item weights).
        """

        x_i = self._x[index]
        y_i = self._y[index]
        scaled_y_i = self._scaled_y[index]
        sol_i = self._solutions[index]
        solve_params_i = self._solve_params[index]
        opt_prob_params_i = self._opt_prob_params[index]

        return x_i, y_i, scaled_y_i, sol_i, solve_params_i, opt_prob_params_i

########################################################################################################################


class FractionalKPDataWrapper(DataWrapper):
    def __init__(self,
                 input_features: Union[torch.Tensor, np.ndarray],
                 target_features: Union[torch.Tensor, np.ndarray],
                 scaled_target_features: Union[torch.Tensor, np.ndarray],
                 penalties: Union[torch.Tensor, np.ndarray],
                 capacities: Union[torch.Tensor, np.ndarray],
                 solutions: Union[torch.Tensor, np.ndarray],
                 solve_params: Iterable[Dict] = None):
        """
        Wrapper of torch dataset.
        :param input_features: torch.Tensor or numpy.ndarray; the input features of the dataset.
        :param target_features: torch.Tensor or numpy.ndarray; the target features of the dataset.
        :param scaled_target_features: torch.Tensor or numpy.ndarray; the scaled target features of the dataset.
        :param solve_params: list of dict; the solver parameters for each instance of the dataset.
        """

        super().__init__(input_features=input_features,
                         target_features=target_features,
                         scaled_target_features=scaled_target_features,
                         solutions=solutions,
                         solve_params=solve_params)

        # Sanity checks
        check_data_type(penalties)
        check_data_type(capacities)

        assert len(input_features) == len(penalties) and len(input_features) == len(capacities), \
            "input_features, target_features, scaled_target_features, penalties and capacities must have the same length"

        # If input data are numpy arrays then convert them of torch Tensor
        self._penalties = from_numpy_to_torch(penalties)
        self._capacities = from_numpy_to_torch(capacities)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple:
        """
        Built-in method to get a single entry of the dataset.
        :param index: int; index of the entry.
        :return: tuple with input features, target features, scaled target features, weights, capacity optimal solution, solver parameters
                 and the index of the entry. The index is useful when the optimization problem has different parameters
                 for each entry (e.g. Knapsack with different item weights).
        """

        x_i = self._x[index]
        y_i = self._y[index]
        scaled_y_i = self._scaled_y[index]
        sol_i = self._solutions[index]
        solve_params_i = self._solve_params[index]

        capacity = torch.unsqueeze(self._capacities[index], dim=0)
        opt_prob_params_i = {'penalty': self._penalties[index], 'capacity': capacity}

        return x_i, y_i, scaled_y_i, sol_i, solve_params_i, opt_prob_params_i

########################################################################################################################


class StochasticKnapsackDataWrapper(DataWrapper):
    def __init__(self,
                 input_features: Union[torch.Tensor, np.ndarray],
                 target_features: Union[torch.Tensor, np.ndarray],
                 scaled_target_features: Union[torch.Tensor, np.ndarray],
                 values: Union[torch.Tensor, np.ndarray],
                 capacities: Union[torch.Tensor, np.ndarray],
                 solutions: Union[torch.Tensor, np.ndarray],
                 solve_params: Iterable[Dict] = None):
        """
        Wrapper of torch dataset.
        :param input_features: torch.Tensor or numpy.ndarray; the input features of the dataset.
        :param target_features: torch.Tensor or numpy.ndarray; the target features of the dataset.
        :param scaled_target_features: torch.Tensor or numpy.ndarray; the scaled target features of the dataset.
        :param solve_params: list of dict; the solver parameters for each instance of the dataset.
        """

        super().__init__(input_features=input_features,
                         target_features=target_features,
                         scaled_target_features=scaled_target_features,
                         solutions=solutions,
                         solve_params=solve_params)

        # Sanity checks
        check_data_type(values)
        check_data_type(capacities)

        assert len(input_features) == len(values) and len(input_features) == len(capacities), \
            "input_features, target_features, scaled_target_features, values and capacities must have the same length"

        # If input data are numpy arrays then convert them of torch Tensor
        self._values = from_numpy_to_torch(values)
        self._capacities = from_numpy_to_torch(capacities)
        capacities = np.expand_dims(capacities, -1)
        capacities = from_numpy_to_torch(capacities)

        self._opt_prob_params = [{'values': self._values[index],
                                  'capacity': capacities[index]} for index in range(len(self))]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple:
        """
        Built-in method to get a single entry of the dataset.
        :param index: int; index of the entry.
        :return: tuple with input features, target features, scaled target features, weights, capacity optimal solution, solver parameters
                 and the index of the entry. The index is useful when the optimization problem has different parameters
                 for each entry (e.g. Knapsack with different item weights).
        """

        x_i = self._x[index]
        y_i = self._y[index]
        scaled_y_i = self._scaled_y[index]
        sol_i = self._solutions[index]
        solve_params_i = self._solve_params[index]
        opt_prob_params_i = self._opt_prob_params[index]

        return x_i, y_i, scaled_y_i, sol_i, solve_params_i, opt_prob_params_i

########################################################################################################################


class StochasticCapacityKnapsackDataWrapper(DataWrapper):
    def __init__(self,
                 input_features: Union[torch.Tensor, np.ndarray],
                 target_features: Union[torch.Tensor, np.ndarray],
                 scaled_target_features: Union[torch.Tensor, np.ndarray],
                 values: Union[torch.Tensor, np.ndarray],
                 weights: Union[torch.Tensor, np.ndarray],
                 solutions: Union[torch.Tensor, np.ndarray],
                 solve_params: Iterable[Dict] = None):
        """
        Wrapper of torch dataset.
        :param input_features: torch.Tensor or numpy.ndarray; the input features of the dataset.
        :param target_features: torch.Tensor or numpy.ndarray; the target features of the dataset.
        :param scaled_target_features: torch.Tensor or numpy.ndarray; the scaled target features of the dataset.
        :param solver; optimization_problems.optimization_problem.OptimizationProblem; the instance that wraps the
                       optimization problem and its solver.
        :param solve_params: list of dict; the solver parameters for each instance of the dataset.
        """

        super().__init__(input_features=input_features,
                         target_features=target_features,
                         scaled_target_features=scaled_target_features,
                         solutions=solutions,
                         solve_params=solve_params)

        # Sanity checks
        check_data_type(values)
        check_data_type(weights)

        assert len(input_features) == len(values) and len(input_features) == len(weights), \
            "input_features, target_features, scaled_target_features, values and capacities must have the same length"

        # If input data are numpy arrays then convert them of torch Tensor
        self._values = from_numpy_to_torch(values)
        self._weights = from_numpy_to_torch(weights)

        self._opt_prob_params = [{'values': self._values[index],
                                  'weights': self._weights[index]} for index in range(len(self))]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple:
        """
        Built-in method to get a single entry of the dataset.
        :param index: int; index of the entry.
        :return: tuple with input features, target features, scaled target features, weights, capacity optimal solution, solver parameters
                 and the index of the entry. The index is useful when the optimization problem has different parameters
                 for each entry (e.g. Knapsack with different item weights).
        """

        x_i = self._x[index]
        y_i = self._y[index]
        scaled_y_i = self._scaled_y[index]
        sol_i = self._solutions[index]
        solve_params_i = self._solve_params[index]
        opt_prob_params_i = self._opt_prob_params[index]

        return x_i, y_i, scaled_y_i, sol_i, solve_params_i, opt_prob_params_i
