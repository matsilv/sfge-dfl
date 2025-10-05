"""
    PyTorch DataLoaders creation.
"""

# FIXME: the build_dataloaders function are way too long and there is repeated code

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# FIXME: import too long
from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems.knapsack_problem import KnapsackProblem
from optimization_problems.quadratic_kp_problem import QuadraticKP
from optimization_problems.set_cover_problem import StochasticWeightedSetMultiCover
from optimization_problems.fractional_kp import FractionalKPProblem
from optimization_problems.stochastic_weights_kp import StochasticWeightsKnapsackProblem
from optimization_problems.stochastic_capacity_kp import StochasticCapacityKnapsackProblem
from optimization_problems.knapsack_problem import PROBLEM_ID as KNAPSACK
from optimization_problems.quadratic_kp_problem import PROBLEM_ID as QUADRATIC_KP
from optimization_problems.set_cover_problem import PROBLEM_ID as WSMC
from optimization_problems.fractional_kp import PROBLEM_ID as FRACTIONAL_KP
from optimization_problems.stochastic_weights_kp import PROBLEM_ID as STOCHASTIC_WEIGHTS_KP
from optimization_problems.stochastic_capacity_kp import PROBLEM_ID as STOCHASTIC_CAPACITY_KP
from utils.datawrapper import KnapsackDataWrapper, SetCoverDataWrapper, FractionalKPDataWrapper
from utils.datawrapper import StochasticKnapsackDataWrapper, StochasticCapacityKnapsackDataWrapper
from data.generation_scripts.generate_wsmc_data import get_datapath as get_wsmc_datapath
from data.generation_scripts import DATAPATH_PREFIX

from typing import Dict, Tuple, List, Iterable

########################################################################################################################


def dataset_split(arrays: List[Iterable],
                  train_split: float,
                  val_split: float,
                  seed: int,
                  reduce_test_len: int = 4) -> Tuple[List[Iterable]]:
    """
    Split a list of arrays in training, validation and test sets (with the same split).
    :param arrays: list of iterables; the array to split.
    :param train_split: float; the fraction of samples to use for train.
    :param val_split: float; the fraction of samples to use for validation.
    :param seed: int; the seed use to reproduce the random split.
    :param max_eval: int; since testing the PFL+SAA method is computational expensive, we focus on a maximum of
    len(test) // 4 instances as we do in the pfl_plus_saa.py script.
    :return:
    """

    # Sanity checks
    assert isinstance(arrays, list)
    assert isinstance(train_split, float)
    assert isinstance(val_split, float)
    assert isinstance(seed, int)

    n_samples = len(arrays[0])

    train_samples = list()
    val_samples = list()
    test_samples = list()

    for a in arrays:

        # Sanity check
        assert len(a) == n_samples, "The arrays must have the same length"

        # Random split the dataset in training, test and validation sets

        indexes = np.arange(n_samples)
        idx_train, idx_test = train_test_split(indexes, train_size=train_split, random_state=seed)
        idx_train, idx_val = train_test_split(idx_train, test_size=val_split, random_state=seed)

        a_train = a[idx_train]
        a_val = a[idx_val]
        a_test = a[idx_test]
        a_test = a_test[:len(a_test) // reduce_test_len]

        train_samples.append(a_train)
        val_samples.append(a_val)
        test_samples.append(a_test)

    return train_samples, val_samples, test_samples

########################################################################################################################


def build_data_loaders(config: Dict,
                       rnd_split_seed: int,
                       dataset_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, OptimizationProblem]:
    """
    Create the PyTorch dataloaders for training, validation, and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader: training, validation, and test dataloaders.
    """

    # Get the optimization problem name from configuration file
    optimization_problem_name = config["optimization_problem"]

    # Define a dictionary to map optimization problem names to function calls
    problem_functions = {
        KNAPSACK: build_data_loaders_knapsack,
        QUADRATIC_KP: build_data_loaders_quadratic_kp,
        WSMC: build_data_loaders_wsmc,
        FRACTIONAL_KP: build_data_loaders_fractional_kp,
        STOCHASTIC_WEIGHTS_KP: build_data_loaders_stochastic_weights_kp,
        STOCHASTIC_CAPACITY_KP: build_data_loaders_stochastic_capacity_kp
    }

    # Check if the optimization problem name is supported
    if optimization_problem_name not in problem_functions:
        raise Exception("Optimization problem type is not supported")

    # Call the corresponding function based on the optimization problem name
    problem_function = problem_functions[optimization_problem_name]
    train_dl, validation_dl, test_dl, optimization_problem = \
        problem_function(config, rnd_split_seed=rnd_split_seed, dataset_seed=dataset_seed)

    return train_dl, validation_dl, test_dl, optimization_problem


########################################################################################################################


def build_data_loaders_knapsack(config: Dict,
                                rnd_split_seed: int,
                                dataset_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create the PyTorch dataloaders for training, validation and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader: training, validation and test dataloaders,
             and the knapsack instance.
    """

    # Extract the configuration parameters
    num_instances = config['n']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    mult_noise = config['mult_noise']
    add_noise = config['add_noise']
    deg = config['deg']
    relative_capacity = config['relative_capacity']
    correlation_type = config['correlate_values_and_weights']
    rho = config['rho']
    train_split = config['prop_training']
    val_split = config['prop_validation']

    # Reading dataframes
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    suffix += f'_mult_noise_{mult_noise}_add_noise_{add_noise}_deg_{deg}'
    suffix += f'_relative_capacity_{relative_capacity}_correlation_type_{correlation_type}'
    suffix += f'_rho_{rho}.csv'

    prefix = os.path.join(DATAPATH_PREFIX, KNAPSACK, f'seed-{dataset_seed}')

    features_filepath = os.path.join(prefix, 'features'+suffix)
    target_filepath = os.path.join(prefix, 'targets'+suffix)
    weights_filepath = os.path.join(prefix, 'weights'+suffix)
    solutions_filepath = os.path.join(prefix, 'solutions'+suffix)

    # Converting data to NumPy arrays
    weights_df = pd.read_csv(weights_filepath, index_col=0)
    targets_df = pd.read_csv(target_filepath, index_col=0)
    features_df = pd.read_csv(features_filepath, index_col=0)
    solutions_df = pd.read_csv(solutions_filepath, index_col=0)

    x = features_df.values.astype(np.float32)
    y = targets_df.values.astype(np.float32)
    solutions = solutions_df.values.astype(np.float32)
    weights = weights_df.values.flatten()

    # Random split between training, validation and test sets
    train_samples, val_samples, test_samples = \
        dataset_split([x, y, solutions],
                      train_split=train_split,
                      val_split=val_split,
                      seed=rnd_split_seed)

    x_train, y_train, sol_train = train_samples[0], train_samples[1], train_samples[2]
    x_val, y_val, sol_val = val_samples[0], val_samples[1], val_samples[2]
    x_test, y_test, sol_test = test_samples[0], test_samples[1], test_samples[2]

    n_training = len(x_train)
    n_validation = len(x_val)
    n_test = len(x_test)

    # Compute the capacity as a function of the weights
    capacity = sum(weights) * relative_capacity

    # Weights are the same for all the instances, so we simply repeat them
    weights_train = np.asarray([weights for _ in range(n_training)])
    weights_val = np.asarray([weights for _ in range(n_validation)])
    weights_test = np.asarray([weights for _ in range(n_test)])

    # Capacity is the same for all the instances, so we simply repeat them
    capacity_train = np.asarray([capacity for _ in range(n_training)])
    capacity_val = np.asarray([capacity for _ in range(n_validation)])
    capacity_test = np.asarray([capacity for _ in range(n_test)])

    # Standardize targets
    if 'scale_predictions' in config.keys():
        scaler = StandardScaler()
        scaled_y_train = scaler.fit_transform(y_train)
        scaled_y_validation = scaler.transform(y_val)
        scaled_y_test = scaler.transform(y_test)
    else:
        scaled_y_train = y_train
        scaled_y_validation = y_val
        scaled_y_test = y_test

    knapsack_problem = KnapsackProblem(dim=output_dim)

    # Build data wrappers and loaders
    train_data_wrapper = \
        KnapsackDataWrapper(input_features=x_train,
                            target_features=y_train,
                            scaled_target_features=scaled_y_train,
                            weights=weights_train,
                            capacities=capacity_train,
                            solutions=sol_train)

    test_data_wrapper = \
        KnapsackDataWrapper(input_features=x_test,
                            target_features=y_test,
                            scaled_target_features=scaled_y_test,
                            weights=weights_test,
                            capacities=capacity_test,
                            solutions=sol_test)

    val_data_wrapper = \
        KnapsackDataWrapper(input_features=x_val,
                            target_features=y_val,
                            scaled_target_features=scaled_y_validation,
                            weights=weights_val,
                            capacities=capacity_val,
                            solutions=sol_val)

    train_dl = DataLoader(train_data_wrapper, batch_size=config['batch_size'])
    validation_dl = DataLoader(val_data_wrapper, batch_size=1)
    test_dl = DataLoader(test_data_wrapper, batch_size=1)

    return train_dl, validation_dl, test_dl, knapsack_problem

########################################################################################################################


def build_data_loaders_quadratic_kp(config: Dict,
                                    rnd_split_seed: int,
                                    dataset_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, OptimizationProblem]:
    """
    Create the PyTorch dataloaders for training, validation and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader and 1 instance of
             optimization_problems.optimization_problem.OptimizationProblem: training, validation and test dataloaders,
             and the quadratic knapsack instance.
    """

    # Extract the configuration parameters
    batch_size = config['batch_size']
    num_instances = config['n']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    mult_noise = config['mult_noise']
    add_noise = config['add_noise']
    deg = config['deg']
    relative_capacity = config['relative_capacity']
    correlation_type = config['correlate_values_and_weights']
    rho = config['rho']
    train_split = config['prop_training']
    val_split = config['prop_validation']

    # Sanity check of the config file
    num_items = int(np.sqrt(output_dim))
    assert num_items**2 == output_dim, "output_dim must be a perfect square"

    # Reading dataframes
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    suffix += f'_mult_noise_{mult_noise}_add_noise_{add_noise}_deg_{deg}'
    suffix += f'_relative_capacity_{relative_capacity}_correlation_type_{correlation_type}'
    suffix += f'_rho_{rho}.csv'

    prefix = os.path.join(DATAPATH_PREFIX, QUADRATIC_KP, f'seed-{dataset_seed}')

    features_filepath = os.path.join(prefix, 'features' + suffix)
    df_features = pd.read_csv(features_filepath, index_col=0)

    target_filepath = os.path.join(prefix, 'targets' + suffix)
    df_targets = pd.read_csv(target_filepath, index_col=0)

    weights_filepath = os.path.join(prefix, 'weights' + suffix)
    df_weights = pd.read_csv(weights_filepath, index_col=0)

    solutions_filepath = os.path.join(prefix, 'solutions' + suffix)
    solutions_df = pd.read_csv(solutions_filepath, index_col=0)

    # Converting data to NumPy arrays
    x = df_features.values.astype(np.float32)
    y = df_targets.values.astype(np.float32)
    solutions = solutions_df.values.astype(np.float32)
    weights = df_weights.values.flatten()

    # Random split between training, validation and test sets
    train_samples, val_samples, test_samples = \
        dataset_split([x, y, solutions],
                      train_split=train_split,
                      val_split=val_split,
                      seed=rnd_split_seed)

    x_train, y_train, sol_train = train_samples[0], train_samples[1], train_samples[2]
    x_val, y_val, sol_val = val_samples[0], val_samples[1], val_samples[2]
    x_test, y_test, sol_test = test_samples[0], test_samples[1], test_samples[2]

    n_training = len(x_train)
    n_validation = len(x_val)
    n_test = len(x_test)

    # Building KnapsackProblem instance
    sum_of_weights = weights.sum()
    capacity = sum_of_weights * relative_capacity
    quadratic_kp_problem = QuadraticKP(dim=num_items)

    # Weights are the same for all the instances so we simply repeat them
    weights_train = np.asarray([weights for _ in range(n_training)])
    weights_val = np.asarray([weights for _ in range(n_validation)])
    weights_test = np.asarray([weights for _ in range(n_test)])

    # Capacity is the same for all the instances so we simply repeat them
    capacity_train = np.asarray([capacity for _ in range(n_training)])
    capacity_val = np.asarray([capacity for _ in range(n_validation)])
    capacity_test = np.asarray([capacity for _ in range(n_test)])

    # Standardize targets
    if 'scale_predictions' in config.keys():
        scaler = StandardScaler()
        scaled_y_train = scaler.fit_transform(y_train)
        scaled_y_validation = scaler.transform(y_val)
        scaled_y_test = scaler.transform(y_test)
    else:
        scaled_y_train = y_train
        scaled_y_validation = y_val
        scaled_y_test = y_test

    # Build data wrappers and loaders
    train_data_wrapper = \
        KnapsackDataWrapper(input_features=x_train,
                            target_features=y_train,
                            scaled_target_features=scaled_y_train,
                            weights=weights_train,
                            capacities=capacity_train,
                            solutions=sol_train)

    test_data_wrapper = \
        KnapsackDataWrapper(input_features=x_test,
                            target_features=y_test,
                            scaled_target_features=scaled_y_test,
                            weights=weights_test,
                            capacities=capacity_test,
                            solutions=sol_test)

    val_data_wrapper = \
        KnapsackDataWrapper(input_features=x_val,
                            target_features=y_val,
                            scaled_target_features=scaled_y_validation,
                            weights=weights_val,
                            capacities=capacity_val,
                            solutions=sol_val)

    # Build data wrappers and loaders
    train_dl = DataLoader(train_data_wrapper, batch_size=batch_size)
    validation_dl = DataLoader(val_data_wrapper, batch_size=1)
    test_dl = DataLoader(test_data_wrapper, batch_size=1)

    return train_dl, validation_dl, test_dl, quadratic_kp_problem

########################################################################################################################


def build_data_loaders_wsmc(config: Dict,
                            rnd_split_seed: int,
                            dataset_seed: int):
    """
    Create the PyTorch dataloaders for training, validation and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader and 1 instance of
             optimization_problems.optimization_problem.OptimizationProblem: training, validation and test dataloaders,
             and the knapsack instance.
    """

    # Sanity check of the config file
    num_prods = config['num_prods']
    num_sets = config['num_sets']
    penalty_factor = config['penalty_factor']
    train_split = config['prop_training']
    val_split = config['prop_validation']
    batch_size = config['batch_size']

    # Sanity checks
    assert isinstance(num_prods, int)
    assert isinstance(num_sets, int)
    assert isinstance(penalty_factor, float)

    # Get WSMC loadpath
    loadpath = \
        get_wsmc_datapath(prefix=DATAPATH_PREFIX,
                          num_prods=num_prods,
                          num_sets=num_sets,
                          penalty_factor=penalty_factor,
                          seed=dataset_seed)

    # Load optimization problem parameters
    availability = np.load(os.path.join(loadpath, 'availability.npy'))
    prod_costs = np.load(os.path.join(loadpath, 'prod_costs.npy'))
    set_costs = np.load(os.path.join(loadpath, 'set_costs.npy'))
    targets = pd.read_csv(os.path.join(loadpath, 'targets.csv'), index_col=0)
    features = pd.read_csv(os.path.join(loadpath, 'features.csv'), index_col=0)
    solutions = pd.read_csv(os.path.join(loadpath, 'solutions.csv'), index_col=0)

    # Converting data to NumPy arrays
    x = np.asarray(features.values, dtype=np.float32)
    y = np.asarray(targets.values, dtype=np.float32)
    solutions = np.asarray(solutions.values, dtype=np.float32)

    # Random split between training, validation and test sets
    train_samples, val_samples, test_samples = \
        dataset_split([x, y, solutions],
                      train_split=train_split,
                      val_split=val_split,
                      seed=rnd_split_seed)

    x_train, y_train, sol_train = train_samples[0], train_samples[1], train_samples[2]
    x_val, y_val, sol_val = val_samples[0], val_samples[1], val_samples[2]
    x_test, y_test, sol_test = test_samples[0], test_samples[1], test_samples[2]

    n_training = len(x_train)
    n_validation = len(x_val)
    n_test = len(x_test)

    # The availability matrix is the same for all the instances so we simply repeat it
    availability_train = np.asarray([availability for _ in range(n_training)])
    availability_validation = np.asarray([availability for _ in range(n_validation)])
    availability_test = np.asarray([availability for _ in range(n_test)])

    # The product costs are is the same for all the instances so we simply repeat it
    prod_costs_train = np.asarray([prod_costs for _ in range(n_training)])
    prod_costs_validation = np.asarray([prod_costs for _ in range(n_validation)])
    prod_costs_test = np.asarray([prod_costs for _ in range(n_test)])

    # The set costs are is the same for all the instances so we simply repeat it
    set_costs_train = np.asarray([set_costs for _ in range(n_training)])
    set_costs_validation = np.asarray([set_costs for _ in range(n_validation)])
    set_costs_test = np.asarray([set_costs for _ in range(n_test)])

    # Standardize features
    if 'scale_predictions' in config.keys():
        scaler = StandardScaler()
        scaled_y_train = scaler.fit_transform(y_train)
        scaled_y_validation = scaler.transform(y_val)
        scaled_y_test = scaler.transform(y_test)
    else:
        scaled_y_train = y_train
        scaled_y_validation = y_val
        scaled_y_test = y_test

    # Building the WSMC instance
    problem = \
        StochasticWeightedSetMultiCover(num_sets=num_sets,
                                        num_products=num_prods)

    # Build data loaders
    train_data_wrapper = \
        SetCoverDataWrapper(input_features=x_train,
                            target_features=y_train,
                            scaled_target_features=scaled_y_train,
                            set_costs=set_costs_train,
                            prod_costs=prod_costs_train,
                            availabilities=availability_train,
                            solutions=sol_train)

    test_data_wrapper = \
        SetCoverDataWrapper(input_features=x_test,
                            target_features=y_test,
                            scaled_target_features=scaled_y_test,
                            set_costs=set_costs_test,
                            prod_costs=prod_costs_test,
                            availabilities=availability_test,
                            solutions=sol_test)

    validation_data_wrapper = \
        SetCoverDataWrapper(input_features=x_val,
                            target_features=y_val,
                            scaled_target_features=scaled_y_validation,
                            set_costs=set_costs_validation,
                            prod_costs=prod_costs_validation,
                            availabilities=availability_validation,
                            solutions=sol_val)

    train_dl = DataLoader(train_data_wrapper, batch_size=batch_size)
    test_dl = DataLoader(test_data_wrapper, batch_size=1)
    validation_dl = DataLoader(validation_data_wrapper, batch_size=1)

    return train_dl, validation_dl, test_dl, problem

########################################################################################################################


def build_data_loaders_fractional_kp(config: Dict,
                                     rnd_split_seed: int,
                                     # FIXME: this is needed to allow interface compatibility
                                     dataset_seed: int):
    """
    Create the PyTorch dataloaders for training, validation and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader and 1 instance of
             optimization_problems.optimization_problem.OptimizationProblem: training, validation and test dataloaders,
             and the knapsack instance.
    """

    # Sanity check of the config file
    penalty_factor = config['penalty_factor']
    capacity = config['capacity']
    dim = config['num_items']
    batch_size = config['batch_size']
    train_split = config['prop_training']
    val_split = config['prop_validation']

    assert isinstance(penalty_factor, str)
    assert isinstance(capacity, int)
    assert isinstance(dim, int)

    loadpath = os.path.join(DATAPATH_PREFIX, FRACTIONAL_KP)

    # Load optimization problem parameters
    features = pd.read_csv(os.path.join(loadpath, 'features.csv'), index_col=0)
    penalties = pd.read_csv(os.path.join(loadpath, f'penalty{penalty_factor}.csv'), index_col=0)
    item_values = pd.read_csv(os.path.join(loadpath, f'values.csv'), index_col=0)
    item_weights = pd.read_csv(os.path.join(loadpath, f'weights.csv'), index_col=0)
    solutions = pd.read_csv(os.path.join(loadpath, f'solutions-{capacity}.csv'), index_col=0)

    # Converting data to NumPy arrays
    features = np.asarray(features.values, dtype=np.float32)
    penalties = np.asarray(penalties.values, dtype=np.float32)
    item_values = np.asarray(item_values.values, dtype=np.float32)
    item_weights = np.asarray(item_weights.values, dtype=np.float32)
    solutions = np.asarray(solutions.values, dtype=np.float32)

    x = features
    y = np.concatenate((item_values, item_weights), axis=1)

    # Random split between training, validation and test sets
    train_samples, val_samples, test_samples = \
        dataset_split([x, y, penalties, solutions],
                      train_split=train_split,
                      val_split=val_split,
                      seed=rnd_split_seed)

    x_train, y_train, p_train, sol_train = train_samples[0], train_samples[1], train_samples[2], train_samples[3]
    x_val, y_val, p_val, sol_val = val_samples[0], val_samples[1], val_samples[2], val_samples[3]
    x_test, y_test, p_test, sol_test = test_samples[0], test_samples[1], test_samples[2], test_samples[3]

    n_training = len(x_train)
    n_validation = len(x_val)
    n_test = len(x_test)

    # The capacity is the same for all the instances so we simply repeat it
    capacity_train = np.asarray([capacity for _ in range(n_training)])
    capacity_validation = np.asarray([capacity for _ in range(n_validation)])
    capacity_test = np.asarray([capacity for _ in range(n_test)])

    # Standardize targets
    if 'scale_predictions' in config.keys():
        scaler = StandardScaler()
        scaled_y_train = scaler.fit_transform(y_train)
        scaled_y_validation = scaler.transform(y_val)
        scaled_y_test = scaler.transform(y_test)
    else:
        scaled_y_train = y_train
        scaled_y_validation = y_val
        scaled_y_test = y_test

    # Building the Fractional KP instance
    problem = FractionalKPProblem(dim=dim)

    # Build data loaders
    train_data_wrapper = \
        FractionalKPDataWrapper(input_features=x_train,
                                target_features=y_train,
                                scaled_target_features=scaled_y_train,
                                penalties=p_train,
                                capacities=capacity_train,
                                solutions=sol_train)

    test_data_wrapper = \
        FractionalKPDataWrapper(input_features=x_test,
                                target_features=y_test,
                                scaled_target_features=scaled_y_test,
                                penalties=p_test,
                                capacities=capacity_test,
                                solutions=sol_test)

    validation_data_wrapper = \
        FractionalKPDataWrapper(input_features=x_val,
                                target_features=y_val,
                                scaled_target_features=scaled_y_validation,
                                penalties=p_val,
                                capacities=capacity_validation,
                                solutions=sol_val)

    train_dl = DataLoader(train_data_wrapper, batch_size=batch_size)
    test_dl = DataLoader(test_data_wrapper, batch_size=1)
    validation_dl = DataLoader(validation_data_wrapper, batch_size=1)

    return train_dl, validation_dl, test_dl, problem

########################################################################################################################


def build_data_loaders_stochastic_weights_kp(config: Dict,
                                             rnd_split_seed: int,
                                             dataset_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, StochasticWeightsKnapsackProblem]:
    """
    Create the PyTorch dataloaders for training, validation and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader: training, validation and test dataloaders,
             and the knapsack instance.
    """

    # Extract the configuration parameters
    batch_size = config['batch_size']
    num_instances = config['n']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    mult_noise = config['mult_noise']
    add_noise = config['add_noise']
    deg = config['deg']
    relative_capacity = config['relative_capacity']
    correlation_type = config['correlate_values_and_weights']
    rho = config['rho']
    train_split = config['prop_training']
    val_split = config['prop_validation']
    penalty = config['penalty']

    # Reading dataframes
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    suffix += f'_mult_noise_{mult_noise}_add_noise_{add_noise}_deg_{deg}'
    suffix += f'_relative_capacity_{relative_capacity}_correlation_type_{correlation_type}'
    suffix += f'_rho_{rho}'

    prefix = os.path.join(DATAPATH_PREFIX, STOCHASTIC_WEIGHTS_KP, f'seed-{dataset_seed}')

    features_filepath = os.path.join(prefix, 'features' + suffix + '.csv')
    df_features = pd.read_csv(features_filepath, index_col=0)

    target_filepath = os.path.join(prefix, 'targets' + suffix + '.csv')
    df_targets = pd.read_csv(target_filepath, index_col=0)

    values_filepath = os.path.join(prefix, 'values' + suffix + '.csv')
    df_values = pd.read_csv(values_filepath, index_col=0)

    capacity_filepath = os.path.join(prefix, 'capacity' + suffix + '.npy')
    capacity = np.load(capacity_filepath)

    solutions_filepath = os.path.join(prefix, 'solutions' + suffix + '.csv')
    df_solutions = pd.read_csv(solutions_filepath, index_col=0)

    # Converting data to NumPy arrays
    x = df_features.values.astype(np.float32)
    y = df_targets.values.astype(np.float32)
    solutions = df_solutions.values.astype(np.float32)
    values = df_values.values.flatten()

    # Random split between training, validation and test sets
    train_samples, val_samples, test_samples = \
        dataset_split([x, y, solutions],
                      train_split=train_split,
                      val_split=val_split,
                      seed=rnd_split_seed)

    x_train, y_train, sol_train = train_samples[0], train_samples[1], train_samples[2]
    x_val, y_val, sol_val = val_samples[0], val_samples[1], val_samples[2]
    x_test, y_test, sol_test = test_samples[0], test_samples[1], test_samples[2]

    n_training = len(x_train)
    n_validation = len(x_val)
    n_test = len(x_test)

    # Weights are the same for all the instances so we simply repeat them
    values_train = np.asarray([values for _ in range(n_training)])
    values_val = np.asarray([values for _ in range(n_validation)])
    values_test = np.asarray([values for _ in range(n_test)])

    # Capacity is the same for all the instances so we simply repeat them
    capacity_train = np.asarray([capacity for _ in range(n_training)])
    capacity_val = np.asarray([capacity for _ in range(n_validation)])
    capacity_test = np.asarray([capacity for _ in range(n_test)])

    # Standardize targets
    if 'scale_predictions' in config.keys():
        scaler = StandardScaler()
        scaled_y_train = scaler.fit_transform(y_train)
        scaled_y_validation = scaler.transform(y_val)
        scaled_y_test = scaler.transform(y_test)
    else:
        scaled_y_train = y_train
        scaled_y_validation = y_val
        scaled_y_test = y_test

    stochastic_kp_problem = StochasticWeightsKnapsackProblem(dim=output_dim, penalty=penalty)

    # Build data wrappers and loaders
    train_data_wrapper = \
        StochasticKnapsackDataWrapper(input_features=x_train,
                                      target_features=y_train,
                                      scaled_target_features=scaled_y_train,
                                      values=values_train,
                                      capacities=capacity_train,
                                      solutions=sol_train)

    test_data_wrapper = \
        StochasticKnapsackDataWrapper(input_features=x_test,
                                      target_features=y_test,
                                      scaled_target_features=scaled_y_test,
                                      values=values_test,
                                      capacities=capacity_test,
                                      solutions=sol_test)

    val_data_wrapper = \
        StochasticKnapsackDataWrapper(input_features=x_val,
                                      target_features=y_val,
                                      scaled_target_features=scaled_y_validation,
                                      values=values_val,
                                      capacities=capacity_val,
                                      solutions=sol_val)

    train_dl = DataLoader(train_data_wrapper, batch_size=batch_size)
    validation_dl = DataLoader(val_data_wrapper, batch_size=1)
    test_dl = DataLoader(test_data_wrapper, batch_size=1)

    return train_dl, validation_dl, test_dl, stochastic_kp_problem

########################################################################################################################


def build_data_loaders_stochastic_capacity_kp(config: Dict,
                                              rnd_split_seed: int,
                                              dataset_seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create the PyTorch dataloaders for training, validation and test sets.
    :param config: dict; configuration dictionary.
    :param rnd_split_seed: int; the seed used to reproduce the training/validation/test sets.
    :param dataset_seed: int; the seed sued to reproduce the data generation process.
    :return: tuple of 3 instances of torch.utils.data.DataLoader: training, validation and test dataloaders,
             and the knapsack instance.
    """

    # Extract the configuration parameters
    batch_size = config['batch_size']
    num_instances = config['n']
    input_dim = config['input_dim']
    problem_dim = config['problem_dim']
    mult_noise = config['mult_noise']
    deg = config['deg']
    correlation_type = config['correlate_values_and_weights']
    rho = config['rho']
    train_split = config['prop_training']
    val_split = config['prop_validation']
    penalty = config['penalty']

    # Reading dataframes
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{problem_dim}'
    suffix += f'_mult_noise_{mult_noise}_deg_{deg}'
    suffix += f'_correlation_type_{correlation_type}'
    suffix += f'_rho_{rho}'

    prefix = os.path.join(DATAPATH_PREFIX, STOCHASTIC_CAPACITY_KP, f'seed-{dataset_seed}')

    features_filepath = os.path.join(prefix, 'features' + suffix + '.csv')
    df_features = pd.read_csv(features_filepath, index_col=0)

    target_filepath = os.path.join(prefix, 'targets' + suffix + '.csv')
    df_targets = pd.read_csv(target_filepath, index_col=0)

    values_filepath = os.path.join(prefix, 'values' + suffix + '.npy')
    values = np.load(values_filepath)

    weights_filepath = os.path.join(prefix, 'weights' + suffix + '.npy')
    weights = np.load(weights_filepath)

    solutions_filepath = os.path.join(prefix, 'solutions' + suffix + '.csv')
    df_solutions = pd.read_csv(solutions_filepath, index_col=0)

    # Converting data to NumPy arrays
    x = df_features.values.astype(np.float32)
    y = df_targets.values.astype(np.float32)
    solutions = df_solutions.values.astype(np.float32)

    # Random split between training, validation and test sets
    train_samples, val_samples, test_samples = \
        dataset_split([x, y, solutions],
                      train_split=train_split,
                      val_split=val_split,
                      seed=rnd_split_seed)

    x_train, y_train, sol_train = train_samples[0], train_samples[1], train_samples[2]
    x_val, y_val, sol_val = val_samples[0], val_samples[1], val_samples[2]
    x_test, y_test, sol_test = test_samples[0], test_samples[1], test_samples[2]

    n_training = len(x_train)
    n_validation = len(x_val)
    n_test = len(x_test)

    # Values are the same for all the instances, so we simply repeat them
    values_train = np.asarray([values for _ in range(n_training)])
    values_val = np.asarray([values for _ in range(n_validation)])
    values_test = np.asarray([values for _ in range(n_test)])

    # Weights are the same for all the instances so we simply repeat them
    weights_train = np.asarray([weights for _ in range(n_training)])
    weights_val = np.asarray([weights for _ in range(n_validation)])
    weights_test = np.asarray([weights for _ in range(n_test)])

    # Standardize targets
    if 'scale_predictions' in config.keys():
        scaler = StandardScaler()
        scaled_y_train = scaler.fit_transform(y_train)
        scaled_y_validation = scaler.transform(y_val)
        scaled_y_test = scaler.transform(y_test)
    else:
        scaled_y_train = y_train
        scaled_y_validation = y_val
        scaled_y_test = y_test

    stochastic_kp_capacity_problem = StochasticCapacityKnapsackProblem(dim=problem_dim, penalty=penalty)

    # Build data wrappers and loaders
    train_data_wrapper = \
        StochasticCapacityKnapsackDataWrapper(input_features=x_train,
                                              target_features=y_train,
                                              scaled_target_features=scaled_y_train,
                                              values=values_train,
                                              weights=weights_train,
                                              solutions=sol_train)

    test_data_wrapper = \
        StochasticCapacityKnapsackDataWrapper(input_features=x_test,
                                              target_features=y_test,
                                              scaled_target_features=scaled_y_test,
                                              values=values_test,
                                              weights=weights_test,
                                              solutions=sol_test)

    val_data_wrapper = \
        StochasticCapacityKnapsackDataWrapper(input_features=x_val,
                                              target_features=y_val,
                                              scaled_target_features=scaled_y_validation,
                                              values=values_val,
                                              weights=weights_val,
                                              solutions=sol_val)

    train_dl = DataLoader(train_data_wrapper, batch_size=batch_size)
    validation_dl = DataLoader(val_data_wrapper, batch_size=1)
    test_dl = DataLoader(test_data_wrapper, batch_size=1)

    return train_dl, validation_dl, test_dl, stochastic_kp_capacity_problem
