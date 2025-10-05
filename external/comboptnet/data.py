import json
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils import data

from models.comboptnet import ilp_solver
from utils.constraint_generation import sample_constraints
from utils.utils import compute_normalized_solution, save_pickle, load_pickle, AvgMeters, check_equal_ys, \
    solve_unconstrained, load_with_default_yaml, save_dict_as_one_line_csv

from typing import Dict, Tuple

########################################################################################################################


def load_dataset(dataset_type, base_dataset_path, **dataset_params):
    dataset_path = os.path.join(base_dataset_path, dataset_type)
    dataset_loader_dict = dict(static_constraints=static_constraint_dataloader,
                               knapsack=knapsack_dataloader,
                               stochastic_weights_kp=stochastic_weights_kp_dataloader,
                               stochastic_capacity_kp=stochastic_capacity_kp_dataloader,
                               wsmc=stochastic_set_cover_dataloader)
    return dataset_loader_dict[dataset_type](dataset_path=dataset_path, **dataset_params)


# FIXME: not tested
def static_constraint_dataloader(dataset_path, dataset_specification, num_gt_variables, num_gt_constraints,
                                 dataset_seed, train_dataset_size, loader_params):
    dataset_path = os.path.join(dataset_path, dataset_specification, str(num_gt_variables) + '_dim',
                                str(num_gt_constraints) + '_const', str(dataset_seed), 'dataset.p')
    datasets = load_pickle(dataset_path)

    train_ys = [tuple(y) for c, y in datasets['train'][:train_dataset_size]]
    test_ys = [tuple(y) for c, y in datasets['test'][:train_dataset_size]]

    print(f'Successfully loaded Static Constraints dataset.\n'
          f'Number of distinct solutions in train set: {len(set(train_ys))}\n'
          f'Number of distinct solutions in test set: {len(set(test_ys))}')

    training_set = Dataset(datasets['train'][:train_dataset_size])
    train_iterator = data.DataLoader(training_set, **loader_params)

    test_iterator = data.DataLoader(Dataset(datasets['test']), **loader_params)

    return (train_iterator, test_iterator), datasets['metadata']


# FIXME: not tested
def knapsack_dataloader(dataset_path, loader_params):
    variable_range = dict(lb=0, ub=1)
    num_variables = 10

    train_encodings = np.load(os.path.join(dataset_path, 'train_encodings.npy'))
    train_ys = compute_normalized_solution(np.load(os.path.join(dataset_path, 'train_sols.npy')), **variable_range)
    train_dataset = list(zip(train_encodings, train_ys))
    training_set = Dataset(train_dataset)
    train_iterator = data.DataLoader(training_set, **loader_params)

    test_encodings = np.load(os.path.join(dataset_path, 'test_encodings.npy'))
    test_ys = compute_normalized_solution(np.load(os.path.join(dataset_path, 'test_sols.npy')), **variable_range)
    test_dataset = list(zip(test_encodings, test_ys))
    test_set = Dataset(test_dataset)
    test_iterator = data.DataLoader(test_set, **loader_params)

    distinct_ys_train = len(set([tuple(y) for y in train_ys]))
    distinct_ys_test = len(set([tuple(y) for y in test_ys]))
    print(f'Successfully loaded Knapsack dataset.\n'
          f'Number of distinct solutions in train set: {distinct_ys_train},\n'
          f'Number of distinct solutions in test set: {distinct_ys_test}')

    metadata = {"variable_range": variable_range,
                "num_variables": num_variables}

    return (train_iterator, test_iterator), metadata

########################################################################################################################


def stochastic_weights_kp_dataloader(dataset_path: str,
                                     loader_params: Dict,
                                     num_items: int,
                                     seed: int,
                                     rnd_split_seed: int,
                                     penalty: float) -> Tuple[Tuple[data.DataLoader, data.DataLoader], Dict]:
    """
    The dataloader for the KP with unknown item weights.
    :param dataset_path: str; base loadpath.
    :param loader_params: dict; additional loader parameters from the configuration file.
    :param num_items: int; the KP dimension.
    :param seed: int; the numpy random seed used to identify the dataset.
    :param rnd_split_seed: int; the random seed used to split dataset.
    :return: training and test data loaders and the metadata.
    """

    dataset_path = os.path.join(dataset_path, f'seed-{seed}')

    # Decision variables range
    variable_range = dict(lb=0, ub=1)

    num_variables = num_items

    # FIXME: lines too long
    features_filepath = os.path.join(dataset_path, f'features_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0.csv')
    features = pd.read_csv(features_filepath, index_col=0).values
    solutions_filepath = os.path.join(dataset_path, f'solutions_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0.npy')
    ys = compute_normalized_solution(np.load(solutions_filepath), **variable_range)
    weights_filepath = os.path.join(dataset_path, f'targets_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0.csv')
    weights = pd.read_csv(weights_filepath, index_col=0).values
    values_filepath = os.path.join(dataset_path, f'values_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0.csv')
    values = pd.read_csv(values_filepath, index_col=0).values
    # Switch from dimension (num_items, 1) to (1, num_items)
    values = np.swapaxes(values, 0, 1)

    capacity = np.load(os.path.join(dataset_path, f'capacity_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0.npy'))

    # We can eventually have a different capacity and item values for each instance; it is not the case for the current
    # problem setup, so we simply repeat it
    tiled_capacity = np.expand_dims(np.expand_dims(capacity, 0), axis=1)
    tiled_capacity = np.tile(tiled_capacity, (len(features), 1))
    tiled_values = np.tile(values, (len(features), 1))

    features = np.concatenate((features, tiled_values, weights, tiled_capacity), axis=1)
    capacity = capacity.item()

    # FIXME: repeated code
    # Split between training and test sets
    train_val_features, test_features, \
    train_val_ys, test_ys, train_val_weights, test_weights = \
        train_test_split(features, ys, weights, test_size=0.2, random_state=rnd_split_seed)

    train_features, val_features, \
    train_ys, val_ys, train_weights, val_weights = \
        train_test_split(train_val_features, train_val_ys, train_val_weights, test_size=0.1, random_state=rnd_split_seed)

    min_weight = np.min(train_weights)
    max_weight = np.max(train_weights)

    train_dataset = list(zip(train_features, train_ys))
    training_set = Dataset(train_dataset)
    train_iterator = data.DataLoader(training_set, **loader_params)

    val_dataset = list(zip(val_features, val_ys))
    validation_set = Dataset(val_dataset)
    val_iterator = data.DataLoader(validation_set, **loader_params)

    test_dataset = list(zip(test_features, test_ys))
    test_set = Dataset(test_dataset)
    test_iterator = data.DataLoader(test_set, **loader_params)

    distinct_ys_train = len(set([tuple(y) for y in train_ys]))
    distinct_ys_test = len(set([tuple(y) for y in test_ys]))
    print(f'Successfully loaded Knapsack dataset.\n'
          f'Number of distinct solutions in train set: {distinct_ys_train},\n'
          f'Number of distinct solutions in test set: {distinct_ys_test}')

    # Additional optimization problem metadata
    metadata = {"variable_range": variable_range,
                "num_variables": num_variables,
                "capacity": capacity,
                "min_weight": min_weight,
                "max_weight": max_weight,
                "opt_prob_dir": 1}

    return (train_iterator, val_iterator, test_iterator), metadata

########################################################################################################################


def stochastic_capacity_kp_dataloader(dataset_path: str,
                                     loader_params: Dict,
                                     num_items: int,
                                     seed: int,
                                     rnd_split_seed: int,
                                     penalty: float) -> Tuple[Tuple[data.DataLoader, data.DataLoader], Dict]:
    """
    The dataloader for the KP with stochastic capacity.
    :param dataset_path: str; base loadpath.
    :param loader_params: dict; additional loader parameters from the configuration file.
    :param num_items: int; the KP dimension.
    :param seed: int; the numpy random seed used to identify the dataset.
    :param rnd_split_seed: int; the random seed used to split dataset.
    :return: training and test data loaders and the metadata.
    """

    dataset_path = os.path.join(dataset_path, f'seed-{seed}')

    # Decision variables range
    variable_range = dict(lb=0, ub=1)

    num_variables = num_items

    # FIXME: lines too long
    features_filepath = os.path.join(dataset_path, f'features_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_deg_5_correlation_type_1_rho_0.csv')
    features = pd.read_csv(features_filepath, index_col=0).values
    solutions_filepath = os.path.join(dataset_path, f'solutions_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_deg_5_correlation_type_1_rho_0.npy')
    ys = compute_normalized_solution(np.load(solutions_filepath), **variable_range)
    weights_filepath = os.path.join(dataset_path, f'weights_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_deg_5_correlation_type_1_rho_0.npy')
    weights = np.load(weights_filepath)
    values_filepath = os.path.join(dataset_path, f'values_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_deg_5_correlation_type_1_rho_0.npy')
    values = np.load(values_filepath)
    capacity_filepath = os.path.join(dataset_path, f'targets_n_1000_input_dim_5_output_dim_{num_items}_mult_noise_0.1_deg_5_correlation_type_1_rho_0.csv')
    capacity = pd.read_csv(capacity_filepath, index_col=0).values

    # We can eventually have a different item values and weights for each instance; it is not the case for the current
    # problem setup, so we simply repeat it
    tiled_weights = np.expand_dims(weights, axis=0)
    tiled_values = np.expand_dims(values, axis=0)
    tiled_weights = np.tile(tiled_weights, (len(features), 1))
    tiled_values = np.tile(tiled_values, (len(features), 1))

    # Since the optimization problem parameters are concatenated to the input features, we need easy way to unpack the
    # input tensor and know the position of each item

    features_start_idx = 0
    features_end_idx = features.shape[1]

    values_start_idx = features_end_idx
    values_end_idx = values_start_idx + tiled_values.shape[1]

    weights_start_idx = values_end_idx
    weights_end_idx = weights_start_idx + tiled_weights.shape[1]

    capacity_start_idx = weights_end_idx
    capacity_end_idx = capacity_start_idx + 1

    features = np.concatenate((features, tiled_values, tiled_weights, capacity), axis=1)

    # FIXME: repeated code
    # Split between training and test sets
    train_val_features, test_features, \
    train_val_ys, test_ys, train_val_capacity, test_capacity = \
        train_test_split(features, ys, capacity, test_size=0.2, random_state=rnd_split_seed)

    min_capacity = np.min(train_val_capacity)
    max_capacity = np.max(train_val_capacity)

    train_features, val_features, \
    train_ys, val_ys, train_capacity, val_capacity = \
        train_test_split(train_val_features, train_val_ys, train_val_capacity, test_size=0.1, random_state=rnd_split_seed)

    train_dataset = list(zip(train_features, train_ys))
    training_set = Dataset(train_dataset)
    train_iterator = data.DataLoader(training_set, **loader_params)

    val_dataset = list(zip(val_features, val_ys))
    validation_set = Dataset(val_dataset)
    val_iterator = data.DataLoader(validation_set, **loader_params)

    test_dataset = list(zip(test_features, test_ys))
    test_set = Dataset(test_dataset)
    test_iterator = data.DataLoader(test_set, **loader_params)

    distinct_ys_train = len(set([tuple(y) for y in train_ys]))
    distinct_ys_test = len(set([tuple(y) for y in test_ys]))
    print(f'Successfully loaded Knapsack dataset.\n'
          f'Number of distinct solutions in train set: {distinct_ys_train},\n'
          f'Number of distinct solutions in test set: {distinct_ys_test}')

    # Additional optimization problem metadata
    metadata = {"variable_range": variable_range,
                "num_variables": num_variables,
                "min_capacity": min_capacity,
                "max_capacity": max_capacity,
                "features_start_idx": features_start_idx,
                "features_end_idx": features_end_idx,
                "values_start_idx": values_start_idx,
                "values_end_idx": values_end_idx,
                "weights_start_idx": weights_start_idx,
                "weights_end_idx": weights_end_idx,
                "capacity_start_idx": capacity_start_idx,
                "capacity_end_idx": capacity_end_idx,
                "opt_prob_dir": 1}

    return (train_iterator, val_iterator, test_iterator), metadata

########################################################################################################################


def stochastic_set_cover_dataloader(dataset_path: str,
                                    loader_params: Dict,
                                    seed: int,
                                    rnd_split_seed: int,
                                    num_prods: int,
                                    num_sets: int,
                                    penalty: float) -> Tuple[Tuple[data.DataLoader, data.DataLoader], Dict]:
    """
    The data loader for the weighted set multi-cover with stochastic demands (aka coverage requirements).
    :param dataset_path: str; the base loadpath.
    :param loader_params: dict; additional loader parameters from the configuration file.
    :param seed: int; the numpy random seed used to identify the dataset.
    :param rnd_split_seed: int; the random seed used to split dataset.
    :param num_prods: int; the number of products (aka items).
    :param num_sets: int; the number of sets.
    :return: training and test data loaders and the metadata.
    """

    dataset_path = os.path.join(dataset_path,
                                f'{num_prods}x{num_sets}',
                                f'penalty-{float(penalty)}',
                                f'seed-{seed}')

    features = pd.read_csv(os.path.join(dataset_path, 'features.csv'), index_col=0).values
    ys = np.load(os.path.join(dataset_path, 'solutions.npy'))
    availability = np.load(os.path.join(dataset_path, 'availability.npy'))
    prod_costs = np.load(os.path.join(dataset_path, 'prod_costs.npy'))
    set_costs = np.load(os.path.join(dataset_path, 'set_costs.npy'))
    demands = pd.read_csv(os.path.join(dataset_path, 'targets.csv'), index_col=0).values

    min_y = int(np.min(ys))
    max_y = int(np.max(ys))

    # Decision variables range
    variable_range = dict(lb=min_y, ub=max_y)

    ys = compute_normalized_solution(ys, **variable_range)

    with open(os.path.join(dataset_path, 'wsmc_dim.json'), 'r') as file:
        problem_dim = json.load(file)

    # The number of decision variables
    num_variables = problem_dim['num_sets']

    availability = availability.reshape(-1)
    # We can eventually have a different availability matrix, product and set costs for each instance; it is not the
    # case for the current problem setup, so we simply repeat them
    tiled_availability = [availability for _ in range(len(features))]
    tiled_availability = np.asarray(tiled_availability)
    tiled_prod_costs = [prod_costs for _ in range(len(features))]
    tiled_prod_costs = np.asarray(tiled_prod_costs)
    tiled_set_costs = [set_costs for _ in range(len(features))]
    tiled_set_costs = np.asarray(tiled_set_costs)

    # Since the optimization problem parameters are concatenated to the input features, we need easy way to unpack the
    # input tensor and know the position of each item
    features_start_idx = 0
    features_end_idx = features.shape[1]

    avlbty_start_idx = features_end_idx
    avlbty_end_idx = avlbty_start_idx + tiled_availability.shape[1]

    prod_costs_start_idx = avlbty_end_idx
    prod_costs_end_idx = prod_costs_start_idx + tiled_prod_costs.shape[1]

    set_costs_start_idx = prod_costs_end_idx
    set_costs_end_idx = set_costs_start_idx + tiled_set_costs.shape[1]

    demands_start_idx = set_costs_end_idx
    demands_end_idx = demands_start_idx + demands.shape[1]

    features = np.concatenate((features, tiled_availability, tiled_prod_costs, tiled_set_costs, demands), axis=1)

    # FIXME: repeated code
    # Split training and test sets
    train_features, test_features, \
    train_ys, test_ys, train_demands, _ = \
        train_test_split(features, ys, demands, test_size=0.2, random_state=rnd_split_seed)
    train_dataset = list(zip(train_features, train_ys))
    training_set = Dataset(train_dataset)
    train_iterator = data.DataLoader(training_set, **loader_params)

    min_demand = np.min(train_demands)
    max_demand = np.max(train_demands)

    test_dataset = list(zip(test_features, test_ys))
    test_set = Dataset(test_dataset)
    test_iterator = data.DataLoader(test_set, **loader_params)

    distinct_ys_train = len(set([tuple(y) for y in train_ys]))
    distinct_ys_test = len(set([tuple(y) for y in test_ys]))
    print(f'Successfully loaded Knapsack dataset.\n'
          f'Number of distinct solutions in train set: {distinct_ys_train},\n'
          f'Number of distinct solutions in test set: {distinct_ys_test}')

    # Additional optimization problem metadata
    metadata = {"variable_range": variable_range,
                "num_variables": num_variables,
                "opt_prob_dir": -1,
                "min_demand": min_demand,
                "max_demand": max_demand,
                "features_start_idx": features_start_idx,
                "features_end_idx": features_end_idx,
                "availability_start_idx": avlbty_start_idx,
                "availability_end_idx": avlbty_end_idx,
                "prod_costs_start_idx": prod_costs_start_idx,
                "prod_costs_end_idx": prod_costs_end_idx,
                "set_costs_start_idx": set_costs_start_idx,
                "set_costs_end_idx": set_costs_end_idx,
                "demands_start_idx": demands_start_idx,
                "demands_end_idx": demands_end_idx}

    return (train_iterator, test_iterator), metadata

########################################################################################################################


class Dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = [torch.from_numpy(_x) for _x in self.dataset[index]]
        return x, y


# FIXME: not tested
def gen_constraints_dataset(train_dataset_size, test_dataset_size, seed, variable_range, num_variables,
                            num_constraints, positive_costs, constraint_params):
    np.random.seed(seed)
    constraints = sample_constraints(variable_range=variable_range,
                                     num_variables=num_variables,
                                     num_constraints=num_constraints,
                                     seed=seed, **constraint_params)
    metadata = dict(true_constraints=constraints, num_variables=num_variables, num_constraints=num_constraints,
                    variable_range=variable_range)

    c_l = []
    y_l = []
    dataset = []
    for _ in range(test_dataset_size + train_dataset_size):
        cost_vector = 2 * (np.random.rand(constraints.shape[1] - 1) - 0.5)
        if positive_costs:
            cost_vector = np.abs(cost_vector)
        y = ilp_solver(cost_vector=cost_vector, constraints=constraints, **variable_range)[0]
        y_norm = compute_normalized_solution(y, **variable_range)
        dataset.append((cost_vector, y_norm))
        c_l.append(cost_vector)
        y_l.append(y)
    cs, ys = np.stack(c_l, axis=0), np.stack(y_l, axis=0)

    num_distinct_ys = len(set([tuple(y) for _, y in dataset]))
    ys_uncon = solve_unconstrained(cs, **variable_range)
    match_boxconst_solution_acc = check_equal_ys(y_1=ys, y_2=ys_uncon)[1].mean()
    metrics = dict(num_distinct_ys=num_distinct_ys, match_boxconst_solution_acc=match_boxconst_solution_acc)
    print(f'Num distinct ys: {num_distinct_ys}, Match boxconst acc: {match_boxconst_solution_acc}')

    test_set = dataset[:test_dataset_size]
    train_set = dataset[test_dataset_size:]
    datasets = dict(metadata=metadata, train=train_set, test=test_set)
    return datasets, metrics


# FIXME: not tested
def main(working_dir, num_seeds, num_constraints, num_variables, data_gen_params):
    avg_meter = AvgMeters()
    all_metrics = {}
    for num_const, num_var in zip(num_constraints, num_variables):
        print(f'Gnerating dataset with {num_var} variables and {num_const} constraints...')
        for seed in range(num_seeds):
            dir = os.path.join(working_dir, str(num_var) + "_dim", str(num_const) + "_const", str(seed))
            os.makedirs(dir, exist_ok=True)
            datasets, metrics = gen_constraints_dataset(seed=seed, num_variables=num_var,
                                                        num_constraints=num_const, **data_gen_params)
            save_pickle(datasets, os.path.join(dir, 'dataset.p'))
            avg_meter.update(metrics)
        all_metrics.update(
            avg_meter.get_averages(prefix=str(num_var) + "_dim_" + str(num_const) + "_const_"))
        avg_meter.reset()
    save_dict_as_one_line_csv(all_metrics, filename=os.path.join(working_dir, "metrics.csv"))
    return all_metrics


if __name__ == "__main__":
    param_path = sys.argv[1]
    param_dict = load_with_default_yaml(path=param_path)
    main(**param_dict)
