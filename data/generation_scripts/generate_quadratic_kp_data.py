"""
    NOTE. I kept the data generation process of the quadratic KP01 separated.
"""

import random
import os
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import argparse

from optimization_problems.quadratic_kp_problem import PROBLEM_ID, QuadraticKP
from data.generation_scripts.utility import bernoulli, set_seeds
from data.generation_scripts import DATAPATH_PREFIX

########################################################################################################################


def generate_instances(datapath: str,
                       num_instances: int,
                       input_dim: int = 5,
                       output_dim: int = 48,
                       deg: int = 2,
                       multiplicative_noise: float = 0.5,
                       additive_noise: float = 0,
                       weights: np.ndarray = None,
                       relative_capacity: float = None,
                       correlate_values_and_weights: int = 0,
                       rho: float = 0.5,
                       seed: int = 0):
    """
    Generate input features, values vector and weights of the KP.
    :param datapath: str; where instances are saved to.
    :param num_instances: int; number of instances to generate.
    :param input_dim: int; number of input features.
    :param output_dim: int; number of output features.
    :param deg: int; degree of the polynomial relationship between the input features and the cost vectors.
    :param multiplicative_noise: float; the multiplicative noise of the relationship.
    :param additive_noise: float; the additive noise of the relationship.
    :param weights: numpy.ndarray; the weights of the knapsack items.
    :param relative_capacity: float; the KP capacity is expressed in terms of the sum of the weights.
    :param correlate_values_and_weights: int; introduce a correlation between weights and items values.
    :param rho: float; coefficient of the correlation between weights and values.
    :param seed: int; numpy randome seed.
    :return:
    """

    # Since the cost matrix is squared, check that output_dim is squared as well
    num_items = int(np.sqrt(output_dim))
    assert num_items**2 == output_dim, "The output_dim must be a perfect square"

    # This is the filename suffix that describes how data will be generated
    filepath_suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    filepath_suffix += f'_mult_noise_{multiplicative_noise}_add_noise_{additive_noise}_deg_{deg}'
    filepath_suffix += f'_relative_capacity_{relative_capacity}_correlation_type_{correlate_values_and_weights}'
    filepath_suffix += f'_rho_{rho}.csv'

    # Where the files will be saved to
    folder_path = os.path.join(datapath, PROBLEM_ID, f'seed-{seed}')

    # We have one file for targets, features and weights
    file_output_targets = os.path.join(folder_path, 'targets' + filepath_suffix)
    file_output_features = os.path.join(folder_path, 'features' + filepath_suffix)
    file_output_weights = os.path.join(folder_path, 'weights' + filepath_suffix)
    file_output_solutions = os.path.join(folder_path, 'solutions' + filepath_suffix)

    # Create the empty dataframes for targets and features
    index = np.arange(num_instances)

    # We have one cost for item and one cost for each pair of items
    cost_columns = [f'c_{i}_{j}' for i in range(num_items) for j in range(num_items)]

    # Create the dataframes with the instances parameters
    targets_df = pd.DataFrame(index=index,
                              columns=cost_columns,
                              dtype=np.float64)
    features_df = pd.DataFrame(index=index,
                               columns=[f'at{i}' for i in range(1, input_dim+1)],
                               dtype=np.float64)
    weights_df = pd.Series(index=np.arange(num_items), data=weights, dtype=np.float32)
    solutions_df = pd.DataFrame(index=np.arange(num_instances),
                                columns=[f'x_{idx}' for idx in range(num_items)],
                                dtype=np.float64)

    # Create the optimization problem instance to compute the optimal solutions
    opt_problem = QuadraticKP(dim=num_items)

    # Define the capacity as a relative value of the sum of the weights
    sum_of_weights = weights.sum()
    capacity = sum_of_weights * relative_capacity

    # These are the optimization problem parameters that are shared among all the instances
    opt_prob_params = {'weights': torch.as_tensor(weights), 'capacity': torch.as_tensor(capacity)}

    # Create the data folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create one model for the entire instance
    B = np.array([[bernoulli(0.5) for k in range(input_dim)] for e in range(output_dim)])

    # Iterate over all the instances
    for instance_idx in tqdm(range(num_instances), total=num_instances, desc='Data generation'):

        # Create the input features
        x = np.array([round(random.gauss(0, 1), 3) for _ in range(input_dim)])

        # A subset of the input features (chosen according to a Bernoulli distribution) will affect the items value
        B_matmul_x = np.matmul(B, x)

        # Create the value for each item and for each pair of items
        for i in range(num_items):
            for j in range(num_items):

                # Generate the true model
                pred = B_matmul_x[j]
                c_val = (1 + (pred / math.sqrt(input_dim) + 3) ** deg) * random.uniform(1 - multiplicative_noise,
                                                                                        1 + multiplicative_noise)

                # Optionally, we can add a correlation between the weights and items values
                if correlate_values_and_weights == 1:
                    c_val *= weights[j]
                elif correlate_values_and_weights == 2:
                    c_val = rho * weights[j] + (1 - rho) * c_val
                c_val = round(c_val + additive_noise, 5)

                # Assign the cost value to the costs matrix
                targets_df.loc[instance_idx][f'c_{i}_{j}'] = c_val

        # Save the input features in the dataframe
        features_df.loc[instance_idx] = x

        # Compute and store the optimal solution
        instance_costs = targets_df.iloc[instance_idx].values
        sol, _ = opt_problem.solve(y=instance_costs, opt_prob_params=opt_prob_params)
        solutions_df.iloc[instance_idx] = sol

    # Save dataframes to files
    weights_df.to_csv(file_output_weights)
    targets_df.to_csv(file_output_targets)
    features_df.to_csv(file_output_features)
    solutions_df.to_csv(file_output_solutions)

########################################################################################################################


if __name__ == '__main__':

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True, help="Number of input features")
    parser.add_argument("--output-dim", type=int, required=True, help="Number of output features")
    parser.add_argument("--relative-capacity",
                        type=float,
                        required=True,
                        help="The KP capacity is the multiplication between the relative capacity and the total " + \
                             "sum of the weights")
    parser.add_argument("--degree", type=int, required=True, help="The degree of the polynomial function")
    parser.add_argument("--num-instances", type=int, required=True, help="The number of instances to generate")
    parser.add_argument("--multiplicative-noise",
                        type=float,
                        default=0,
                        help="The multiplicative noise added to the predictions")
    parser.add_argument("--additive-noise",
                        type=float,
                        default=0,
                        help="The additive noise added to the predictions")
    parser.add_argument("--correlate-values-weights", type=int, choices=[0, 1, 2], default=0,
                        help="correlate_values_and_weights is a variable that can be used to introduce a " + \
                             "correlation between the values and the weights of the knapsack items. Value 0 does " + \
                             "not introduce any explicit correlation. Value 1 denotes that after generating values " + \
                             "from features in a manner analogous to the shortest path datageneration process " + \
                             "described in the SPO paper, each generated item value gets multiplied with the " + \
                             "associated weight. Value 2 denotes that the generated item value vector will be " + \
                             "linearly transformed introduce a correlation between the item value vector and the " + \
                             "cost value vector as follows: c <- rho * w + (1 - rho) * c.")
    parser.add_argument("--rho", type=float, default=0,
                        help="An increasing value of rho increases the correlation coefficient between c and w, " + \
                             "but rho does not itself denote the exact value of that correlation coefficient. " + \
                             "The value rho is only relevant when correlate_values_and_weights has value 2. " + \
                             "In other cases the value of rho is ignored).")
    parser.add_argument("--seeds", type=int, nargs='+', required=True, help="Numpy random seeds")

    # Parse the arguments
    args = parser.parse_args()
    input_dim = args.input_dim
    output_dim = args.output_dim
    relative_capacity = args.relative_capacity
    deg = args.degree
    num_instances = args.num_instances
    multiplicative_noise = args.multiplicative_noise
    additive_noise = args.additive_noise
    correlate_values_and_weights = args.correlate_values_weights
    rho = float(args.rho)
    seeds = args.seeds

    num_items = int(np.sqrt(output_dim))

    for seed in seeds:

        set_seeds(seed)

        weights = np.random.uniform(0, 1, num_items)
        generate_instances(datapath=DATAPATH_PREFIX,
                           num_instances=num_instances,
                           input_dim=input_dim,
                           output_dim=output_dim,
                           deg=deg,
                           multiplicative_noise=multiplicative_noise,
                           additive_noise=additive_noise,
                           weights=weights,
                           relative_capacity=relative_capacity,
                           correlate_values_and_weights=correlate_values_and_weights,
                           rho=rho,
                           seed=seed)
