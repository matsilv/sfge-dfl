"""
    Data generation for the KP. The procedure is very similar to what is described in the shortest path version of [1]
    Elmachtoub, Adam N., and Paul Grigas. "Smart “predict, then optimize”." Management Science 68.1 (2022): 9-26.
"""

import random
import os
import math
import numpy as np
import pandas as pd
import argparse

import torch
from tqdm import tqdm

from data.generation_scripts.utility import bernoulli, set_seeds
from data.generation_scripts import DATAPATH_PREFIX
from optimization_problems.knapsack_problem import KnapsackProblem

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
    :param seed: int; numpy random seed.
    :return:
    """

    # Output files path
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    suffix += f'_mult_noise_{multiplicative_noise}_add_noise_{additive_noise}_deg_{deg}'
    suffix += f'_relative_capacity_{relative_capacity}_correlation_type_{correlate_values_and_weights}'
    suffix += f'_rho_{rho}.csv'

    prefix = os.path.join(datapath, f'seed-{seed}')

    file_output_targets = os.path.join(prefix, 'targets' + suffix)
    file_output_features = os.path.join(prefix, 'features' + suffix)
    file_output_weights = os.path.join(prefix, 'weights' + suffix)
    file_output_solutions = os.path.join(prefix, 'solutions' + suffix)

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    cost_df = pd.DataFrame(index=np.arange(num_instances),
                           columns=[f'c_{idx}' for idx in range(output_dim)],
                           dtype=np.float64)
    input_features_df = pd.DataFrame(index=np.arange(num_instances),
                                     columns=['at{0}'.format(i + 1) for i in range(input_dim)],
                                     dtype=np.float64)
    solutions_df = pd.DataFrame(index=np.arange(num_instances),
                                columns=[f'x_{idx}' for idx in range(output_dim)],
                                dtype=np.float64)
    weights_df = pd.Series(index=np.arange(output_dim),
                           data=weights,
                           dtype=np.float64)

    # Create the optimization problem instance to compute the optimal solutions
    opt_problem = KnapsackProblem(dim=output_dim)

    # Define the capacity as a relative value of the sum of the weights
    sum_of_weights = weights.sum()
    capacity = sum_of_weights * relative_capacity

    # These are the optimization problem parameters that are shared among all the instances
    opt_prob_params = {'weights': torch.as_tensor(weights), 'capacity': torch.as_tensor(capacity)}

    # B^{\star} as described in [1]
    bernoulli_matrix = np.array([[bernoulli(0.5) for _ in range(input_dim)] for _ in range(output_dim)])

    # For each instance...
    for i in tqdm(range(num_instances), total=num_instances, desc='Data generation'):
        # The input features are random values in [0, 1] generated according to a Gaussian distributions as described
        # in [1]
        x = np.array([round(random.gauss(0, 1), 3) for _ in range(input_dim)])
        input_features_df.iloc[i] = x

        # Only a subset of input features affects the targets
        b_matmul_x = np.matmul(bernoulli_matrix, x)

        # For each target dimension...
        for j in range(output_dim):
            # Generate the true model
            pred = b_matmul_x[j]

            # Noisy polinomial relationship as described in [1]
            val = 1 + (pred / math.sqrt(input_dim) + 3) ** deg
            val = val * random.uniform(1 - multiplicative_noise, 1 + multiplicative_noise)

            # Optionally, add a correlation between weights and items values
            if correlate_values_and_weights == 1:
                val *= weights[j]
            elif correlate_values_and_weights == 2:
                val = rho * weights[j] + (1 - rho) * val

            val = round(val + additive_noise, 5)

            cost_df.iloc[i][f'c_{j}'] = val

        # Compute and store the optimal solution
        instance_costs = cost_df.iloc[i].values
        sol, _ = opt_problem.solve(y=instance_costs, opt_prob_params=opt_prob_params)
        solutions_df.iloc[i] = sol

    # Save results on a file
    input_features_df.to_csv(file_output_features)
    cost_df.to_csv(file_output_targets)
    weights_df.to_csv(file_output_weights)
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
    parser.add_argument("--correlate-values-weights", type=int, choices=[0, 1, 2],default=0,
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

    for seed in seeds:

        set_seeds(seed)

        weights = np.random.uniform(0, 1, output_dim)
        generate_instances(datapath=os.path.join(DATAPATH_PREFIX, 'knapsack'),
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
