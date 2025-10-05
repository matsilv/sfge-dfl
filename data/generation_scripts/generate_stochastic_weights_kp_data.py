import random
import os
import math
import numpy as np
import pandas as pd
import argparse

import torch
from sklearn.preprocessing import MinMaxScaler

from data.generation_scripts.utility import bernoulli, set_seeds
from optimization_problems.stochastic_weights_kp import StochasticWeightsKnapsackProblem
from optimization_problems import PENALTY_COST
from tqdm import tqdm

########################################################################################################################


def generate_instances(datapath: str,
                       num_instances: int,
                       penalty: float,
                       input_dim: int = 5,
                       output_dim: int = 48,
                       deg: int = 2,
                       multiplicative_noise: float = 0.5,
                       additive_noise: float = 0,
                       values: np.ndarray = None,
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
    :param relative_capacity: float; the KP capacity is expressed in terms of the sum of the weights.
    :param correlate_values_and_weights: int; introduce a correlation between weights and items values.
    :param rho: float; coefficient of the correlation between weights and values.
    :param seed: int; numpy random seed.
    :return:
    """

    # Set numpy random seed
    np.random.seed(seed)

    # Output files path
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    suffix += f'_mult_noise_{multiplicative_noise}_add_noise_{additive_noise}_deg_{deg}'
    suffix += f'_relative_capacity_{relative_capacity}_correlation_type_{correlate_values_and_weights}'
    suffix += f'_rho_{rho}'

    prefix = os.path.join(datapath, f'seed-{seed}')

    file_output_values = os.path.join(prefix, 'values' + suffix + '.csv')
    file_output_features = os.path.join(prefix, 'features' + suffix + '.csv')
    file_output_targets = os.path.join(prefix, 'targets' + suffix + '.csv')
    file_capacity = os.path.join(prefix, 'capacity' + suffix + '.npy')
    file_solutions = os.path.join(prefix, 'solutions' + suffix + '.csv')

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Create the dataframes with input features and weights
    input_features_df = pd.DataFrame(index=np.arange(num_instances),
                                     columns=['at{0}'.format(i + 1) for i in range(input_dim)])
    weights_df = pd.DataFrame(columns=[f'w_{idx}' for idx in range(output_dim)],
                              index=np.arange(num_instances),
                              dtype=float)
    solutions_df = pd.DataFrame(index=np.arange(num_instances),
                                columns=[f'x_{idx}' for idx in range(output_dim)])

    # Create one Bernoulli distribution for the entire instance
    B = np.array([[bernoulli(0.5) for k in range(input_dim)] for e in range(output_dim)])

    # Keep track of the weights sum for each instance; this is needed to compute the capacity
    all_item_weights = list()

    # For each instance...
    for i in tqdm(range(num_instances), total=num_instances, desc='Generating instances'):
        # Keep track of Poisson rate for each product
        lmbd_list = list()

        # The input features are random values in [0, 1] generated according to a Gaussian distributions
        x = np.array([round(random.gauss(0, 1), 3) for _ in range(input_dim)])
        input_features_df.iloc[i] = x

        # Only a subset of input features affects the targets
        B_matmul_x = np.matmul(B, x)

        # For each target dimension...
        for j in range(output_dim):
            # Generate the true model
            pred = B_matmul_x[j]

            # Noisy polinomial relationship between the input features and the Poisson rate
            lmbd_val = 1 + (pred / math.sqrt(input_dim) + 3) ** deg
            lmbd_val *= random.uniform(1 - multiplicative_noise, 1 + multiplicative_noise)
            lmbd_val = round(lmbd_val + additive_noise)
            lmbd_list.append(lmbd_val)

        weights_val = np.random.poisson(lmbd_list, size=output_dim)
        weights_df.iloc[i] = weights_val

        # Store the sum of item weights for current instance
        all_item_weights.append(np.sum(weights_val))

    # Optionally, add a correlation between weights and items values
    if correlate_values_and_weights == 1:
        scaler = MinMaxScaler()
        std_weights = scaler.fit_transform(weights_df)
        values = values * std_weights
        values = np.sum(values, axis=0)

    # Create a dataframe with the item values
    values_df = pd.Series(index=np.arange(output_dim), data=values)

    # To compute the capacity, we first compute the mean of the item weights for each instance and then we multiply this
    # value by the relative capacity and convert to closes integer
    capacity = round(np.mean(all_item_weights) * relative_capacity)

    # Compute the optimal solutions (the penalty does not matter since it is only used to compute the cost of the
    # recourse action)
    opt_problem = StochasticWeightsKnapsackProblem(dim=output_dim, penalty=penalty)
    opt_prob_params = {'values': torch.as_tensor(values), 'capacity': torch.as_tensor(capacity)}

    for idx in tqdm(range(num_instances), total=num_instances, desc='Solving instances'):
        weights = weights_df.iloc[idx].values

        sol, _ = opt_problem.solve(y=weights, opt_prob_params=opt_prob_params)
        cost, _ = \
            opt_problem.get_objective_values(y=torch.as_tensor(weights),
                                             sols=torch.as_tensor(sol),
                                             opt_prob_params=opt_prob_params)
        solutions_df.iloc[idx] = sol

        # NOTE: this is not actually required, but we opted for a penalty value that discourages recourse actions;
        # ideally the optimal solution does not require any recourse action
        assert cost[PENALTY_COST] == 0

    # Save item values, weights, capacity and input features to file
    np.save(file_capacity, capacity)
    values_df.to_csv(file_output_values)
    weights_df.to_csv(file_output_targets)
    input_features_df.to_csv(file_output_features)
    solutions_df.to_csv(file_solutions)

########################################################################################################################


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True, help="Number of input features")
    parser.add_argument("--output-dim", type=int, required=True, help="Number of output features")
    parser.add_argument("--penalty", type=float, required=True,
                        help="The penalty value associated with the recourse action")
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
    rho = args.rho
    seeds = args.seeds
    penalty = args.penalty

    for seed in seeds:

        set_seeds(seed)

        values = np.random.uniform(0, 1, output_dim)
        generate_instances(datapath=os.path.join('data', 'data', 'stochastic_weights_kp'),
                           num_instances=num_instances,
                           penalty=penalty,
                           input_dim=input_dim,
                           output_dim=output_dim,
                           deg=deg,
                           multiplicative_noise=multiplicative_noise,
                           additive_noise=additive_noise,
                           values=values,
                           relative_capacity=relative_capacity,
                           correlate_values_and_weights=correlate_values_and_weights,
                           rho=rho,
                           seed=seed)
