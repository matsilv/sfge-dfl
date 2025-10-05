import random
import os
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm

from data.generation_scripts.utility import bernoulli, set_seeds
from optimization_problems.stochastic_capacity_kp import StochasticCapacityKnapsackProblem
from optimization_problems import PENALTY_COST

########################################################################################################################


def generate_instances(datapath: str,
                       num_instances: int,
                       input_dim: int,
                       # FIXME: output_dim should be replaced with something like problem_dim
                       output_dim: int,
                       deg: int,
                       multiplicative_noise: float,
                       values: np.ndarray,
                       weights: np.ndarray,
                       correlate_values_and_weights: int,
                       rho: float,
                       penalty: float,
                       rnd_seed: int = 0):
    """
    Generate input features, values vector and weights of the KP.
    :param datapath: str; where instances are saved to.
    :param num_instances: int; number of instances to generate.
    :param input_dim: int; number of input features.
    :param output_dim: int; number of output features.
    :param deg: int; degree of the polynomial relationship between the input features and the cost vectors.
    :param multiplicative_noise: float; the multiplicative noise of the relationship.
    :param weights: numpy.ndarray; the weights of the knapsack items.
    :param values: numpy.ndarray; the values of the knapsack items.
    :param correlate_values_and_weights: int; introduce a correlation between weights and items values.
    :param rho: float; coefficient of the correlation between weights and values.
    :param rnd_seed: int; numpy random seed.
    :return:
    """

    # Suffix of the output file paths
    suffix = f'_n_{num_instances}_input_dim_{input_dim}_output_dim_{output_dim}'
    suffix += f'_mult_noise_{multiplicative_noise}_deg_{deg}'
    suffix += f'_correlation_type_{correlate_values_and_weights}'
    suffix += f'_rho_{rho}'

    # Prefix of the output file paths
    prefix = os.path.join(datapath, f'seed-{rnd_seed}')

    # Output file paths
    file_item_values = os.path.join(prefix, 'values' + suffix + '.npy')
    file_input_features = os.path.join(prefix, 'features' + suffix + '.csv')
    file_capacity = os.path.join(prefix, 'targets' + suffix + '.csv')
    file_item_weights = os.path.join(prefix, 'weights' + suffix + '.npy')
    file_solutions = os.path.join(prefix, 'solutions' + suffix + '.csv')

    # Make the directory if it does not exist
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Create the dataframes with input features and capacity values
    input_features_df = pd.DataFrame(index=np.arange(num_instances),
                                     columns=['at{0}'.format(i + 1) for i in range(input_dim)])
    capacity_df = pd.DataFrame(index=np.arange(num_instances), columns=['Capacity'], dtype=int)
    solutions_df = pd.DataFrame(index=np.arange(num_instances), columns=[f'x_{idx}' for idx in range(output_dim)])

    # Create one Bernoulli distribution for the entire instance
    B = np.array([bernoulli(0.5) for _ in range(input_dim)])

    opt_problem = StochasticCapacityKnapsackProblem(dim=output_dim, penalty=penalty)

    opt_prob_params = {'weights': torch.as_tensor(weights),
                       'values': torch.as_tensor(values)}

    # Optionally, add a correlation between weights and items values
    if correlate_values_and_weights == 1:
        values *= weights
    elif correlate_values_and_weights == 2:
        values = rho * values + (1 - rho) * values

    # For each instance...
    for i in tqdm(range(num_instances), total=num_instances, desc='Computing optimal solutions'):

        # The input features are random values in [0, 1] generated according to a Gaussian distributions
        x = np.array([round(random.gauss(0, 1), 3) for _ in range(input_dim)])
        # The distribution we employ to generate the capacity values is a Beta distribution and the alpha and beta
        # values can not be negative
        x = np.abs(x)
        input_features_df.iloc[i] = x

        # Only a subset of input features affects the targets
        B_matmul_x = np.matmul(B, x)

        # Noisy polinomial relationship between the input features and alpha and beta of the Beta distribution
        val = 1 + B_matmul_x ** deg
        val *= random.uniform(1 - multiplicative_noise, 1 + multiplicative_noise)

        # The relative capacity value is distributed according to a beta distribution; its minimum value is 0.1
        relative_capacity_val = np.random.beta(a=val, b=val)
        relative_capacity_val = np.clip(relative_capacity_val, a_min=0.1, a_max=np.inf)

        # The capacity is a fraction of the total item weights
        capacity = round(np.sum(weights) * relative_capacity_val)
        capacity_df.iloc[i] = capacity

        # Compute the optimal solution
        capacity = np.array([capacity])

        sol, _ = \
            opt_problem.solve(y=capacity, opt_prob_params=opt_prob_params, solve_to_optimality=True)

        cost, _ = \
            opt_problem.get_objective_values(y=torch.as_tensor(capacity),
                                             sols=torch.as_tensor(sol),
                                             opt_prob_params=opt_prob_params)

        # NOTE: this is not actually required, but we opted for a penalty value that discourages recourse actions;
        # ideally the optimal solution does not require any recourse action
        assert cost[PENALTY_COST] == 0

        solutions_df.iloc[i] = sol

    """for idx in range(input_dim):
        plt.scatter(input_features_df.iloc[:, idx], capacity_df.iloc[:, 0])
        plt.show()"""

    # Save item values, weights, capacity and input features to file
    capacity_df.to_csv(file_capacity)
    np.save(file_item_values, values)
    np.save(file_item_weights, weights)
    input_features_df.to_csv(file_input_features)
    solutions_df.to_csv(file_solutions)

########################################################################################################################


if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True, help="Number of input features")

    parser.add_argument("--output-dim", type=int, required=True, help="Number of output features")

    parser.add_argument("--degree", type=int, required=True, help="The degree of the polynomial function")

    parser.add_argument("--num-instances", type=int, required=True, help="The number of instances to generate")

    parser.add_argument("--multiplicative-noise", type=float, default=0,
                        help="The multiplicative noise added to the predictions")

    parser.add_argument("--penalty", type=float, required=True,
                        help="The penalty value associated with the recourse action")

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
    parsed_input_dim = args.input_dim
    parsed_output_dim = args.output_dim
    parsed_deg = args.degree
    parsed_num_instances = args.num_instances
    parsed_multiplicative_noise = args.multiplicative_noise
    parsed_correlate_values_and_weights = args.correlate_values_weights
    parsed_rho = args.rho
    parsed_seeds = args.seeds
    penalty = args.penalty

    for seed in parsed_seeds:

        set_seeds(seed)

        generated_values = np.random.uniform(0, 1, parsed_output_dim)
        generated_weights = np.random.uniform(0, 1, parsed_output_dim)

        generate_instances(datapath=os.path.join('data', 'data', 'stochastic_capacity_kp'),
                           num_instances=parsed_num_instances,
                           input_dim=parsed_input_dim,
                           output_dim=parsed_output_dim,
                           deg=parsed_deg,
                           penalty=penalty,
                           multiplicative_noise=parsed_multiplicative_noise,
                           values=generated_values,
                           weights=generated_weights,
                           correlate_values_and_weights=parsed_correlate_values_and_weights,
                           rho=parsed_rho,
                           rnd_seed=seed)
