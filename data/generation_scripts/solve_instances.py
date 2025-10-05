import json
import os.path
from tqdm import tqdm
import numpy as np
import pandas as pd

from optimization_problems.set_cover_problem import StochasticWeightedSetMultiCover
from optimization_problems.fractional_kp import FractionalKPProblem
from optimization_problems.stochastic_capacity_kp import StochasticCapacityKnapsackProblem
from optimization_problems.stochastic_weights_kp import StochasticWeightsKnapsackProblem

########################################################################################################################


def solve_stochastic_weights_kp(values_filepath, capacity_filepath, targets_filepath, savepath):
    values = pd.read_csv(values_filepath, index_col=0).values
    values = np.squeeze(values)
    weights = pd.read_csv(targets_filepath, index_col=0).values
    capacity = np.load(capacity_filepath)
    capacity = np.asarray([capacity])

    opt_prob_params = np.concatenate((values, capacity), axis=-1)
    problem = StochasticKnapsackProblem(dim=len(values), penalty=0)

    solutions = list()

    for wgt in tqdm(weights, total=len(weights), desc='Solving optimization problems'):
        sol, _ = problem.solve(y=wgt, opt_prob_params=opt_prob_params)
        solutions.append(sol)

    solutions = np.asarray(solutions)
    np.save(savepath, solutions)

########################################################################################################################


def solve_wsmc_instances(availability_filepath,
                         prod_costs_filepath,
                         set_costs_filepath,
                         targets_filepath,
                         wsmc_dim_filepath,
                         savepath):

    availability = np.load(availability_filepath)
    prod_costs = np.load(prod_costs_filepath)
    set_costs = np.load(set_costs_filepath)
    targets = pd.read_csv(targets_filepath, index_col=0).values

    with open(wsmc_dim_filepath, 'r') as file:
        wsmc_dim = json.load(file)

    problem = StochasticWeightedSetMultiCover(num_sets=wsmc_dim['num_sets'], num_products=wsmc_dim['num_prods'])

    solutions = list()

    for demands in tqdm(targets, total=len(targets), desc='Solving optimization problems'):
        opt_prob_params = set_costs
        opt_prob_params = np.concatenate((opt_prob_params, prod_costs), axis=-1)
        opt_prob_params = np.concatenate((opt_prob_params, availability.reshape(-1)), axis=-1)

        sol, _ = problem.solve(y=demands, opt_prob_params=opt_prob_params)
        sol = sol.astype(np.int32)
        solutions.append(sol)

    solutions = np.asarray(solutions)
    np.save(os.path.join(savepath, 'solutions'), solutions)

########################################################################################################################


def solve_fractional_kp(values_filepath, weights_filepath, capacity, savepath):

    values = pd.read_csv(values_filepath, index_col=0).values
    weights = pd.read_csv(weights_filepath, index_col=0).values
    num_items = values.shape[1]
    penalties = np.zeros(shape=num_items)

    opt_problem = FractionalKPProblem(dim=num_items)

    solutions = list()

    for val, wgt in tqdm(zip(values, weights), total=len(values), desc='Solving optimization problems'):
        opt_prob_params = {'penalty': penalties, 'capacity': capacity}
        sol, _ = opt_problem.solve(y=np.concatenate((val, wgt), axis=-1),
                                   opt_prob_params=opt_prob_params)
        solutions.append(sol)

    solutions = np.asarray(solutions)
    solutions = pd.DataFrame(data=solutions, columns=[f'x_{idx}' for idx in range(num_items)], index=np.arange(len(values)))
    solutions.to_csv(os.path.join(savepath, f'solutions-{capacity}.csv'))

########################################################################################################################


def solve_stochastic_capacity_kp(values_filepath, weights_filepath, targets_filepath, savepath):
    values = np.load(values_filepath)
    weights = np.load(weights_filepath)
    capacity = pd.read_csv(targets_filepath, index_col=0).values

    opt_prob_params = {'values': values, 'weights': weights}
    problem = StochasticCapacityKnapsackProblem(dim=len(values), penalty=0)

    solutions = list()

    for cpty in tqdm(capacity, total=len(capacity), desc='Solving optimization problems'):
        sol, _ = problem.solve(y=cpty, opt_prob_params=opt_prob_params)
        solutions.append(sol)

    solutions = np.asarray(solutions)
    np.save(savepath, solutions)

########################################################################################################################


if __name__ == '__main__':
    """for penalty in ['1.0', '5.0', '10.0']:
        for seed in range(5):
            filepath_prefix = os.path.join('wsmc', '10x50', f'penalty-{penalty}', f'seed-{seed}')
            availability_filepath = os.path.join(filepath_prefix, 'availability.npy')
            prod_costs_filepath = os.path.join(filepath_prefix, 'prod_costs.npy')
            set_costs_filepath = os.path.join(filepath_prefix, 'set_costs.npy')
            targets_filepath = os.path.join(filepath_prefix, 'targets.csv')
            wsmc_dim_filepath = os.path.join(filepath_prefix, 'wsmc_dim.json')

            solve_wsmc_instances(availability_filepath=availability_filepath,
                                 prod_costs_filepath=prod_costs_filepath,
                                 set_costs_filepath=set_costs_filepath,
                                 targets_filepath=targets_filepath,
                                 wsmc_dim_filepath=wsmc_dim_filepath,
                                 savepath=filepath_prefix)"""

    filepath_prefix = 'fractional_kp'
    values_filepath = os.path.join(filepath_prefix, 'values.csv')
    weights_filepath = os.path.join(filepath_prefix, 'weights.csv')

    for capacity in [50, 75]:
        solve_fractional_kp(values_filepath=values_filepath,
                            weights_filepath=weights_filepath,
                            capacity=capacity,
                            savepath=filepath_prefix)

    """for seed in range(5):
        for suffix in ['_n_1000_input_dim_5_output_dim_50_mult_noise_0.1_deg_5_correlation_type_1_rho_0',
                       '_n_1000_input_dim_5_output_dim_75_mult_noise_0.1_deg_5_correlation_type_1_rho_0']:
            filepath_prefix = os.path.join('stochastic_capacity_kp', f'seed-{seed}')
            values_filepath = os.path.join(filepath_prefix, f'values{suffix}.npy')
            weights_filepath = os.path.join(filepath_prefix, f'weights{suffix}.npy')
            targets_filepath = os.path.join(filepath_prefix, f'targets{suffix}.csv')
            savepath = os.path.join(filepath_prefix, 'solutions' + suffix)

            solve_stochastic_capacity_kp(values_filepath=values_filepath,
                                         weights_filepath=weights_filepath,
                                         targets_filepath=targets_filepath,
                                         savepath=savepath)"""

    """for seed in range(5):
        for suffix in ['_n_1000_input_dim_5_output_dim_50_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0',
                       '_n_1000_input_dim_5_output_dim_75_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0']:
            filepath_prefix = os.path.join('stochastic_weights_kp', f'seed-{seed}')
            values_filepath = os.path.join(filepath_prefix, f'values{suffix}.csv')
            capacity_filepath = os.path.join(filepath_prefix, f'capacity{suffix}.npy')
            targets_filepath = os.path.join(filepath_prefix, f'targets{suffix}.csv')
            savepath = os.path.join(filepath_prefix, 'solutions' + suffix)

            solve_stochastic_weights_kp(values_filepath=values_filepath,
                                        capacity_filepath=capacity_filepath,
                                        targets_filepath=targets_filepath,
                                        savepath=savepath)"""
