"""
    Experiments for prediction-focused learning plus the SAA.
"""

import os
import argparse
import json

import numpy as np
import torch
from torch.random import manual_seed
from tqdm import tqdm
from tabulate import tabulate
import pickle
import shutil

from utils.build_data_loaders import build_data_loaders
from utils.build_models import build_models
from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems import TOTAL_COST
from methods.prob_regressor import ProbRegressorModule
from methods.sfge import SFGEModule
from dfl_module import DFLModule

from typing import List, Dict

########################################################################################################################

COST = 'Cost'
RELATIVE_REGRET = 'Relative regret'
RUNTIME = 'Runtime'

########################################################################################################################


def load_best_model(filepath: str, model: DFLModule) -> DFLModule:
    """
    Load the best model saved restored after early stopping.
    :param filepath: str; where the model is loaded from.
    :param model: dfl_module.DFLModule; instance of the same DFLModule subclass of the best model; this instance is used
                  as backbone for the loading function of PyTorch lighting..
    :return: dfl_module.DFLModule; the best model loaded from file.
    """

    # Get the best model filepath
    best_model_filepath = os.path.join(filepath, 'best-model')
    best_model_filename = os.listdir(best_model_filepath)

    # Sanity check
    assert len(best_model_filename) == 1, "A single file is expected in best-model folder"
    best_model_filename = best_model_filename[0]

    # Load the best model from file

    absolute_path = os.path.join(best_model_filepath, best_model_filename)

    best_model = \
        type(model).load_from_checkpoint(absolute_path,
                                   net=model.net,
                                   optimization_problem=model.optimization_problem)

    # Sanity check
    assert isinstance(best_model, (ProbRegressorModule, SFGEModule)), "Only probabilistic regressor models are allowed"

    return best_model

########################################################################################################################


def sample_scenarios(
                     # FIXME: 'num_scenarios' should be a single int
                     num_scenarios: List[int],
                     model: DFLModule,
                     input_features: torch.Tensor,
                     # FIXME: we can remove it and directly pass the desired input features as input
                     instance_idx: int):
    """
    Sample scenarios from a probabilistic model.
    :param num_scenarios: list of int; each scenario value we would like to test (e.g. [1, 5, 10, 50, 100]).
    :param model: dfl_module.DFLModule; probabilistic model used to sample scenarios.
    :param input_features: torch.Tensor; the input features for which we collect scenarios.
    :param instance_idx:
    :return:
    """

    # These are the instances sampled from the ML model and given as input to the SAA algorithm
    sampled_scenarios = list()

    for idx in range(max(num_scenarios)):

        # Force the first scenario to have demands equal to the mean value of the distribution...
        if idx == 0:
            _, sampled_param = \
                model.forward(input_features=input_features[instance_idx],
                              sample=False,
                              return_scaled=True)
        # ... and sample for the others
        else:
            _, sampled_param = \
                model.forward(input_features=input_features[instance_idx],
                              sample=True,
                              return_scaled=True)

        # Each scenario is obtained with a sample of the uncertain optimization model parameter

        # Remove eventual fake batch dimension
        sampled_param = torch.squeeze(sampled_param)
        sampled_scenarios.append(sampled_param)

    return sampled_scenarios

########################################################################################################################


def sample_average_approximation(scenarios: List[int],
                                 optimization_problem: OptimizationProblem,
                                 params: torch.Tensor,
                                 target_value: torch.Tensor) -> Dict:
    """
    Apply the Sample Average Approximation (SAA) algorithm to the list of scenarios given as input.
    :param scenarios: list of int; we apply SAA with different number of scenarios.
    :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the stochastic
                                 optimization problem to solve.
    :param params: torch:tensor; the parameters of the current optimization problem instance.
    :param target_value: torch.Tensor; the ground-truth target value.
    :return: dict; the updated version of the dictionary with SAA results.
    """

    # Cast list of torch.Tensor to torch.Tensor
    current_scenarios = torch.stack(scenarios)

    # This tensor is expected to have 2 dimensions: the first is the scenario and the second one is true dimensionality
    # of the parameters
    if len(current_scenarios.shape) == 1:
        current_scenarios = torch.unsqueeze(current_scenarios, dim=1)

    # Keep track of the SAA cost for each scenario value
    saa_sol, saa_runtime = \
        optimization_problem.solve_from_torch(current_scenarios,
                                              opt_prob_params=params,
                                              return_runtime=True)
    saa_sol = torch.as_tensor(saa_sol, dtype=torch.float32)
    saa_cost, _ = \
        optimization_problem.get_objective_values(y=target_value,
                                                  sols=saa_sol,
                                                  opt_prob_params=params)

    return saa_cost, saa_runtime

########################################################################################################################


def predict_then_optimize(experiment_path: str,
                          run_dictionary: Dict,
                          optimization_problem: OptimizationProblem,
                          num_scenarios: List[int]) -> Dict:
    """
    Ask the probabilistic model for samples that are given as input to the Sample Average Approximation (SAA) algorithm.
    :param experiment_path: str; where the model is loaded from.
    :param run_dictionary: dict; dictionary with the run configuration.
    :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem; the stochastic
                                 optimization problem to solve.
    :param num_scenarios: list of int; the list with the number of scenarios used to run the SAA algorithm.
    :return:
    """

    # Sanity check
    assert 'test_dl' in run_dictionary, 'Missing test_dl key in run_dictionary'
    assert 'model' in run_dictionary, 'Missing model key in run_dictionary'
    assert 'model_name' in run_dictionary, 'Missing model_name key in run_dictionary'
    assert 'run' in run_dictionary, 'Missing run key in run_dictionary'

    # PyTorch dataloader for test set
    test_dl = run_dictionary["test_dl"]
    # Instance of PyTorch DFL model; it is used as a backbone for loading the model to evaluate
    model = run_dictionary["model"]
    # Name of the method
    model_name = run_dictionary["model_name"]
    # For each training set, we run a different training routine; this is the run index
    run = run_dictionary["run"]
    # Random seeds for dataset split
    rnd_split_seed = run_dictionary["rnd_split_seed"]
    # Numpy seeds for dataset generation
    numpy_seed = run_dictionary["numpy_seed"]

    run_filepath = \
        os.path.join(experiment_path,
                     f'seed-{numpy_seed}',
                     f'rnd-split-seed-{rnd_split_seed}',
                     model_name,
                     f'run_{run}')

    best_model = \
        load_best_model(filepath=run_filepath,
                        model=model)

    # Get the input and target features
    x = test_dl.dataset.x
    y = test_dl.dataset.y
    opt_prob_params = test_dl.dataset.opt_prob_params

    # Keep track of the SAA cost, relative regret and runtime for each scenario value
    saa_res_num_scenarios = dict()

    # Initialize the datastructures to store results
    for num in num_scenarios:
        saa_res_num_scenarios[num] = dict()
        saa_res_num_scenarios[num][COST] = list()
        saa_res_num_scenarios[num][RELATIVE_REGRET] = list()
        saa_res_num_scenarios[num][RUNTIME] = list()

    n_test = len(test_dl)

    # Iterate over all the test instances and evaluate the predict-then-optimize + SAA approach
    for inst_idx in tqdm(range(n_test), desc='SAA evaluation'):

        # Sample a set of scenarios
        sampled_scenarios = \
            sample_scenarios(num_scenarios=num_scenarios,
                             model=best_model,
                             input_features=x,
                             instance_idx=inst_idx)

        # Keep track of the total optimal cost
        optimal_sol = optimization_problem.solve_from_torch(y[inst_idx], opt_prob_params=opt_prob_params[inst_idx])
        optimal_sol = torch.as_tensor(optimal_sol)

        optimal_cost, _ = \
            optimization_problem.get_objective_values(y=y[inst_idx],
                                                      sols=optimal_sol,
                                                      opt_prob_params=opt_prob_params[inst_idx])

        # Apply SAA with the user defined number of scenarios
        for current_num_scenarios in num_scenarios:

            samples = sampled_scenarios[:current_num_scenarios]

            # Solution cost and runtime for SAA
            saa_cost, saa_runtime = \
                sample_average_approximation(samples,
                                             optimization_problem,
                                             opt_prob_params[inst_idx],
                                             y[inst_idx])

            rel_regret = (saa_cost[TOTAL_COST] - optimal_cost[TOTAL_COST]) / optimal_cost[TOTAL_COST]
            saa_res_num_scenarios[current_num_scenarios][RELATIVE_REGRET].append(rel_regret)
            saa_res_num_scenarios[current_num_scenarios][COST].append(saa_cost[TOTAL_COST])
            saa_res_num_scenarios[current_num_scenarios][RUNTIME].append(saa_runtime)

    # Pretty visualization
    results_table = list()
    header = ["Num. scenarios", "Cost", "Relative regret", "Runtime"]

    for num in num_scenarios:
        saa_res_num_scenarios[num][RELATIVE_REGRET] = np.mean(saa_res_num_scenarios[num][RELATIVE_REGRET])
        saa_res_num_scenarios[num][COST] = np.mean(saa_res_num_scenarios[num][COST])
        saa_res_num_scenarios[num][RUNTIME] = np.mean(saa_res_num_scenarios[num][RUNTIME])

        row = [
            num,
            saa_res_num_scenarios[num][COST],
            saa_res_num_scenarios[num][RELATIVE_REGRET],
            saa_res_num_scenarios[num][RUNTIME]
        ]
        results_table.append(row)

    # Create the higher level header
    higher_level_header = [f'Dataset n.{numpy_seed} - Split n.{rnd_split_seed} - Run n.{run}']

    # Format the main header and results table
    table_str = tabulate([header] + results_table, headers="firstrow", tablefmt="fancy_grid")

    # Determine the width of the higher level header
    higher_level_header_width = len(higher_level_header[0])

    # Print the higher level header as the title
    print(higher_level_header[0])

    # Print separator
    print('-' * higher_level_header_width)

    # Print the main header and results table
    print(table_str)

    # Save run results on file
    with open(os.path.join(run_filepath, 'saa-results.pkl'), 'wb') as file:
        pickle.dump(saa_res_num_scenarios, file)

########################################################################################################################


def main():
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_dir', type=str, help='Configuration file loadpath')
    parser.add_argument('res_dir', type=str, help='Results savepath')
    parser.add_argument("--num-scenarios",
                        type=int,
                        nargs='+',
                        required=True,
                        help="Number of scenarios for the stochastic algo")

    # Parse the arguments
    args = parser.parse_args()
    config_filepath = args.config_dir
    results_filepath = args.res_dir
    num_scenarios = args.num_scenarios

    # Read configuration files
    with open(config_filepath) as json_file:
        config = json.load(json_file)

    # Build run dictionaries for parallelization
    run_dictionaries = list()

    # Get the dataset splits and generation seeds
    numpy_seeds = config['numpy_seed']
    rnd_split_seeds = config['rnd_split_seed']
    num_runs_per_seed = config['num_runs_per_seed']

    shared_config = config.copy()
    del shared_config['numpy_seed']
    del shared_config['rnd_split_seed']

    if isinstance(numpy_seeds, int):
        numpy_seeds = [numpy_seeds]

    if isinstance(rnd_split_seeds, int):
        rnd_split_seeds = [rnd_split_seeds]

    # Create a dictionary with the arguments for each run
    for numpy_sd in numpy_seeds:
        for rnd_split_sd in rnd_split_seeds:

            # print(f'Dataset n.{numpy_sd} | Split: {rnd_split_sd}\n')

            # Set PyTorch random seed
            manual_seed(config['torch_seed'])

            # Create train, validation and test PyTorch data loaders and the optimization problem instance
            train_dl, validation_dl, test_dl, optimization_problem = \
                build_data_loaders(config, rnd_split_seed=rnd_split_sd, dataset_seed=numpy_sd)

            for run in range(num_runs_per_seed):

                # Create the DFL models
                models = build_models(config=config, train_dl=train_dl, optimization_problem=optimization_problem)

                for model_name in models:
                    run_dictionary = {
                        "config": shared_config,
                        "train_dl": train_dl,
                        "validation_dl": validation_dl,
                        "test_dl": test_dl,
                        "model": models[model_name],
                        "model_name": model_name,
                        "run": run,
                        "numpy_seed": numpy_sd,
                        "rnd_split_seed": rnd_split_sd
                    }
                    run_dictionaries.append(run_dictionary)

    # For each configuration, run the probabilistic model + Sample Average Approximation algorithm
    for run_dictionary in run_dictionaries:

        predict_then_optimize(experiment_path=results_filepath,
                              run_dictionary=run_dictionary,
                              num_scenarios=num_scenarios,
                              optimization_problem=optimization_problem)

########################################################################################################################


if __name__ == '__main__':
    main()

