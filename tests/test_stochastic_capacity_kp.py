import unittest
import numpy as np
import os
import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from torch.utils.data import DataLoader
import json
import sys
import shutil

from experiments.experiment_handler import train_model, main
from optimization_problems import TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST
from optimization_problems.optimization_problem import OptimizationProblem
from optimization_problems.stochastic_capacity_kp import PROBLEM_ID, StochasticCapacityKnapsackProblem
from data.generation_scripts.generate_stochastic_capacity_kp_data import generate_instances
from utils.build_data_loaders import build_data_loaders_stochastic_capacity_kp

from tests.base import BaseTest
from tests import set_seeds

########################################################################################################################


class TestStochasticCapacityKnapsack(unittest.TestCase, BaseTest):
    def setUp(self) -> None:

        self._input_dim = 5
        self._output_dim = 1
        self._num_items = 50
        self._deg = 5
        self._n = 1000
        self._multiplicative_noise = 0.1
        self._correlate_values_and_weights = 1
        self._rho = 0
        self._seed = 0
        self._penalty = 10

        set_seeds(seed=self._seed)

        self._values = np.random.uniform(0, 1, self._num_items)
        self._weights = np.random.uniform(0, 1, self._num_items)
        self._random_capacity = np.array([int(sum(self._weights) * 0.2)])

        self._problem_id = PROBLEM_ID

        self._datapath = os.path.join('data', 'data', self._problem_id)

        self._filepath_suffix = f'_n_{self._n}_input_dim_{self._input_dim}_output_dim_{self._num_items}'
        self._filepath_suffix += f'_mult_noise_{self._multiplicative_noise}_deg_{self._deg}'
        self._filepath_suffix += f'_correlation_type_{self._correlate_values_and_weights}_rho_{self._rho}'

        self._values_filename = \
            os.path.join(self._datapath, f'seed-{self._seed}', 'values' + self._filepath_suffix + '.npy')
        self._weights_filename = \
            os.path.join(self._datapath, f'seed-{self._seed}', 'weights' + self._filepath_suffix + '.npy')
        self._features_filename = \
            os.path.join(self._datapath, f'seed-{self._seed}', 'features' + self._filepath_suffix + '.csv')
        self._targets_filename = \
            os.path.join(self._datapath, f'seed-{self._seed}', 'targets' + self._filepath_suffix + '.csv')
        self._solutions_filename = \
            os.path.join(self._datapath, f'seed-{self._seed}', 'solutions' + self._filepath_suffix + '.csv')

        self._config_filepath = os.path.join('tests', self._problem_id, 'config.json')
        self._results_filepath = os.path.join('tests', self._problem_id, 'results')
        self._sanity_check_filepath = os.path.join('tests', self._problem_id, 'sanity-check')

    def test_01_generate_instances(self):
        generate_instances(datapath=self._datapath,
                           num_instances=self._n,
                           input_dim=self._input_dim,
                           output_dim=self._num_items,
                           deg=self._deg,
                           multiplicative_noise=self._multiplicative_noise,
                           values=self._values,
                           weights=self._weights,
                           correlate_values_and_weights=self._correlate_values_and_weights,
                           rho=self._rho,
                           penalty=self._penalty)

        # Check that the files have been created
        self.assertTrue(os.path.isfile(self._values_filename))
        self.assertTrue(os.path.isfile(self._targets_filename))
        self.assertTrue(os.path.isfile(self._features_filename))
        self.assertTrue(os.path.isfile(self._weights_filename))
        self.assertTrue(os.path.isfile(self._solutions_filename))

    def test_02_check_instances(self):
        values = np.load(self._values_filename)
        weights = np.load(self._weights_filename)
        targets_df = pd.read_csv(self._targets_filename, index_col=0)
        features_df = pd.read_csv(self._features_filename, index_col=0)
        solutions_df = pd.read_csv(self._solutions_filename, index_col=0)

        x = features_df.values.astype(np.float32)
        y = targets_df.values.astype(np.float32)
        solutions = solutions_df.values.astype(np.float32)

        # Check whether the data dimension is the expected one
        self.assertTrue(values.shape == (self._num_items, ))
        self.assertTrue(weights.shape == (self._num_items, ))
        self.assertTrue(y.shape == (self._n, self._output_dim))
        self.assertTrue(x.shape == (self._n, self._input_dim))
        self.assertTrue(solutions.shape == (self._n, self._num_items))

    def test_03_check_opt_prob(self):
        problem = StochasticCapacityKnapsackProblem(dim=self._num_items, penalty=self._penalty)

        opt_prob_params = {'values': torch.tensor(self._values),
                           'weights': torch.tensor(self._weights)}

        sol, runtime = problem.solve(y=self._random_capacity, opt_prob_params=opt_prob_params)

        self.assertTrue(isinstance(sol, np.ndarray))
        self.assertTrue(isinstance(runtime, float))

        cost, feasible = \
            problem.get_objective_values(y=torch.as_tensor(self._random_capacity),
                                         sols=torch.as_tensor(sol),
                                         opt_prob_params=opt_prob_params)

        self.assertTrue(TOTAL_COST in cost.keys())
        self.assertTrue(SUBOPTIMALITY_COST in cost.keys())
        self.assertTrue(PENALTY_COST in cost.keys())
        self.assertIsInstance(feasible, bool)

        cost = cost[TOTAL_COST]

        self.assertGreater(sum(sol), 0)
        self.assertGreater(cost, 0)

    def test_04_check_data_loader(self):
        with open(self._config_filepath, 'r') as file:
            config = json.load(file)

        val_split = config['prop_validation']
        train_split = config['prop_training']

        # Check that the configuration file is a dictionary
        self.assertIsInstance(config, dict)

        train_dl, val_dl, test_dl, optimization_problem = \
            build_data_loaders_stochastic_capacity_kp(config=config, dataset_seed=0, rnd_split_seed=0)

        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(val_dl, DataLoader)
        self.assertIsInstance(test_dl, DataLoader)
        self.assertIsInstance(optimization_problem, OptimizationProblem)

        saved_features = pd.read_csv(self._features_filename, index_col=0)
        saved_features = saved_features.values
        loaded_features = [train_dl.dataset.x.numpy(), test_dl.dataset.x.numpy(), val_dl.dataset.x.numpy()]
        self._compare_saved_and_loaded_vals(saved_vals=saved_features,
                                            loaded_vals=loaded_features,
                                            val_split=val_split,
                                            train_split=train_split)

        saved_targets = pd.read_csv(self._targets_filename, index_col=0)
        saved_targets = saved_targets.values
        loaded_targets = [train_dl.dataset.y.numpy(), test_dl.dataset.y.numpy(), val_dl.dataset.y.numpy()]
        self._compare_saved_and_loaded_vals(saved_vals=saved_targets,
                                            loaded_vals=loaded_targets,
                                            val_split=val_split,
                                            train_split=train_split)

        saved_solutions = pd.read_csv(self._solutions_filename, index_col=0)
        saved_solutions = saved_solutions.values
        loaded_solutions = [train_dl.dataset.solutions.numpy(),
                            test_dl.dataset.solutions.numpy(),
                            val_dl.dataset.solutions.numpy()]
        self._compare_saved_and_loaded_vals(saved_vals=saved_solutions,
                                            loaded_vals=loaded_solutions,
                                            val_split=val_split,
                                            train_split=train_split)

    def test_05_methods(self):

        # Start from a clean folder: re-training into an existing one leaves extra checkpoints in best-model/
        shutil.rmtree(self._results_filepath, ignore_errors=True)
        os.makedirs(self._results_filepath)

        sys.argv[1:] = [self._config_filepath, self._results_filepath, "--mode", "train"]
        main()

    def test_06_results_check(self):
        super(TestStochasticCapacityKnapsack, self).test_06_results_check()

########################################################################################################################


if __name__ == '__main__':
    unittest.main()

