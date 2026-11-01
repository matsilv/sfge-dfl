import unittest
import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
import json
import torch
from torch.utils.data.dataloader import DataLoader
import sys
import shutil

from data.generation_scripts.generate_quadratic_kp_data import generate_instances
from utils.build_data_loaders import build_data_loaders_quadratic_kp
from optimization_problems.quadratic_kp_problem import PROBLEM_ID as QUADRATIC_KP
from optimization_problems.quadratic_kp_problem import QuadraticKP
from optimization_problems import TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST
from experiments.experiment_handler import main

from tests.base import BaseTest
from tests import set_seeds

########################################################################################################################


class TestQuadraticKP01(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        self._input_dim = 5
        self._output_dim = 64
        self._num_items = int(np.sqrt(self._output_dim))
        self._relative_capacity = 0.5
        self._deg = 5
        self._n = 1000
        self._multiplicative_noise = 0.1
        self._additive_noise = 0.03
        self._correlate_values_and_weights = 1
        self._rho = 0
        self._seed = 0

        set_seeds(self._seed)

        self._weights = np.random.uniform(0, 1, self._num_items)
        self._random_costs = np.random.uniform(0, 1, self._output_dim)
        self._capacity = sum(self._weights) * self._relative_capacity

        self._datapath = os.path.join('data', 'data')

        self._filepath_suffix = f'_n_{self._n}_input_dim_{self._input_dim}_output_dim_{self._output_dim}'
        self._filepath_suffix += f'_mult_noise_{self._multiplicative_noise}_add_noise_{self._additive_noise}'
        self._filepath_suffix += f'_deg_{self._deg}_relative_capacity_{self._relative_capacity}'
        self._filepath_suffix += f'_correlation_type_{self._correlate_values_and_weights}_rho_{self._rho}.csv'
        prefix = os.path.join(self._datapath, QUADRATIC_KP, f'seed-{self._seed}')
        self._weights_filename = os.path.join(prefix, 'weights' + self._filepath_suffix)
        self._targets_filename = os.path.join(prefix, 'targets' + self._filepath_suffix)
        self._features_filename = os.path.join(prefix, 'features' + self._filepath_suffix)
        self._solutions_filename = os.path.join(prefix, 'solutions' + self._filepath_suffix)

        self._problem_id = 'quadratic_kp'
        self._config_filepath = os.path.join('tests', self._problem_id, 'config.json')
        self._results_filepath = os.path.join('tests', self._problem_id, 'results')
        self._sanity_check_filepath = os.path.join('tests', self._problem_id, 'sanity-check')

    def test_01_generate_instances(self):
        generate_instances(datapath=self._datapath,
                           num_instances=self._n,
                           input_dim=self._input_dim,
                           output_dim=self._output_dim,
                           deg=self._deg,
                           multiplicative_noise=self._multiplicative_noise,
                           additive_noise=self._additive_noise,
                           weights=self._weights,
                           relative_capacity=self._relative_capacity,
                           correlate_values_and_weights=self._correlate_values_and_weights,
                           rho=self._rho)

        # Check that the files have been created
        self.assertTrue(os.path.isfile(self._weights_filename))
        self.assertTrue(os.path.isfile(self._targets_filename))
        self.assertTrue(os.path.isfile(self._features_filename))
        self.assertTrue(os.path.isfile(self._solutions_filename))

    def test_02_check_instances(self):
        weights_df = pd.read_csv(self._weights_filename, index_col=0)
        targets_df = pd.read_csv(self._targets_filename, index_col=0)
        features_df = pd.read_csv(self._features_filename, index_col=0)
        solutions_df = pd.read_csv(self._solutions_filename, index_col=0)

        # Check that dataframes do not contain null values
        self.assertFalse(weights_df.isnull().values.any())
        self.assertFalse(targets_df.isnull().values.any())
        self.assertFalse(features_df.isnull().values.any())
        self.assertFalse(solutions_df.isnull().values.any())

        # Check whether the data dimension is the expected one
        self.assertTrue(weights_df.shape == (self._num_items, 1))
        self.assertTrue(targets_df.shape == (self._n, self._output_dim))
        self.assertTrue(features_df.shape == (self._n, self._input_dim))
        self.assertTrue(solutions_df.shape == (self._n, self._num_items))

    def test_03_check_opt_prob(self):
        problem = QuadraticKP(dim=self._num_items)

        opt_prob_params = {'weights': torch.as_tensor(self._weights), 'capacity': torch.as_tensor(self._capacity)}

        sol, runtime = problem.solve(y=self._random_costs,
                                     opt_prob_params=opt_prob_params)

        self.assertTrue(isinstance(sol, np.ndarray))
        self.assertTrue(isinstance(runtime, float))

        cost, feasible = \
            problem.get_objective_values(y=torch.as_tensor(self._random_costs),
                                         sols=torch.as_tensor(sol),
                                         opt_prob_params=opt_prob_params)

        self.assertTrue(TOTAL_COST in cost.keys())
        self.assertTrue(SUBOPTIMALITY_COST in cost.keys())
        self.assertTrue(PENALTY_COST in cost.keys())

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

        train_dl, val_dl, test_dl, quadratic_kp_problem = \
            build_data_loaders_quadratic_kp(config=config,
                                            rnd_split_seed=self._seed,
                                            dataset_seed=self._seed)

        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(val_dl, DataLoader)
        self.assertIsInstance(test_dl, DataLoader)
        self.assertIsInstance(quadratic_kp_problem, QuadraticKP)

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

    # FIXME:
    # def test_06_results_check(self):
    #     super(TestQuadraticKP01, self).test_06_results_check()


########################################################################################################################


if __name__ == '__main__':
    unittest.main()
