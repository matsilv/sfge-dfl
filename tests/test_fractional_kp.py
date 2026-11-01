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
from tqdm import tqdm

from optimization_problems.fractional_kp import FractionalKPProblem, PROBLEM_ID
from optimization_problems import TOTAL_COST, SUBOPTIMALITY_COST, PENALTY_COST
from utils.build_data_loaders import build_data_loaders_fractional_kp
from experiments.experiment_handler import main

from tests.base import BaseTest
from tests import set_seeds

########################################################################################################################


# There is no generation script for this dataset: the files must be copied in data/data/fractional_kp by hand
@unittest.skipUnless(os.path.isdir(os.path.join('data', 'data', PROBLEM_ID)),
                     'Fractional KP dataset not found in data/data/fractional_kp')
class TestFractionalKP(unittest.TestCase, BaseTest):
    def setUp(self) -> None:

        self._seed = 0
        set_seeds(self._seed)
        self._input_dim = 40960
        self._num_items = 10
        self._dataset_len = 1000
        self._datapath = os.path.join('data', 'data', 'fractional_kp')
        self._penalty_values = ['1', '2', '05', '025']
        self._rnd_item_values = np.random.uniform(low=0, high=1, size=self._num_items)
        self._rnd_item_weights = np.random.uniform(low=0, high=1, size=self._num_items)
        self._capacity = np.array([10])
        self._problem_id = PROBLEM_ID
        self._config_filepath = os.path.join('tests', self._problem_id, 'config.json')
        self._results_filepath = os.path.join('tests', self._problem_id, 'results')
        self._sanity_check_filepath = os.path.join('tests', self._problem_id, 'sanity-check')

    def test_01_check_datasets(self):
        print('Loading features...')
        features = pd.read_csv(os.path.join(self._datapath, 'features.csv'), index_col=0)
        print('Finished\n')

        self.assertEqual(features.shape, (self._dataset_len, self._input_dim))

        print('Loading values...')
        values = pd.read_csv(os.path.join(self._datapath, 'values.csv'), index_col=0)
        print('Finished\n')

        self.assertEqual(values.shape, (self._dataset_len, self._num_items))

        print('Loading weights...')
        weights = pd.read_csv(os.path.join(self._datapath, 'weights.csv'), index_col=0)
        print('Finished\n')

        self.assertEqual(weights.shape, (self._dataset_len, self._num_items))

        print('Loading penalties...')
        for p in self._penalty_values:
            penalty = pd.read_csv(os.path.join(self._datapath, f'penalty{p}.csv'), index_col=0)
            self.assertEqual(penalty.shape, (self._dataset_len, self._num_items))
        print('Finished\n')

    def test_02_check_opt_prob(self):

        print('Loading values...')
        values = pd.read_csv(os.path.join(self._datapath, 'values.csv'), index_col=0)
        values = values.values
        print('Finished\n')

        print('Loading weights...')
        weights = pd.read_csv(os.path.join(self._datapath, 'weights.csv'), index_col=0)
        weights = weights.values
        print('Finished\n')

        print('Loading penalties...')
        penalties = pd.read_csv(os.path.join(self._datapath, f'penalty{self._penalty_values[0]}.csv'), index_col=0)
        penalties = penalties.values
        print('Finished\n')

        problem = FractionalKPProblem(dim=self._num_items)

        for v, w, p in tqdm(zip(values, weights, penalties),
                            total=self._dataset_len,
                            desc='Solving dataset opt. problems'):

            true_targets = np.concatenate((v, w), axis=-1)
            preds = np.concatenate((self._rnd_item_values, self._rnd_item_weights), axis=-1)
            opt_prb_params = {'capacity': self._capacity, 'penalty': p}

            opt_sol, runtime = problem.solve(y=true_targets, opt_prob_params=opt_prb_params)
            pred_sol, pred_runtime = problem.solve(y=preds, opt_prob_params=opt_prb_params)

            self.assertTrue(isinstance(opt_sol, np.ndarray))
            self.assertTrue(isinstance(runtime, float))

            self.assertTrue(isinstance(pred_sol, np.ndarray))
            self.assertTrue(isinstance(pred_runtime, float))

            opt_cost, feasible = \
                problem.get_objective_values(y=torch.as_tensor(true_targets),
                                             sols=torch.as_tensor(opt_sol),
                                             opt_prob_params=opt_prb_params)

            pred_cost, feasible = \
                problem.get_objective_values(y=torch.as_tensor(true_targets),
                                             sols=torch.as_tensor(pred_sol),
                                             opt_prob_params=opt_prb_params)

            self.assertIsInstance(feasible, bool)
            self.assertTrue(TOTAL_COST in opt_cost.keys())
            self.assertTrue(SUBOPTIMALITY_COST in opt_cost.keys())
            self.assertTrue(PENALTY_COST in opt_cost.keys())

            self.assertTrue(TOTAL_COST in pred_cost.keys())
            self.assertTrue(SUBOPTIMALITY_COST in pred_cost.keys())
            self.assertTrue(PENALTY_COST in pred_cost.keys())

            self.assertGreater(sum(opt_sol), 0)
            self.assertGreater(sum(pred_sol), 0)
            self.assertGreaterEqual(opt_cost[TOTAL_COST], 0)
            self.assertEqual(opt_cost[PENALTY_COST], 0)
            self.assertLessEqual(pred_cost[TOTAL_COST], opt_cost[TOTAL_COST])

    def test_03_check_data_loader(self):
        with open(self._config_filepath, 'r') as file:
            config = json.load(file)

        val_split = config['prop_validation']
        train_split = config['prop_training']

        features_filename = os.path.join(self._datapath, 'features.csv')
        values_filename = os.path.join(self._datapath, 'values.csv')
        weights_filename = os.path.join(self._datapath, 'weights.csv')
        solutions_filename = os.path.join(self._datapath, f'solutions-{config["capacity"]}.csv')

        # Check that the configuration file is a dictionary
        self.assertIsInstance(config, dict)

        train_dl, val_dl, test_dl, fractional_kp_problem = \
            build_data_loaders_fractional_kp(config=config,
                                             rnd_split_seed=self._seed,
                                             dataset_seed=self._seed)

        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(val_dl, DataLoader)
        self.assertIsInstance(test_dl, DataLoader)
        self.assertIsInstance(fractional_kp_problem, FractionalKPProblem)

        saved_features = pd.read_csv(features_filename, index_col=0)
        saved_features = saved_features.values
        loaded_features = [train_dl.dataset.x.numpy(), test_dl.dataset.x.numpy(), val_dl.dataset.x.numpy()]
        self._compare_saved_and_loaded_vals(saved_vals=saved_features,
                                            loaded_vals=loaded_features,
                                            val_split=val_split,
                                            train_split=train_split)

        saved_values = pd.read_csv(values_filename, index_col=0)
        saved_values = saved_values.values
        saved_weights = pd.read_csv(weights_filename, index_col=0)
        saved_weights = saved_weights.values
        saved_targets = np.concatenate((saved_values, saved_weights), axis=1)
        loaded_targets = [train_dl.dataset.y.numpy(), test_dl.dataset.y.numpy(), val_dl.dataset.y.numpy()]
        self._compare_saved_and_loaded_vals(saved_vals=saved_targets,
                                            loaded_vals=loaded_targets,
                                            val_split=val_split,
                                            train_split=train_split)

        saved_solutions = pd.read_csv(solutions_filename, index_col=0)
        saved_solutions = saved_solutions.values
        loaded_solutions = [train_dl.dataset.solutions.numpy(),
                            test_dl.dataset.solutions.numpy(),
                            val_dl.dataset.solutions.numpy()]
        self._compare_saved_and_loaded_vals(saved_vals=saved_solutions,
                                            loaded_vals=loaded_solutions,
                                            val_split=val_split,
                                            train_split=train_split)

    def test_04_methods(self):

        # Start from a clean folder: re-training into an existing one leaves extra checkpoints in best-model/
        shutil.rmtree(self._results_filepath, ignore_errors=True)
        os.makedirs(self._results_filepath)

        sys.argv[1:] = [self._config_filepath, self._results_filepath, "--mode", "train"]
        main()

    def test_06_results_check(self):
        super(TestFractionalKP, self).test_06_results_check()


########################################################################################################################


if __name__ == '__main__':
    unittest.main()
