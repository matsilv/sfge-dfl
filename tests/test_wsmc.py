import unittest
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
import json
import pandas as pd
from pandas.testing import assert_frame_equal
import sys
import shutil

from data.generation_scripts.generate_wsmc_data import generate_dataset
from optimization_problems.set_cover_problem import StochasticWeightedSetMultiCover, PROBLEM_ID
from optimization_problems import TOTAL_COST, PENALTY_COST, SUBOPTIMALITY_COST
from utils.build_data_loaders import build_data_loaders_wsmc
from experiments.experiment_handler import main

from tests.base import BaseTest
from tests import set_seeds

########################################################################################################################


class TestWSMC(unittest.TestCase, BaseTest):
    def setUp(self) -> None:
        self._problem_id = PROBLEM_ID
        self._num_instances = 1000
        self._num_sets = 25
        self._num_prods = 5
        self._input_dim = 5
        self._deg = 5
        self._mult_noise = 0.1
        self._add_noise = 0.03
        self._datapath = os.path.join('data', 'data')
        self._density = 0.02
        self._min_lmbd = 1
        self._max_lmbd = 10
        self._penalty_factor = 10.0
        self._seed = 0
        set_seeds(self._seed)
        self._random_demands = np.random.randint(self._min_lmbd, self._max_lmbd, size=self._num_prods)
        self._random_scenarios = np.random.randint(self._min_lmbd, self._max_lmbd, size=(5, self._num_prods))

        self._saved_files = {'features.csv',
                             'targets.csv',
                             'availability.npy',
                             'set_costs.npy',
                             'prod_costs.npy',
                             'wsmc_dim.json',
                             'solutions.csv'}

        self._savepath = os.path.join(self._datapath,
                                      self._problem_id,
                                      f'{self._num_prods}x{self._num_sets}',
                                      f'penalty-{self._penalty_factor}',
                                      f'seed-{self._seed}')

        self._features_filename = os.path.join(self._savepath, 'features.csv')
        self._targets_filename = os.path.join(self._savepath, 'targets.csv')
        self._solutions_filename = os.path.join(self._savepath, 'solutions.csv')

        self._config_filepath = os.path.join('tests', self._problem_id, 'config.json')
        self._results_filepath = os.path.join('tests', self._problem_id, 'results')
        self._sanity_check_filepath = os.path.join('tests', self._problem_id, 'sanity-check')

    def test_01_generate_data(self):
        generate_dataset(data_path=self._datapath,
                         num_instances=self._num_instances,
                         num_sets=self._num_sets,
                         num_prods=self._num_prods,
                         density=self._density,
                         input_dim=self._input_dim,
                         deg=self._deg,
                         multiplicative_noise=self._mult_noise,
                         additive_noise=self._add_noise,
                         penalty_factor=self._penalty_factor,
                         seed=self._seed)

        files = os.listdir(self._savepath)
        files = set(files)

        # Check that the data folder contains the required files
        self.assertEqual(files, self._saved_files)

        # Load and check shape of the optimization model parameters
        availability = np.load(os.path.join(self._savepath, 'availability.npy'))
        self.assertEqual(availability.shape, (self._num_prods, self._num_sets))

        prod_costs = np.load(os.path.join(self._savepath, 'prod_costs.npy'))
        self.assertEqual(prod_costs.shape, (self._num_prods, ))

        set_costs = np.load(os.path.join(self._savepath, 'set_costs.npy'))
        self.assertEqual(set_costs.shape, (self._num_sets, ))

        with open(os.path.join(self._savepath, 'wsmc_dim.json'), 'r') as file:
            wsmc_dim = json.load(file)

        self.assertEqual(wsmc_dim.keys(), {'num_sets', 'num_prods'})
        self.assertEqual(wsmc_dim['num_sets'], self._num_sets)
        self.assertEqual(wsmc_dim['num_prods'], self._num_prods)

        # Load and check shape of the features and targets
        features = pd.read_csv(os.path.join(self._savepath, 'features.csv'), index_col=0)
        self.assertEqual(features.shape, (self._num_instances, self._input_dim))

        targets = pd.read_csv(os.path.join(self._savepath, 'targets.csv'), index_col=0)
        self.assertEqual(targets.shape, (self._num_instances, self._num_prods))

        solutions = pd.read_csv(os.path.join(self._savepath, 'solutions.csv'), index_col=0)
        self.assertEqual(solutions.shape, (self._num_instances, self._num_sets))

    def test_02_check_opt_prob(self):

        # Load optimization problem parameters
        availability = np.load(os.path.join(self._savepath, 'availability.npy'))
        prod_costs = np.load(os.path.join(self._savepath, 'prod_costs.npy'))
        set_costs = np.load(os.path.join(self._savepath, 'set_costs.npy'))
        targets = pd.read_csv(os.path.join(self._savepath, 'targets.csv'), index_col=0)

        availability = availability.reshape(-1)
        opt_prob_params = {'set_costs': torch.as_tensor(set_costs),
                           'prod_costs': torch.as_tensor(prod_costs),
                           'availability': torch.as_tensor(availability)}

        with open(os.path.join(self._savepath, 'wsmc_dim.json'), 'r') as file:
            wsmc_dim = json.load(file)

        problem = \
            StochasticWeightedSetMultiCover(num_sets=wsmc_dim['num_sets'],
                                            num_products=wsmc_dim['num_prods'])

        # Solve each instance and check that the optimal solution is better than a random one
        for demands in tqdm(targets.values, total=self._num_instances, desc='Solving instances'):
            rnd_sol, runtime = problem.solve(y=self._random_demands,
                                             opt_prob_params=opt_prob_params)
            saa_rnd_sol, runtime = problem.solve(y=self._random_scenarios,
                                                 opt_prob_params=opt_prob_params)

            rnd_sol_cost, feasible = \
                problem.get_objective_values(y=torch.as_tensor(demands),
                                             sols=torch.as_tensor(rnd_sol),
                                             opt_prob_params=opt_prob_params)
            rnd_saa_sol_cost, feasible = \
                problem.get_objective_values(y=torch.as_tensor(demands),
                                             sols=torch.as_tensor(saa_rnd_sol),
                                             opt_prob_params=opt_prob_params)

            self.assertIsInstance(feasible, bool)
            self.assertTrue(TOTAL_COST in rnd_sol_cost.keys())
            self.assertTrue(SUBOPTIMALITY_COST in rnd_sol_cost.keys())
            self.assertTrue(PENALTY_COST in rnd_sol_cost.keys())

            rnd_sol_cost = rnd_sol_cost[TOTAL_COST]
            rnd_saa_sol_cost = rnd_saa_sol_cost[TOTAL_COST]

            true_sol, runtime = problem.solve(y=demands,
                                              opt_prob_params=opt_prob_params)
            true_sol_cost, feasible = \
                problem.get_objective_values(y=torch.as_tensor(demands),
                                             sols=torch.as_tensor(true_sol),
                                             opt_prob_params=opt_prob_params)

            self.assertIsInstance(feasible, bool)
            self.assertTrue(TOTAL_COST in true_sol_cost.keys())
            self.assertTrue(SUBOPTIMALITY_COST in true_sol_cost.keys())
            self.assertTrue(PENALTY_COST in true_sol_cost.keys())

            true_sol_cost = true_sol_cost[TOTAL_COST]

            # The true solution cost is optimal
            self.assertLessEqual(true_sol_cost, rnd_sol_cost)
            self.assertLessEqual(true_sol_cost, rnd_saa_sol_cost)

    def test_03_check_data_loader(self):
        with open(self._config_filepath, 'r') as file:
            config = json.load(file)

        val_split = config['prop_validation']
        train_split = config['prop_training']

        # Check that the configuration file is a dictionary
        self.assertIsInstance(config, dict)

        train_dl, val_dl, test_dl, problem = \
            build_data_loaders_wsmc(config=config,
                                    rnd_split_seed=self._seed,
                                    dataset_seed=self._seed)

        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(val_dl, DataLoader)
        self.assertIsInstance(test_dl, DataLoader)
        self.assertIsInstance(problem, StochasticWeightedSetMultiCover)

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

    def test_04_methods(self):

        # Start from a clean folder: re-training into an existing one leaves extra checkpoints in best-model/
        shutil.rmtree(self._results_filepath, ignore_errors=True)
        os.makedirs(self._results_filepath)

        sys.argv[1:] = [self._config_filepath, self._results_filepath, "--mode", "train"]
        main()

    def test_06_results_check(self):
        super(TestWSMC, self).test_06_results_check()

########################################################################################################################


if __name__ == '__main__':
    unittest.main()
