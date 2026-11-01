import os
import json
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

from typing import List

from . import dataset_split

########################################################################################################################


class BaseTest:

    def _compare_saved_and_loaded_vals(self,
                                       saved_vals: np.ndarray,
                                       loaded_vals: List[np.ndarray],
                                       val_split: float,
                                       train_split: float):
        """
        Make sure that values saved in the CSV and loaded using the data loading utilities are the same.
        :param saved_vals: numpy.ndarray; values loaded directly from the CSV file.
        :param loaded_vals: numpy.ndarray; values loaded with the data loading utilities.
        :param val_split: float; the fraction of the training set used from validation.
        :param train_split: float; the fraction of the dataset used for training.
        :return:
        """

        assert isinstance(saved_vals, np.ndarray)
        assert isinstance(loaded_vals, list)

        saved_vals = saved_vals.astype(np.float32)

        train_saved_vals, val_saved_vals, test_saved_vals = \
            dataset_split(arrays=[saved_vals],
                          train_split=train_split,
                          val_split=val_split,
                          seed=self._seed)

        train_saved_vals = train_saved_vals[0]
        test_saved_vals = test_saved_vals[0]
        val_saved_vals = val_saved_vals[0]

        self.assertEqual(len(loaded_vals[0]), len(train_saved_vals))
        self.assertEqual(len(loaded_vals[1]), len(test_saved_vals))
        self.assertEqual(len(loaded_vals[2]), len(val_saved_vals))

        self.assertTrue(np.array_equal(train_saved_vals, loaded_vals[0]))
        self.assertTrue(np.array_equal(test_saved_vals, loaded_vals[1]))
        self.assertTrue(np.array_equal(val_saved_vals, loaded_vals[2]))

    def test_06_results_check(self):
        # The sanity-check folder holds reference results of a previous training run (config.json + metrics.csv
        # files). Create it by copying `results` to `sanity-check` after a run you trust.
        if not os.path.isdir(self._sanity_check_filepath):
            self.skipTest(f'No reference results found in {self._sanity_check_filepath}')

        with open(self._config_filepath, 'r') as file:
            config = json.load(file)

        with open(os.path.join(self._sanity_check_filepath, 'config.json'), 'r') as file:
            sanity_check_config = json.load(file)

        self.assertEqual(config, sanity_check_config)

        numpy_seeds = config['numpy_seed']
        rnd_split_seeds = config['rnd_split_seed']
        runs = np.arange(config['num_runs_per_seed'])
        methods_configs = config['method_configurations']

        for npy_sd in numpy_seeds:
            for rnd_split_sd in rnd_split_seeds:
                for method_cfg in methods_configs:
                    for run in runs:
                        name = method_cfg['name']
                        filepath_suffix = \
                            os.path.join(f'seed-{npy_sd}',
                                         f'rnd-split-seed-{rnd_split_sd}',
                                         name,
                                         f'run_{run}',
                                         'metrics.csv')

                        metrics = pd.read_csv(os.path.join(self._results_filepath, filepath_suffix))
                        sanity_check_metrics = pd.read_csv(os.path.join(self._sanity_check_filepath, filepath_suffix))

                        del metrics['val_runtime']
                        del sanity_check_metrics['val_runtime']
                        del metrics['test_runtime']
                        del sanity_check_metrics['test_runtime']

                        assert_frame_equal(metrics, sanity_check_metrics)
