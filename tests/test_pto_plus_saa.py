import unittest
import os
import sys
import shutil

from experiments.experiment_handler import main as train
from experiments.pfl_plus_saa import main as pto_plus_saa

########################################################################################################################


class TestPTOPlusSAA(unittest.TestCase):
    """
    Train a probabilistic model (MLE) on each stochastic problem and run PFL + Sample Average Approximation on top of
    it. Data must have been generated first (test_01 of test_wsmc, test_stochastic_weights_kp and
    test_stochastic_capacity_kp). Results are saved in a separate `results-saa` folder so that they do not clash with
    the ones of the other tests.
    """

    def setUp(self) -> None:
        self._num_scenarios = ['1', '5', '10']

    def _train_and_run_saa(self, problem_id: str):
        config_filepath = os.path.join('tests', problem_id, 'saa_config.json')
        res_dir = os.path.join('tests', problem_id, 'results-saa')

        # Train from scratch: SAA expects a single checkpoint in each best-model folder
        shutil.rmtree(res_dir, ignore_errors=True)
        os.makedirs(res_dir)

        sys.argv[1:] = [config_filepath, res_dir, '--mode', 'train']
        train()

        sys.argv[1:] = [config_filepath, res_dir, '--num-scenarios', *self._num_scenarios]
        pto_plus_saa()

    def test_01_wsmc(self):
        self._train_and_run_saa('wsmc')

    def test_02_stochastic_weights_kp(self):
        self._train_and_run_saa('stochastic_weights_kp')

    def test_03_stochastic_capacity_kp(self):
        self._train_and_run_saa('stochastic_capacity_kp')

########################################################################################################################


if __name__ == '__main__':
    unittest.main()
