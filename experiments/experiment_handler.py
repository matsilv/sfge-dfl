"""
    Methods used for training and test the methods.
"""

import os
import sys
import json
import shutil
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from torch.random import manual_seed
import argparse
import pickle

from utils.build_data_loaders import build_data_loaders
from utils.build_models import build_models

from typing import Dict, Tuple, List

########################################################################################################################


# FIXME: explicit arguments rather than implicit
def train_model(run_dictionary, experiment_path, wandb_configs, wandb_user):

    # Sanity check
    #assert len(args) == 2, "Two arguments are expected"

    # run_dictionary, experiment_path = args[0], args[1]

    assert isinstance(run_dictionary, Dict), "First argument must be a dictionary with the configuration infos"
    assert isinstance(experiment_path, str), "Second argument must be a string representing the experiment filepath"

    config = run_dictionary["config"]
    train_dl = run_dictionary["train_dl"]
    validation_dl = run_dictionary["validation_dl"]
    test_dl = run_dictionary["test_dl"]
    model = run_dictionary["model"]
    model_name = run_dictionary["model_name"]
    run = run_dictionary["run"]
    rnd_split_seed = run_dictionary["rnd_split_seed"]
    numpy_seed = run_dictionary["numpy_seed"]
    patience = config["patience"]
    max_epochs = config["epochs"]

    logging_interval = len(train_dl)

    # Log results in a CSV
    train_res_filepath = \
        os.path.join(experiment_path,
                     f'seed-{numpy_seed}',
                     f'rnd-split-seed-{rnd_split_seed}')

    model.exp_filepath = os.path.join(train_res_filepath, model_name, f'run_{run}')

    csv_logger = CSVLogger(train_res_filepath,
                           name=model_name,
                           version=f"run_{run}")

    loggers = [csv_logger]

    if wandb_user is not None:
        wandb_logger = WandbLogger(entity=wandb_user,
                                   project='dfl',
                                   job_type='train',
                                   config=wandb_configs,
                                   reinit=True)

        loggers.append(wandb_logger)

    # Create an early stopping callbacks
    early_stopping_callback = \
        EarlyStopping(monitor=model.monitor,
                      min_delta=model.min_delta,
                      mode='min',
                      patience=patience)

    # Checkpoint the best model
    best_model_filepath = os.path.join(train_res_filepath, model_name, f'run_{run}', 'best-model')

    checkpoint_callback = \
        ModelCheckpoint(dirpath=best_model_filepath,
                        save_top_k=1,
                        monitor=model.monitor,
                        mode='min')

    # Create the PyTorch Lighting Trainer
    trainer = pl.Trainer(min_epochs=10,
                         max_epochs=max_epochs,
                         log_every_n_steps=logging_interval,
                         val_check_interval=logging_interval,
                         logger=loggers,
                         callbacks=[early_stopping_callback, checkpoint_callback])

    # Mock validation before training
    trainer.validate(model=model, dataloaders=validation_dl)

    # Fit the model
    trainer.fit(model, train_dl, validation_dl)

    # Get the best model and evaluate it on test set
    best_model = \
        type(model).load_from_checkpoint(checkpoint_callback.best_model_path,
                                   net=model.net,
                                   optimization_problem=model.optimization_problem,
                                   annealer=None)

    trainer.test(model=best_model, dataloaders=test_dl)

    if wandb_user is not None:
        wandb.finish()

########################################################################################################################


def evaluate_model(run_dictionary: Dict,
                   experiment_path: str) -> Tuple[dict, dict]:
    """
    Evaluate a trained model on validation and test sets.
    :param run_dictionary: dict; a dictionary with some information useful to run the evaluation.
    :param experiment_path: str; where results of an experiment single run are loaded from.
    :return: 2 dict; validation and test results.
    """

    # Sanity check
    assert 'validation_dl' in run_dictionary, 'Missing validation_dl key in run_dictionary'
    assert 'test_dl' in run_dictionary, 'Missing test_dl key in run_dictionary'
    assert 'model' in run_dictionary, 'Missing model key in run_dictionary'
    assert 'model_name' in run_dictionary, 'Missing model_name key in run_dictionary'
    assert 'run' in run_dictionary, 'Missing run key in run_dictionary'

    # PyTorch dataloader for validation
    validation_dl = run_dictionary["validation_dl"]
    # PyTorch dataloader for test
    test_dl = run_dictionary["test_dl"]
    # Instance of PyTorch DFL model; it is used only for loading the model to evaluate
    model = run_dictionary["model"]
    # Name of the method
    model_name = run_dictionary["model_name"]
    # For each training set, we run a different training routine; this is the run index
    run = run_dictionary["run"]

    numpy_seed = run_dictionary["numpy_seed"]
    rnd_split_seed = run_dictionary["rnd_split_seed"]

    # Create the PyTorch Lighting Trainer
    trainer = pl.Trainer(enable_checkpointing=False, logger=False)

    # Get the best model and evaluate it on test set
    # Checkpoint the best model
    best_model_filepath = \
        os.path.join(experiment_path,
                     f'seed-{numpy_seed}',
                     f'rnd-split-seed-{rnd_split_seed}',
                     model_name,
                     f'run_{run}',
                     'best-model')
    best_model_filename = os.listdir(best_model_filepath)

    # Sanity check
    # assert len(best_model_filename) == 1, "A single file is expected in best-model folder"
    best_model_filename = best_model_filename[0]

    best_model = \
        type(model).load_from_checkpoint(os.path.join(best_model_filepath, best_model_filename),
                                   net=model.net,
                                   optimization_problem=model.optimization_problem,
                                   annealer=None)

    # Validation and test results
    val_res = trainer.validate(model=best_model, dataloaders=validation_dl)
    test_res = trainer.test(model=best_model, dataloaders=test_dl)

    return val_res, test_res

########################################################################################################################


def test_model(run_dictionaries: List[Dict],
               results_filepath: str):
    """
    Run the testing routines.
    :param run_dictionaries: list of dict; a list of dictionaries with some information useful to run the evaluation.
    :param results_filepath: str; where the experiment results are loaded from.
    :return:
    """

    # Keep track of results across runs for each method
    methods_val_res = dict()
    methods_test_res = dict()

    # For each run...
    for run_dictionary in run_dictionaries:

        # Get the method name
        model_name = run_dictionary['model_name']

        # Get the numpy and dataset random split seeds
        numpy_seed = run_dictionary['numpy_seed']
        rnd_split_seed = run_dictionary['rnd_split_seed']

        # If this is the first run of 'model_name' method then create an empty dictionary for both val and test sets
        if model_name not in methods_val_res.keys():
            methods_val_res[model_name] = dict()

        if model_name not in methods_test_res.keys():
            methods_test_res[model_name] = dict()

        # Get val and test results for the current run
        val_res, test_res = evaluate_model(run_dictionary=run_dictionary, experiment_path=results_filepath)

        # PyTorch lighting Trainer validate and test methods return results as a list with a single item
        assert len(val_res) == 1, "val_res is expected to have a single item"
        assert len(test_res) == 1, "test_res is expected to have a single item"

        val_res = val_res[0]
        test_res = test_res[0]

        prefix = os.path.join(results_filepath, f'seed-{numpy_seed}', f'rnd-split-seed-{rnd_split_seed}', model_name)

        with open(os.path.join(prefix, 'test-res.pkl'), 'wb') as file:
            pickle.dump(test_res, file)


########################################################################################################################


def main():
    """
    Main function to run experiments.
    :return:
    """

    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config_dir', type=str, help='Configuration file loadpath')
    parser.add_argument('res_dir', type=str, help='Results savepath')
    parser.add_argument('--mode',
                        type=str,
                        choices=['train', 'test'],
                        required=True,
                        help='Choose to either train or test the model')
    parser.add_argument('--wandb', action='store_true', help='Set this flag to log with W&B')
    parser.add_argument('--wandb-user',
                        type=str,
                        required=False,
                        help='Your W&B username')

    # Parse the arguments
    args = parser.parse_args()
    config_filepath = args.config_dir
    results_filepath = args.res_dir
    mode = args.mode
    wandb = args.wandb

    if wandb:
        import wandb
        wandb_user = args.wandb_user
        assert wandb_user is not None
    else:
        wandb_user = None

    # Read configuration files
    with open(config_filepath) as json_file:
        config = json.load(json_file)

    # Build run dictionaries for parallelization
    run_dictionaries = list()
    wandb_configs = list()

    # Get the dataset splits and generation seeds
    numpy_seeds = config['numpy_seed']
    rnd_split_seeds = config['rnd_split_seed']
    num_runs_per_seed = config['num_runs_per_seed']

    # shared_config contains the configuration shared among all the methods
    shared_config = config.copy()
    del shared_config['numpy_seed']
    del shared_config['rnd_split_seed']
    del shared_config['method_configurations']

    # Create a dictionary with the arguments for each run

    # For each dataset...
    for numpy_sd in numpy_seeds:

        # For each training/validation/test split...
        for rnd_split_sd in rnd_split_seeds:

            # print(f'\nDataset n.{numpy_sd} | split n.{rnd_split_sd}')

            # Set PyTorch random seed
            manual_seed(config['torch_seed'])

            # Create train, validation and test PyTorch data loaders and the optimization problem instance
            train_dl, validation_dl, test_dl, optimization_problem = \
                build_data_loaders(config, rnd_split_seed=rnd_split_sd, dataset_seed=numpy_sd)

            # For each training run...
            for run in range(num_runs_per_seed):

                # Create the DFL models
                models = build_models(config=config, train_dl=train_dl, optimization_problem=optimization_problem)

                # Create a dictionary for each method
                for model_name, model_instance in models.items():

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

                    wandb_cfg = shared_config.copy()
                    wandb_cfg["model_name"] = model_name
                    wandb_cfg["run"] = run
                    wandb_cfg["numpy_seed"] = numpy_sd
                    wandb_cfg["rnd_split"] = rnd_split_sd
                    wandb_cfg.update(model_instance.get_hyperparams())
                    wandb_configs.append(wandb_cfg)

    # Either train the model...
    if mode == 'train':

        # Create the directory if it does not exist
        if not os.path.exists(results_filepath):
            os.makedirs(results_filepath)

        # Make a copy of the configuration file in the experiment folder
        shutil.copyfile(sys.argv[1], os.path.join(results_filepath, "config.json"))

        # run_args = [(run_dict, results_filepath) for run_dict in run_dictionaries]

        run_idx = 0

        # Run the training routine
        for run_args, wandb_args in zip(run_dictionaries, wandb_configs):
            train_model(run_args, results_filepath, wandb_args, wandb_user)
            run_idx += 1

    # ... or test it
    else:

        test_model(run_dictionaries=run_dictionaries,
                   results_filepath=results_filepath)


########################################################################################################################


if __name__ == '__main__':
    main()
