import os
import sys

import numpy as np
import pickle
import ray

from data import load_dataset
from trainer import get_trainer
from utils.utils import print_acc, load_with_default_yaml, save_dict_as_one_line_csv
from utils import MSE_SOL_STR

########################################################################################################################


def main(working_dir, torch_seed, seeds, rnd_split_seeds,  train_epochs, patience, eval_every, use_ray, ray_params, data_params, trainer_params):

    if use_ray:
        ray.init(**ray_params)

    for seed in seeds:
        for rnd_split_seed in rnd_split_seeds:

            penalty = trainer_params['penalty']

            current_working_dir = \
                os.path.join(working_dir,
                             f'penalty-{penalty}',
                             f'seed-{seed}',
                             f'rnd-split-seed-{rnd_split_seed}')

            data_params['seed'] = seed
            data_params['rnd_split_seed'] = rnd_split_seed
            data_params['penalty'] = penalty

            (train_iterator, val_iterator, test_iterator), metadata = load_dataset(**data_params)
            trainer = get_trainer(seed=torch_seed,
                                  train_iterator=train_iterator,
                                  val_iterator=val_iterator,
                                  test_iterator=test_iterator,
                                  metadata=metadata,
                                  **trainer_params)

            eval_metrics = trainer.evaluate(test=False)
            print_acc(eval_metrics, prefix='val')

            best_mse = np.inf
            not_improved_count = 0

            train_metrics_epochs = list()
            eval_metrics_epochs = list()

            for i in range(train_epochs):
                train_metrics = trainer.train_epoch()
                train_metrics_epochs.append(train_metrics)

                print_acc(train_metrics, prefix='train')

                eval_metrics = trainer.evaluate(test=False)
                eval_metrics_epochs.append(eval_metrics)

                print_acc(eval_metrics, prefix='val')
                mse = eval_metrics['val_' + MSE_SOL_STR]

                if mse < best_mse:
                    best_mse = mse
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                    if not_improved_count == patience:
                        break

            eval_metrics = trainer.evaluate(test=False)
            print_acc(eval_metrics, prefix='val')

            test_metrics = trainer.evaluate(test=True)
            print_acc(test_metrics, prefix='test')

            if use_ray:
                ray.shutdown()
            metrics = dict(**train_metrics, **eval_metrics, **test_metrics)
            save_dict_as_one_line_csv(metrics, filename=os.path.join(current_working_dir, "metrics.csv"))

            with open(os.path.join(current_working_dir, "train_metrics.pkl"), "wb") as file:
                pickle.dump(train_metrics_epochs, file)

            with open(os.path.join(current_working_dir, "val_metrics.pkl"), "wb") as file:
                pickle.dump(eval_metrics_epochs, file)

            trainer.save(current_working_dir)

            trainer.build_model(**trainer_params['model_params'])
            trainer.load(current_working_dir)

########################################################################################################################


if __name__ == "__main__":
    param_path = sys.argv[1]
    param_dict = load_with_default_yaml(path=param_path)

    main(**param_dict)
