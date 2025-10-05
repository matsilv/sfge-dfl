import numpy as np
import os

########################################################################################################################


def print_metric(filepath, split_seeds):
    mse_vals = list()

    for seed in split_seeds:
        filename = os.path.join(filepath, f'rnd-split-{seed}', 'mse.npy')
        mse = np.load(filename)
        mse_vals.append(mse)

    mean_mse = np.mean(mse_vals)
    std_dev_mse = np.std(mse_vals)

    print(f'MSE: {mean_mse} +- {std_dev_mse}')

########################################################################################################################


if __name__ == '__main__':
    print_metric(filepath='results/capacity-75/penalty-2', split_seeds=np.arange(10))

