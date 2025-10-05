import pickle
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from models.models import KnapsackExtractWeightsFromFeatures, SetCoverExtractDemandsFromFeatures, KnapsackExtractCapacityFromFeatures


########################################################################################################################


def load_metric(filepath_prefix, ds_seeds, split_seeds):

    all_values = list()
    all_num_epochs = list()

    for i in ds_seeds:
        for j in split_seeds:
            filepath = os.path.join(filepath_prefix, f'seed-{i}', f'rnd-split-seed-{j}')

            with open(os.path.join(filepath, 'val_metrics.pkl'), 'rb') as file:
                train_metrics = pickle.load(file)
                epochs = len(train_metrics)
                all_num_epochs.append(epochs)

            metrics = pd.read_csv(os.path.join(filepath, 'metrics.csv'))
            all_values.append(metrics)

    all_metrics_values = pd.concat(all_values, axis=0)
    mean_metrics_values = all_metrics_values.mean(axis=0)
    std_metric_values = all_metrics_values.std(axis=0)
    avg_metric_values = pd.concat([mean_metrics_values, std_metric_values], axis=1)
    avg_metric_values.columns = ['mean', 'std dev']

    for name in mean_metrics_values.index:
        mean = avg_metric_values.loc[name]['mean']
        std_dev = avg_metric_values.loc[name]['std dev']

        print(f'{name}: {mean} +- {std_dev}')

    mean_epochs = np.mean(all_num_epochs)
    std_dev_epochs = np.std(all_num_epochs)
    print(f'epochs: {mean_epochs} +- {std_dev_epochs}')


########################################################################################################################


def backbone_eval(datasets_base_filepath,
                  backbone_base_filepath,
                  in_features_filename,
                  out_features_filename,
                  opt_prob_params_filename,
                  dataset_seeds,
                  split_seeds):

    all_mse = list()

    for ds_seed in dataset_seeds:
        for splt_seed in split_seeds:
            in_features_filepath = os.path.join(datasets_base_filepath, f'seed-{ds_seed}', in_features_filename)
            out_features_filepath = os.path.join(datasets_base_filepath, f'seed-{ds_seed}', out_features_filename)
            opt_prob_params_filepath = os.path.join(datasets_base_filepath, f'seed-{ds_seed}', opt_prob_params_filename)
            # capacity_filepath = os.path.join(datasets_base_filepath,
            #                                  f'seed-{ds_seed}',
            #                                  'capacity_n_1000_input_dim_5_output_dim_75_mult_noise_0.1_add_noise_0.03_deg_5_relative_capacity_0.2_correlation_type_1_rho_0.npy')

            in_features = pd.read_csv(in_features_filepath, index_col=0).values
            out_features = pd.read_csv(out_features_filepath, index_col=0).values
            # capacity = np.load(capacity_filepath).item()
            weights = np.load(opt_prob_params_filepath)

            train_in_features, test_in_features, \
            train_out_features, test_out_features = \
                train_test_split(in_features, out_features, test_size=0.2, random_state=splt_seed)

            min_out_feature = np.min(train_out_features)
            max_out_feature = np.max(train_out_features)

            backbone_filepath = os.path.join(backbone_base_filepath, f'seed-{ds_seed}', f'rnd-split-seed-{splt_seed}', 'backbone')

            """backbone = KnapsackExtractWeightsFromFeatures(kp_dim=75,
                                                          input_dim=5,
                                                          knapsack_capacity=capacity,
                                                          weight_min=min_out_feature,
                                                          weight_max=max_out_feature,
                                                          out_features=75)"""

            """backbone = SetCoverExtractDemandsFromFeatures(input_dim=5,
                                                          out_features=10,
                                                          demand_min=min_out_feature,
                                                          demand_max=max_out_feature)"""

            backbone = KnapsackExtractCapacityFromFeatures(kp_dim=75,
                                                           input_dim=5,
                                                           capacity_max=max_out_feature,
                                                           capacity_min=min_out_feature,
                                                           out_features=1)

            backbone.load_state_dict(torch.load(backbone_filepath))

            test_in_features = torch.as_tensor(test_in_features).float()
            weights = np.expand_dims(weights, axis=0)
            weights = np.tile(weights, (len(test_in_features), 1))
            weights = torch.as_tensor(weights).float()
            constraints = backbone(test_in_features, weights).detach().numpy()
            preds = np.squeeze(constraints)[:, :-1]
            # preds = np.squeeze(constraints)
            mse = np.mean(np.square(preds - test_out_features))
            all_mse.append(mse)

    mean_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)

    print(f'MSE: {mean_mse} +- {std_mse}')

########################################################################################################################


if __name__ == '__main__':
    penalty = 20
    ds_seeds = [0, 1, 2]
    split_seeds = [0, 1, 2]
    dim_str = '50-items'
    load_metric(f'results/stochastic_weights_kp/{dim_str}/penalty-{penalty}', ds_seeds=ds_seeds, split_seeds=split_seeds)
    backbone_eval(datasets_base_filepath=f'data/datasets/stochastic_capacity_kp/',
                  backbone_base_filepath=f'results/stochastic_capacity_kp/{dim_str}/penalty-{penalty}/',
                  in_features_filename='features_n_1000_input_dim_5_output_dim_75_mult_noise_0.1_deg_5_correlation_type_1_rho_0.csv',
                  out_features_filename='targets_n_1000_input_dim_5_output_dim_75_mult_noise_0.1_deg_5_correlation_type_1_rho_0.csv',
                  opt_prob_params_filename='weights_n_1000_input_dim_5_output_dim_75_mult_noise_0.1_deg_5_correlation_type_1_rho_0.npy',
                  dataset_seeds=ds_seeds,
                  split_seeds=split_seeds)
