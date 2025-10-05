"""
    Methods to display the results.
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import pickle
import re

from experiments.pfl_plus_saa import RELATIVE_REGRET, RUNTIME

from typing import Tuple, List, Dict

########################################################################################################################

# Define the colors for the custom colormap
dark_color = 'navy'
light_color = 'lightblue'

# Create the custom colormap
colors = [dark_color, light_color]
cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)

# FIXME: for old experiments there is a mismatch in the metrics names
TEST_METRICS_NAMES = ['test_mse', 'test_regret', 'test_relative_regret', 'epoch']
VAL_METRICS_NAMES = ['val_mse', 'val_regret', 'val_relative_regret', 'epoch']
sns.set_style('darkgrid')

########################################################################################################################


def _compute_metrics(experiment_filepath: str,
                     metrics_names: List[str],
                     validation: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean and std dev of the validation/test metrics over different runs.
    :param experiment_filepath: str; where the runs results are loaded from.
    :return: tuple of 2 pandas.Dataframe; mean and std dev for each metric and method, and metrics for each run and
             method.
    """

    methods_metrics_mean_std_dev = dict()
    methods_metrics_runs = dict()

    # List all methods directories in the experiments folder
    for method_dir in os.listdir(experiment_filepath):

        method_metrics_list = list()

        # Keep track of the runs results for each method
        metrics_df = pd.DataFrame(columns=['Run'] + metrics_names)
        method_path = os.path.join(experiment_filepath, method_dir)

        # List all the run folders
        if os.path.isdir(method_path):

            idx = 0

            # For each run folder...
            for run_dir in os.listdir(method_path):
                run_path = os.path.join(method_path, run_dir)

                # ...load and store metrics
                if os.path.isdir(run_path):
                    metrics = pd.read_csv(os.path.join(run_path, 'metrics.csv'))
                    metrics = metrics[metrics_names]

                    # The second to last row of the metrics file is the validation result...
                    if validation:
                        last_epoch_idx = -2
                    # ... whereas the last row of the metric file is the test result
                    else:
                        last_epoch_idx = -1

                    last_epoch_res = metrics[metrics_names].iloc[last_epoch_idx].values
                    method_metrics_list.append(last_epoch_res)

                    idx += 1

            # Create a dafaframe with the metrics for each run
            metrics_df['Run'] = np.arange(idx)
            metrics_df[metrics_names] = method_metrics_list
            methods_metrics_runs[method_dir] = metrics_df

            # Compute mean and std dev of the metric over the runs
            mean_std_dev_df = pd.DataFrame(index=metrics_names, columns=['mean', 'stddev'])
            mean_std_dev_df['mean'] = metrics_df[metrics_names].apply(np.mean, axis=0).values
            mean_std_dev_df['stddev'] = metrics_df.apply(np.std, axis=0)

            methods_metrics_mean_std_dev[method_dir] = mean_std_dev_df

    return methods_metrics_mean_std_dev, methods_metrics_runs


########################################################################################################################


def print_metrics(experiment_filepath: str, metrics_names: List[str]) -> pd.DataFrame:
    """
    Compute and return mean and std dev of test metrics for each method.
    :param experiment_filepath: str; where the run results are loaded from.
    :param metrics_names: List[str]; names of the metrics to compute.
    :return: pd.DataFrame; mean and std dev of test metrics for each method.
    """
    try:
        test_metrics, _ = \
            _compute_metrics(experiment_filepath=experiment_filepath,
                             metrics_names=metrics_names,
                             validation=False)

        results = pd.DataFrame(columns=['Method'] + metrics_names)

        # For each method...
        for method_name, test_metrics_df in test_metrics.items():
            method_results = {'Method': method_name}

            for name, mean in test_metrics_df['mean'].items():
                std_dev = test_metrics_df['stddev'][name]
                method_results[name] = f'{mean} +- {std_dev}'

            results = results.append(method_results, ignore_index=True)

        return results
    except Exception as e:
        print(f"An error occurred while reading files: {str(e)}")

########################################################################################################################


def plot_saa_metric_wrt_scenarios(axis: plt.axis,
                                  res_dir: str,
                                  metric_name: str,
                                  y_axis_unit: str = None,
                                  normalize: bool = False,
                                  title: str = None,
                                  plot_sfge: bool = True,
                                  time_limit: int = 30,
                                  plot_log: bool = False):
    """
    Plot a metric of the Sample Average Approximation (SAA) w.r.t. the number of sampled scenarios obtained from the
    probabilistic model.
    :param axis: matplotlib.pyplot.axis; already existing axis where the results are displayed.
    :param res_dir: str; where results are loaded from. In this directory, we expected to found one sub-directory
    (named seed-{seed}) for each seed used to generate the datasets.
    :param metric_name: str; the name of the metric to evaluate.
    :param y_axis_unit: str; unit measure of the y-axis.
    :param normalize: bool; true to normalize the metric w.r.t. to SFGE results.
    :param title: str; the title of the plot.
    :param plot_sfge: bool; True to plot SFGE metrics.
    :param time_limit: int; if you are running the experiments and suspending your laptop when SAA is running, Gurobi
     still keep track of time, even tough the actual time limit is setup. To prevent unfaithful plots of the runtime, we
     have to clip it according to the time limit.
    :param plot_log: bool; if True, plot the y-axis values in logarithmic scale.
    :return:
    """

    # Convert the metric name to lower case
    metric_name = metric_name.lower()

    # Only runtime and relative regret are supported by now
    assert metric_name in ['runtime', 'relative regret']

    # Lookup dictionaries to avoid if/else statements
    sfge_metric_filenames = {
        'runtime': 'test_runtime',
        'relative regret': 'test_relative_regret'
    }

    saa_metric_names = {
        'runtime': RUNTIME,
        'relative regret': RELATIVE_REGRET
    }

    # Keep track of all the seeds directories
    all_seed_dirs = list()

    # For each file in the main results directory...
    for directory in os.listdir(res_dir):

        # ... check if the file is a directory...
        directory_path = os.path.join(res_dir, directory)

        if os.path.isdir(directory_path):
            # ... if so, be sure that the sub-directory name starts with "seed-"
            assert directory.startswith('seed-'), 'Only subdirectories with prefix "seed-" are allowed'

            all_seed_dirs.append(directory_path)

    # Keep track of all the random split sub-directories
    all_rnd_split_dirs = list()

    # For each seed sub-directory...
    for seed_path in all_seed_dirs:

        rnd_split_dirs = os.listdir(seed_path)

        for directory in rnd_split_dirs:

            # ... check if the file is a directory...
            directory_path = os.path.join(seed_path, directory)

            if os.path.isdir(directory_path):
                # ... if so, be sure that the sub-directory name starts with "rnd-split-seed-"
                assert directory.startswith('rnd-split-seed-'), \
                    'Only subdirectories with prefix "rnd-split-seed-" are allowed'

                all_rnd_split_dirs.append(directory_path)

    # Keep track of the Maximum Likelihood (MLE) and Score-function gradient estimation (SFGE) directories
    all_mle_dirs = list()
    all_sfge_dirs = list()

    # For each seed sub-directory...
    for rnd_split_path in all_rnd_split_dirs:

        methods_dirs = os.listdir(rnd_split_path)

        # ... be sure that there are "MLE" and "SFGE" sub-directories
        assert 'MLE' in methods_dirs, 'A "MLE" folder is expected'
        assert 'SFGE' in methods_dirs, 'A "SFGE" folder is expected'

        directory_path = os.path.join(rnd_split_path, 'MLE')
        all_mle_dirs.append(directory_path)

        directory_path = os.path.join(rnd_split_path, 'SFGE')
        all_sfge_dirs.append(directory_path)

    # Keep track of the runs sub-directories
    all_mle_runs_dirs = list()
    # Keep track of SFGE metric
    all_sfge_metric = list()

    # For each "MLE" and "SFGE" paths (one for each seed)...
    for mle_path, sfge_path in zip(all_mle_dirs, all_sfge_dirs):

        # ... list all the files...
        mle_runs_dirs = os.listdir(mle_path)

        # ... for each of these files...
        for mle_directory in mle_runs_dirs:

            mle_directory_path = os.path.join(mle_path, mle_directory)

            # ... consider the only directories
            if os.path.isdir(mle_directory_path):
                # Inside the "MLE", we expect to find the "run_{run}" sub-directories
                assert mle_directory.startswith('run_'), 'Only subdirectories with "run_" prefix are allowed'
                all_mle_runs_dirs.append(mle_directory_path)

        # Load the relative regret results of SFGE for the current seed
        with open(os.path.join(sfge_path, 'test-res.pkl'), 'rb') as file:
            test_res = pickle.load(file)
            sfge_metric = test_res[sfge_metric_filenames[metric_name]]

        all_sfge_metric.append(sfge_metric)

    # Now keep track of all the results for each seed, method ("MLE") and run
    all_mle_res = dict()

    for mle_run_dir in all_mle_runs_dirs:

        # SAA results for each scenario value are saved in a dictionary
        with open(os.path.join(mle_run_dir, 'saa-results.pkl'), 'rb') as file:
            mle_res = pickle.load(file)

        # We only care about the relative regret (not the cost)
        for key in mle_res.keys():

            mle_metric = mle_res[key][saa_metric_names[metric_name]]
            mle_metric = np.abs(mle_metric)

            if metric_name == 'runtime':
                mle_metric = np.clip(mle_metric, a_min=0, a_max=time_limit)

            # Keep track of the relative regret values for each number of scenarios (the key of the dictionary)
            if key not in all_mle_res.keys():
                all_mle_res[key] = [mle_metric]
            else:
                all_mle_res[key].append(mle_metric)

    # Compute the mean and std dev for each number of scenarios value
    mean_mle_res = dict()
    std_mle_res = dict()

    # Compute the mean and std dev of the relative regret (across seeds and runs) for SFGE
    mean_sfge_metric = np.mean(all_sfge_metric)
    std_sfge_metric = np.std(all_sfge_metric)

    for key in all_mle_res.keys():

        if normalize:
            all_mle_res[key] = np.asarray(all_mle_res[key]) / mean_sfge_metric

        if plot_log:
            all_mle_res[key] = np.log10(all_mle_res[key])

        mean_val = np.mean(all_mle_res[key])
        mean_mle_res[key] = mean_val
        std_mle_res[key] = np.std(all_mle_res[key])

    # Convert from "dictkeys" and "dictvalues" to list
    num_scenarios = list(mean_mle_res.keys())
    mean_rel_regret = list(mean_mle_res.values())
    mean_rel_regret = np.asarray(mean_rel_regret)
    std_rel_regret = list(std_mle_res.values())
    std_rel_regret = np.asarray(std_rel_regret)

    # The relative regret of SFGE is scenario-independent so we simply repeat the same value (to plot an horizontal
    # line)
    if normalize:
        if plot_log:
            mean_sfge_metric = 0
        else:
            mean_sfge_metric = 1

    mean_sfge_metric = np.tile(mean_sfge_metric, reps=len(num_scenarios))
    std_sfge_metric = np.tile(std_sfge_metric, reps=len(num_scenarios))

    # Plot mean and std dev for SFGE and SAA
    axis.errorbar(num_scenarios,
                  mean_rel_regret,
                  std_rel_regret,
                  fmt='--o',
                  label='PFL+SAA',
                  color='crimson',
                  ecolor='crimson',
                  capsize=5)

    if plot_sfge:

        axis.plot(num_scenarios, mean_sfge_metric, linestyle='--', color='green', label='SFGE')
        axis.fill_between(num_scenarios,
                          mean_sfge_metric - std_sfge_metric,
                          mean_sfge_metric + std_sfge_metric,
                          color='green',
                          alpha=0.1)

    if title is None:
        title = metric_name.capitalize()

    if y_axis_unit is not None:
        y_axis_unit += '(' + y_axis_unit + ')'

    # axis.set_title(title, fontweight='bold')
    axis.legend(fontsize=11)

########################################################################################################################


def print_from_csv(filepath_prefix: str,
                   method: str,
                   seeds: List[int],
                   splits: List[int],
                   runs: List[int],
                   metric: str,
                   last_epoch=-2):
    """
    Display the results saved in a CSV file.
    :param filepath_prefix: str; base filepath.
    :param method: str; the name of the method.
    :param seeds: list of int; the list of seeds used to generate the datasets.
    :param splits: list of int; the list of seeds used to split the datasets.
    :param runs: list of int; the list of run identifier (as integer).
    :param metric: str; the metric name.
    :param last_epoch: int; set the upper bound of the training interval that will be considered when displaying the
    results. We use the same convention as Python indexing from the end of an iterable.
    :return:
    """

    # Keep track of all the metric values
    all_metric_vals = list()

    # For each dataset seed...
    for seed in seeds:
        # ...for each dataset split seed...
        for split_seed in splits:
            # ...for each run identifier...
            for run in runs:

                metric_filepath = \
                    os.path.join(filepath_prefix,
                                 f'seed-{seed}',
                                 f'rnd-split-seed-{split_seed}',
                                 method,
                                 f'run_{run}',
                                 'metrics.csv')

                # ... load the metric value
                metrics = pd.read_csv(metric_filepath)
                metric_vals = metrics[metric].values[last_epoch]
                all_metric_vals.append(metric_vals)

    mean_vals = np.mean(all_metric_vals)
    std_dev_vals = np.std(all_metric_vals)

    print(f'{metric}: {mean_vals} +- {std_dev_vals}\n')

    return all_metric_vals

########################################################################################################################


def print_from_numpy(method: str,
                     dim: str,
                     problem: str,
                     seeds: List[int],
                     splits: List[int],
                     metric: str):
    """
    Display the results saved in .npy format.
    :param method: str; the name of the method.
    :param dim: str; an identifier of the problem dimension.
    :param problem: str; the problem name.
    :param seeds: list of int; the list of seeds used to generate the datasets.
    :param splits: list of int; the list of seeds used to split the datasets.
    :param metric: str; the metric name.
    :return:
    """

    # Keep track of the metric values
    all_metric_vals = list()

    # Filepath prefix
    filepath_prefix = os.path.join('experiments', problem, dim)

    # For each dataset seed...
    for seed in seeds:

        # ...for each dataset split seed
        for split_seed in splits:
            metric_filepath = os.path.join(filepath_prefix,
                                           f'seed-{seed}',
                                           f'rnd-split-seed-{split_seed}',
                                           method,
                                           f'{metric}.npy')

            metric_val = np.load(metric_filepath)
            all_metric_vals.append(metric_val)

    # Convert to numpy array as print to output
    all_metric_vals = np.asarray(all_metric_vals)

    print(f'{metric}: {np.mean(all_metric_vals)} +- {np.std(all_metric_vals)}\n')

########################################################################################################################


def print_num_epochs(filepath_prefix: str,
                     method: str,
                     seeds: List[int],
                     splits: List[int],
                     runs: List[int]):
    """
    Display the number of training epochs.
    :param filepath_prefix: str; the base filepath.
    :param method: str; the name of the method.
    :param seeds: list of int; the list of seeds used to generate the datasets.
    :param splits: list of int; the list of seeds used to split the datasets.
    :param runs: list of int; the list of runs for a given Torch seed.
    """

    # Keep track of the number of epochs
    all_epochs = list()

    # For each dataset seed...
    for seed in seeds:
        # ...for each dataset split seed
        for split_seed in splits:
            # ...for each run identifier...
            for run in runs:

                try:

                    metric_filepath = \
                        os.path.join(filepath_prefix,
                                     f'seed-{seed}',
                                     f'rnd-split-seed-{split_seed}',
                                     method,
                                     f'run_{run}',
                                     'metrics.csv')

                    metrics = pd.read_csv(metric_filepath)

                    # The last row of the validation results is the second last one of the file
                    epochs = metrics['epoch'].values[-2]

                    all_epochs.append(epochs)
                except FileNotFoundError:
                    all_epochs.append(np.nan)

    print(f'Epochs: {np.nanmean(all_epochs)} +- {np.nanstd(all_epochs)}\n')

    return all_epochs

########################################################################################################################


def print_from_pickle(filepath_prefix: str,
                      method: str,
                      seeds: List[int],
                      splits: List[int],
                      metric: str):
    """
    Display the results saved in a pickle file; the file name is called 'test-res.pkl'.
    :param filepath_prefix: str; the base filepath.
    :param method: str; the name of the method.
    :param seeds: list of int; the list of seeds used to generate the datasets.
    :param splits: list of int; the list of seeds used to split the datasets.
    :param metric: str; the metric name.
    :return:
    """

    # Keep track of the metric values
    all_metric_vals = list()

    # For each dataset seed...
    for seed in seeds:
        # ...for each dataset split seed
        for split_seed in splits:
            try:
                metric_filepath = \
                    os.path.join(filepath_prefix,
                                 f'seed-{seed}',
                                 f'rnd-split-seed-{split_seed}',
                                 method,
                                 'test-res.pkl')

                with open(metric_filepath, 'rb') as file:
                    metric_val = pickle.load(file)
                    all_metric_vals.append(metric_val[metric])
            except FileNotFoundError:
                all_metric_vals.append(np.nan)

    # Convert to numpy array as print to output
    all_metric_vals = np.asarray(all_metric_vals)
    mean_metric_val = np.nanmean(all_metric_vals)
    std_metric_val = np.nanstd(all_metric_vals)

    np.save(os.path.join(filepath_prefix, 'mean_' + metric), mean_metric_val)
    np.save(os.path.join(filepath_prefix, 'std_' + metric), std_metric_val)

    print(f'{metric}: {mean_metric_val} +- {std_metric_val}\n')

    # plt.bar(np.arange(len(all_metric_vals)), all_metric_vals, label=method)
    # plt.show()

    return all_metric_vals

########################################################################################################################


def generate_latex_table(column_names: List[str],
                         method_names: List[str],
                         numeric_values: List[float],
                         row_headers: List[str],
                         output_file: str):

    """
    Automatically generate the source text for a Latex table.
    :param column_names: list of str; the column names to display.
    :param method_names: list of str; the method names to display.
    :param numeric_values: list of float; the cell values.
    :param row_headers: list of str; header for multicolumn environment.
    :param output_file: str; where the table is saved to.
    :return:
    """

    table = r"""
    \begin{table}[]
    \centering
    \caption{\textsc{MLE} and \ouracronym{} results on the KP with uncertain capacity of different sizes and for different penalty coefficient values.}
    \label{table:stochastic_capacity_kp}
    \vspace{2pt}
    \addtolength{\tabcolsep}{2pt}
    \small
    \begin{tabular}{l""" + 'c' * len(column_names) + r"""}
    \toprule
    \textit{Method} & """ + ' & '.join([r"""\textit""" + f'{{{column}}}' for column in column_names]) + r""" \\
    \midrule
    """

    # For each multicolumn environment subtable...
    for i, header in enumerate(row_headers):
        table += r"""
        \multicolumn{""" + str(len(column_names) + 1) + r"""}{c}{""" + header + r"""} \\"""
        table += r"""
        \midrule
        """

        # For each method...
        for j, method in enumerate(method_names):
            table += f"{method}"
            # Visualize the data for each metric
            for data in numeric_values[i][j]:
                formatted_data = [f'{data[0]:.3f} $\\pm$ {data[1]:.3f}']
                table += ' & ' + ' & '.join(formatted_data)
            table += r""" \\"""
            table += r"""
        \midrule
        """

    table += r"""
    \bottomrule
    \end{tabular}
    \end{table}"""

    with open(output_file, 'w') as file:
        file.write(table)

    return table

########################################################################################################################


# FIXME: better separation with "generate_latex_table"
def write_latex_table(column_names: List[str],
                      method_names: List[str],
                      metric_names: List[str],
                      problem_name: str,
                      problem_dim: str,
                      seeds: List[int],
                      splits: List[int],
                      runs: List[int],
                      penalties: List[int],
                      filename: str):
    """
    Load the experiments results and generate a Latex table. This method works the stochastic problems only.
    :param column_names: list of str; the column names to display.
    :param method_names: list of str; the method names to display.
    :param metric_names: list of str; the metric names loaded from file.
    :param problem_name: str; the problem name used in the filepath.
    :param problem_dim: str; the problem dimension used in the filepath.
    :param seeds: list of int; the list of seeds used to generate the datasets.
    :param splits: list of int; the list of seeds used to split the datasets.
    :param runs: list of int; the list of runs for a given Torch seed.
    :param penalties: list of int; the list of penalty coefficients for the stochastic problem.
    :param filename: str; the output filename.
    :return:
    """

    numeric_values = list()
    row_headers = list()

    # for penalty in penalties:

    # dim_str = os.path.join(problem_dim, f'penalty-{penalty}')
    # filepath_prefix = os.path.join('experiments', problem_name, 'dpo', dim_str)
    filepath_prefix = os.path.join('experiments', problem_name, problem_dim)
    # row_headers.append(f'{problem_dim}, penalty-{penalty}')

    penalty_res = list()

    for method in method_names:

        print(f'Method: {method}\n')

        metric_test_mean_values = list()
        metric_test_std_dev_values = list()
        metric_val_mean_values = list()
        metric_val_std_dev_values = list()
        epochs_mean_values = list()
        epochs_std_dev_values = list()

        method_res = list()

        for metric in metric_names:
            all_test_values = \
                print_from_pickle(filepath_prefix=filepath_prefix,
                                  method=method,
                                  metric='test_' + metric,
                                  seeds=seeds,
                                  splits=splits)

            test_mean_value = np.mean(all_test_values)
            test_std_dev_value = np.std(all_test_values)

            metric_test_mean_values.append(test_mean_value)
            metric_test_std_dev_values.append(test_std_dev_value)

            all_val_values = \
                print_from_csv(filepath_prefix=filepath_prefix,
                               method=method,
                               runs=runs,
                               metric='val_' + metric,
                               seeds=seeds,
                               splits=splits)

            val_mean_value = np.mean(all_val_values)
            val_std_dev_value = np.std(all_val_values)

            metric_val_mean_values.append(val_mean_value)
            metric_val_std_dev_values.append(val_std_dev_value)

            method_res.append((test_mean_value, test_std_dev_value))

        all_epochs = \
            print_num_epochs(filepath_prefix=filepath_prefix,
                             method=method,
                             seeds=seeds,
                             splits=splits,
                             runs=[0])

        epoch_mean_val = np.mean(all_epochs)
        epoch_std_dev_val = np.std(all_epochs)
        epochs_mean_values.append(epoch_mean_val)
        epochs_std_dev_values.append(epoch_std_dev_val)

        method_res.append((epoch_mean_val, epoch_std_dev_val))

        print(f'[Validation] - {metric}: {val_mean_value} +- {val_std_dev_value}')
        print(f'[Test] - {metric}: {test_mean_value} +- {test_std_dev_value}')

        print('\n' + '-' * 100 + '\n')

        penalty_res.append(method_res)

    numeric_values.append(penalty_res)

    generate_latex_table(column_names=column_names,
                         method_names=method_names,
                         numeric_values=numeric_values,
                         output_file=filename,
                         row_headers=row_headers)

########################################################################################################################


def regret_barplots(res_paths, names, methods):
    res = pd.DataFrame(index=names, columns=[str(s) for s in [0, 1, 2, 3, 4]])

    for path, n, mtd in zip(res_paths, names, methods):
        regret_vals = print_from_pickle(filepath_prefix=path,
                                        method=mtd,
                                        seeds=[0, 1, 2, 3, 4],
                                        splits=[0],
                                        metric='test_relative_regret')
        res.loc[n] = regret_vals

    sns.barplot(x=names, y=res.mean(axis=1), yerr=res.std(axis=1))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title('Relative regret')
    plt.savefig('predicted-vs-trainable-std-dev', bbox_inches='tight')


########################################################################################################################

def compare_convergence_speed(ds_seeds, rnd_split_seeds, base_filepaths, metric_name, method_names, labels):

    for filepath, method, lbl in zip(base_filepaths, method_names, labels):

        all_metrics = []
        max_lenght = 0

        for seed in ds_seeds:
            for split_seed in rnd_split_seeds:
                metric_filepath = f'{filepath}/seed-{seed}/rnd-split-seed-{split_seed}/{method}/run_0/metrics.csv'
                res = pd.read_csv(metric_filepath)
                vals = list(res[metric_name].values)
                all_metrics.append(vals[:-1])
                if len(vals) > max_lenght:
                    max_lenght = len(vals)

        for i, val in enumerate(all_metrics):
            if len(val) < max_lenght:
                for _ in range(max_lenght - len(val)):
                    all_metrics[i].append(np.nan)

        all_metrics_np = np.asarray(all_metrics)

        plt.plot(np.arange(max_lenght), np.nanmean(all_metrics_np, axis=0), label=lbl)
        plt.fill_between(np.arange(max_lenght), np.nanmean(all_metrics_np, axis=0) - np.nanstd(all_metrics_np, axis=0),
                         np.nanmean(all_metrics_np, axis=0) + np.nanstd(all_metrics_np, axis=0), alpha=0.2)
        plt.legend()
        plt.xlabel('Epoch')
        plt.title(metric_name, fontsize=12, fontweight='bold')
    # plt.savefig('std-vs-no-std.png', dpi=500)
    plt.show()

########################################################################################################################


def hyperparams_tuning(ds_seeds, rnd_split_seeds, base_filepaths, metric_name, method_names, labels):

    all_res = []
    all_epochs = []

    for filepath, name, lbl in zip(base_filepaths, method_names, labels):
        res = print_from_pickle(filepath_prefix=filepath,
                                method=name,
                                seeds=ds_seeds,
                                splits=rnd_split_seeds,
                                metric=metric_name)

        epochs = print_num_epochs(filepath_prefix=filepath,
                                  method=name,
                                  seeds=ds_seeds,
                                  splits=rnd_split_seeds,
                                  runs=[0])

        all_res.append(res)
        all_epochs.append(epochs)

    all_res = np.asarray(all_res)
    all_epochs = np.asarray(all_epochs)
    # all_epochs = np.log10(all_epochs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Primo grafico - Test relative regret (a sinistra)
    sns.barplot(x=labels, y=np.nanmean(all_res, axis=1), yerr=np.nanstd(all_res, axis=1), ax=axes[0])
    axes[0].set_title('Test relative regret', fontweight='bold')
    axes[0].tick_params(axis='x', labelsize=11)
    axes[0].tick_params(axis='y', labelsize=11)
    axes[0].set_xlabel(r'$\sigma$', fontsize=12)

    # Secondo grafico - Epochs (a destra)
    sns.barplot(x=labels, y=np.nanmean(all_epochs, axis=1), yerr=np.nanstd(all_epochs, axis=1), ax=axes[1])
    axes[1].set_title('Epochs', fontweight='bold')
    axes[1].tick_params(axis='x', labelsize=11)
    axes[1].tick_params(axis='y', labelsize=11)
    axes[1].set_xlabel(r'$\sigma$', fontsize=12)

    plt.tight_layout()
    plt.savefig('static-vs-trainable-std-dev.png', dpi=300)


########################################################################################################################


def plot_std_dev(exp_filepath: str, seeds: List[int], splits: List[int], img_name: str, title: str):
    all_exp_log_std_dev = []

    for _seed in seeds:
        for _split in splits:
            all_files = os.listdir(f'{exp_filepath}/seed-{_seed}/rnd-split-seed-{_split}/SFGE/run_0/log-std-dev')
            all_files = sorted(all_files, key=lambda x: int(re.search(r'epoch-(\d+)', x).group(1)))
            single_exp_log_std_dev = []
            for f in all_files:
                single_exp_log_std_dev.append(
                    np.load(f'{exp_filepath}/seed-{_seed}/rnd-split-seed-{_split}/SFGE/run_0/log-std-dev/' + f))
            all_exp_log_std_dev.append(single_exp_log_std_dev)


    min_len = min([len(_) for _ in all_exp_log_std_dev])
    all_exp_log_std_dev = [_exp[:min_len] for _exp in all_exp_log_std_dev]
    # for _exp in all_exp_log_std_dev:
        # if len(_exp) < max_len:
        #     _exp += [[np.nan for _ in range(len(_exp[0]))] for _ in range(max_len - len(_exp))]
        # _exp = _exp[:min_len]

    all_exp_log_std_dev = np.asarray(all_exp_log_std_dev)
    mean_log_std_dev = np.exp(np.nanmean(all_exp_log_std_dev, axis=0))
    std_log_std_dev = np.exp(np.nanstd(all_exp_log_std_dev, axis=0))

    fig, axises = plt.subplots(mean_log_std_dev.shape[1], 1, sharex=True, sharey=True, figsize=(7, 10))
    for idx, _axis in enumerate(axises):
        _axis.plot(mean_log_std_dev[:, idx])
        # _axis.fill_between(mean_log_std_dev[:, idx] - std_log_std_dev[:, idx], mean_log_std_dev[:, idx] + std_log_std_dev[:, idx], alpha=0.2)
    axises[-1].set_xlabel('Epoch')
    axises[0].set_title(title)
    plt.tight_layout()
    plt.savefig(img_name, dpi=1000)

########################################################################################################################


def plot_num_epochs_wrt_problem_dim(exp_dir: str,
                                    problem_dims: List[str],
                                    ds_seeds: List[int],
                                    rnd_split_seeds: List[int],
                                    runs: List[int],
                                    methods: List[str],
                                    x_labels: List[str] | None = None,
                                    method_labels: Dict | None = None):
    """
    The goal of this plot is to inspect whether convergence speed depends on the problem dimension.
    :param exp_dir: This is the base path where the experiments of a given problem are located (e.g. experiments/knapsack).
    We expect a group of subfolder for each problem dimension here.
    :param problem_dims:
    :param ds_seeds:
    :param rnd_split_seeds:
    :param runs:
    :param methods:
    :return:
    """

    epoch_results = {_dim: {_method: [] for _method in methods} for _dim in problem_dims}
    rel_regret_results = {_dim: {_method: [] for _method in methods} for _dim in problem_dims}

    # Gather results
    for _dim in problem_dims:
        for _ds_sd in ds_seeds:
            for _rnd_split_sd in rnd_split_seeds:
                for _method in methods:
                    for _run in runs:
                        exp_filepath = os.path.join(
                            exp_dir, _dim, f'seed-{_ds_sd}', f'rnd-split-seed-{_rnd_split_sd}', _method
                        )
                        metrics = pd.read_csv(os.path.join(exp_filepath, f'run_{_run}', 'metrics.csv'))
                        epochs = metrics['epoch'].iloc[-1]
                        epoch_results[_dim][_method].append(epochs)

                        with open(os.path.join(exp_filepath, 'test-res.pkl'), 'rb') as file:
                            test_metrics = pickle.load(file)
                            rel_regret_results[_dim][_method].append(test_metrics['test_relative_regret'])

    # Prepare data for epochs
    epoch_means = {m: [] for m in methods}
    epoch_stds = {m: [] for m in methods}

    for _dim in problem_dims:
        for _method in methods:
            vals = epoch_results[_dim][_method]
            epoch_means[_method].append(np.mean(vals))
            epoch_stds[_method].append(np.std(vals))

    # Prepare data for regrets
    regret_means = {m: [] for m in methods}
    regret_stds = {m: [] for m in methods}

    for _dim in problem_dims:
        print(f'KP-{_dim}')
        for _method in methods:
            vals = rel_regret_results[_dim][_method]
            regret_means[_method].append(np.mean(vals))
            regret_stds[_method].append(np.std(vals))
            print(f'[{_method}] - Test relative regret: {np.mean(vals)} +- {np.std(vals)}')
        print('\n\n')

    # Plot (two subplots, shared x)
    x = np.arange(len(problem_dims))
    width = 0.8 / len(methods)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Epochs subplot ---
    for i, _method in enumerate(methods):
        ax1.bar(
            x + i * width,
            epoch_means[_method],
            width,
            yerr=epoch_stds[_method],
            label=method_labels.get(_method, _method) if method_labels else _method,
            capsize=4
        )

    ax1.set_ylabel("Number of Epochs")
    # ax1.set_title("Convergence speed (number of epochs)")
    ax1.legend()

    # --- Regret subplot ---
    for i, _method in enumerate(methods):
        ax2.bar(
            x + i * width,
            regret_means[_method],
            width,
            yerr=regret_stds[_method],
            label=method_labels.get(_method, _method) if method_labels else _method,
            capsize=4
        )

    ax2.set_xticks(x + width * (len(methods) - 1) / 2)
    ax2.set_xticklabels(x_labels if x_labels else problem_dims)
    ax2.set_xlabel("Problem Dimension")
    ax2.set_ylabel("Test Relative Regret")
    # ax2.set_title("Test performance (relative regret)")

    plt.tight_layout()
    plt.savefig('epochs-rel-regret-kp-increasing-dimension', dpi=500)

    pass

########################################################################################################################


if __name__ == '__main__':

    problem_dim_str = ['50-items', '75-items', '100-items', '200-items']
    problem_dims = [f'KP-{_dim_str.split('-')[0]}' for _dim_str in problem_dim_str]
    method_names_dict = {
        'MSE': 'PFL',
        'SPO': 'SPO',
        'SFGE+SCE': 'SFGE-MAP'
    }

    plot_num_epochs_wrt_problem_dim(
        exp_dir='experiments/knapsack',
        problem_dims=problem_dim_str,
        ds_seeds=np.arange(5),
        rnd_split_seeds=[0],
        runs=[0],
        methods=['MSE', 'SPO', 'SFGE+SCE'],
        x_labels=problem_dims,
        method_labels=method_names_dict)

    exit()

    # dim = 5
    # plot_std_dev(exp_filepath=f'experiments/knapsack/{dim}-items/', seeds=[0], splits=[0], img_name=f'std-dev-epochs-{dim}-items.png', title=f'KP-{dim}')
    # exit()

    """ for np in range(5):
        for split in range(3):
            poisson = pd.read_csv(
                f'experiments/wsmc/poisson-regressor/10x50/seed-{np}/rnd-split-seed-{split}/MLE/run_0/metrics.csv',
                index_col=0)
            gaussian = pd.read_csv(
                f'experiments/wsmc/10x50/penalty-5/seed-{np}/rnd-split-seed-{split}/MLE/run_0/metrics.csv')
            print(f'Numpy: {np}, split: {split}')
            print(
                f'[Gaussian] - MSE: {gaussian["test_mse"].iloc[-1].item()}, regret: {gaussian["test_relative_regret"].iloc[-1].item()}')
            print(
                f'[Poisson] - MSE: {poisson["test_mse"].iloc[-1].item()}, regret: {poisson["test_relative_regret"].iloc[-1].item()}')
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(poisson.index, poisson['val_relative_regret'], label='regret')
            ax1.legend()
            ax2.plot(poisson.index, poisson['val_mse'], label='mse')
            ax2.legend()
            plt.show()
            print() """

    # features = pd.read_csv('data/data/wsmc/10x50/penalty-5.0/seed-0/features.csv', index_col=0)
    # targets = pd.read_csv('data/data/wsmc/10x50/penalty-5.0/seed-0/targets.csv', index_col=0)
    #
    # # for f_idx in range(features.values.shape[1]):
    # for t_idx in range(targets.values.shape[1]):
    #     # plt.scatter(features.values[:, f_idx], targets.values[:, t_idx])
    #     plt.hist(targets.values[:, t_idx])
    #     plt.show()

    compare_convergence_speed(ds_seeds=np.arange(5),
                              rnd_split_seeds=[0, 1, 2],
                              base_filepaths=['experiments/wsmc/gaussian-regressor/5x25/penalty-5',
                                              'experiments/wsmc/poisson-regressor/5x25/penalty-5'],
                              metric_name='val_relative_regret',
                              method_names=['MLE', 'MLE'],
                              labels=['Gaussian', 'Poisson'])


    """# batch_sizes = [str(2**(i+1)) for i in range(9)]
    # std_dev_vals = ['0.05', '0.25', '1', '1.5', '2']
    
    # prefix = 'experiments/knapsack/std-dev-tuning/50-items/'
    
    # filepaths = [prefix + f'batch-{b}' for b in batch_sizes]
    # filepaths = [prefix + f'std-dev-{v}' for v in std_dev_vals]
    # filepaths.append('experiments/knapsack/50-items/')
    # filepaths = ['experiments/knapsack/50-items/']
    # filepaths.append('experiments/knapsack/hyperparams-tuning/50-items/contextual-std-dev-batch-size-32-lr-0_005')
    filepaths = ['experiments/knapsack/hyperparams-tuning/50-items/contextual-std-dev-batch-size-32-lr-0_005']
    # names = ['MSE' for _ in batch_sizes]
    # names = ['SFGE+SCE' for _ in std_dev_vals]
    # names += ['SFGE+SCE', 'SFGE+SCE']
    names = ['SFGE+SCE']
    # lbls = ['Batch size: ' + b for b in batch_sizes]
    
    hyperparams_tuning(ds_seeds=[0, 1, 2, 3, 4],
                       rnd_split_seeds=[0, 1, 2],
                       base_filepaths=filepaths,
                       metric_name='test_relative_regret',
                       method_names=names,
                       # labels=std_dev_vals + ['Trainable', 'Contextual']
                       labels=['Trainable']
                       )
    
    regret_barplots(res_paths=['experiments/knapsack/hyperparams-tuning/50-items/contextual-std-dev-batch-size-32-lr-0_005',
                               'experiments/knapsack/50-items',
                               'experiments/knapsack/50-items'],
                    names=['predicted std dev', 'trainable std dev', 'SPO'],
                    methods=['SFGE+SCE', 'SFGE+SCE', 'SPO'])"""

    # This is an example on how to generate the source latex code to visualize the results for the KP with stochastic
    # capacity and 50 items

    # columns = ['Rel. PRegret', 'Feas. rel. PRegret', 'Infeas. ratio', 'MSE', 'Epochs']
    columns = ['Rel. PRegret', 'MSE', 'Feas. rel. PRegret', 'Infeas. ratio']
    methods = ['MLE']
    # metrics = ['relative_regret', 'feasible_solutions_relative_regret', 'num_infeasible_solutions', 'mse']
    metrics = ['relative_regret', 'mse', 'feasible_solutions_relative_regret', 'num_infeasible_solutions']
    problem = 'stochastic_weights_kp/gaussian-regressor'
    # problem = 'knapsack/lr-tuning'
    problem_dim = '50-items/penalty-5'
    # problem = 'wsmc'
    # problem_dim = '5x25'
    seeds = np.arange(5)
    splits = np.arange(3)
    runs = np.arange(1)

    # write_latex_table(column_names=columns,
    #                   method_names=methods,
    #                   metric_names=metrics,
    #                   problem_name=problem,
    #                   problem_dim=problem_dim,
    #                   seeds=seeds,
    #                   splits=splits,
    #                   runs=runs,
    #                   filename=problem + '.tex',
    #                   penalties=[5]
    #                   )
    #
    # exit()

    # ------------------------------------------------------------------------------------------------------------------

    # This is an example of visualization of SAA with predict-then-optimize and SFGE on the WSMC of size 10x50 and
    # penalty value of 1 and 10

    # problem = 'stochastic_weights_kp'
    problem = 'wsmc'
    # dim = '50-items'
    dim = '5x25'
    # res_filepath_prefix = os.path.join('experiments', problem, 'gaussian-regressor', dim)
    img_name = f'sfge-vs-pfl+saa-{problem}-{dim}'

    fig, axises = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 7))
    plt.subplots_adjust(hspace=0, wspace=0)

    """plot_saa_metric_wrt_scenarios(axises[0][0],
                                  res_dir=os.path.join(res_filepath_prefix, 'penalty-1'),
                                  metric_name='Relative regret',
                                  title=f'Penalty 1\n\nRelative post-hoc regret')
    plot_saa_metric_wrt_scenarios(axises[1][0],
                                  res_dir=os.path.join(res_filepath_prefix, 'penalty-1'),
                                  metric_name='Runtime',
                                  title='(Log) normalized runtime',
                                  y_axis_unit=None,
                                  normalize=True,
                                  plot_log=True)"""

    axises[0][0].set_title('Gaussian')
    plot_saa_metric_wrt_scenarios(axises[0][0],
                                  res_dir=os.path.join('experiments', problem, 'gaussian-regressor', dim, 'penalty-5'),
                                  metric_name='Relative regret',
                                  title='Gaussian',
                                  plot_sfge=True)
    plot_saa_metric_wrt_scenarios(axises[1][0],
                                  res_dir=os.path.join('experiments', problem, 'gaussian-regressor', dim, 'penalty-5'),
                                  metric_name='Runtime',
                                  title=None,
                                  y_axis_unit=None,
                                  normalize=True,
                                  plot_log=True,
                                  plot_sfge=True)

    axises[0][1].set_title('Poisson')
    plot_saa_metric_wrt_scenarios(axises[0][1],
                                  res_dir=os.path.join('experiments', problem, 'poisson-regressor', dim, 'penalty-5'),
                                  metric_name='Relative regret',
                                  title=None)
    plot_saa_metric_wrt_scenarios(axises[1][1],
                                  res_dir=os.path.join('experiments', problem, 'poisson-regressor', dim, 'penalty-5'),
                                  metric_name='Runtime',
                                  title='Poisson',
                                  y_axis_unit=None,
                                  normalize=True,
                                  plot_log=True)

    axises[1][0].set_xlabel('Num. scenarios', fontsize=14)
    axises[1][1].set_xlabel('Num. scenarios', fontsize=14)
    #axises[1][2].set_xlabel('Num. scenarios', fontsize=12)
    axises[0][0].set_ylabel('Rel. post-hoc regret', fontsize=14, rotation=90)
    axises[1][0].set_ylabel('(Log) norm. runtime', fontsize=14, rotation=90)
    axises[0][0].tick_params(axis='both', which='major', labelsize=12)
    axises[1][0].tick_params(axis='both', which='major', labelsize=12)
    axises[0][0].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    axises[1][0].yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    plt.tight_layout()
    plt.show()
    # plt.savefig(img_name, dpi=500)"""

    # print_from_pickle(filepath_prefix='experiments/knapsack/75-items/', method='SFGE', seeds=[0, 1, 2], splits=[0, 1, 2], metric='test_relative_regret')
    # print_from_pickle(filepath_prefix='experiments/knapsack/75-items/', method='SFGE', seeds=[0, 1, 2], splits=[0, 1, 2], metric='test_mse')
    # print_num_epochs(filepath_prefix='experiments/knapsack/75-items/', method='SFGE', seeds=[0, 1, 2], splits=[0, 1, 2], runs=[0])