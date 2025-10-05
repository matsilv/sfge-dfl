"""
    Create the instances of the different DFL approaches.
"""

from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.random import manual_seed

from methods.dpo import DPOModule
from methods.mse import MSEModule
from methods.spo import SPOModule
from methods.blackbox import BlackboxModule
from methods.sfge import SFGEModule
from methods.prob_regressor import ProbRegressorModule
from methods.sce import SCEModule
from utils.enums import DistributionTypes
from utils.scale_layer import ScaleLayer, MockScaleLayer
from utils.probabilistic_models import MultivariateGaussianModule, PoissonModule
from utils.annealers import LinearAnnealer
from dfl_module import DFLModule
from optimization_problems.optimization_problem import OptimizationProblem

from typing import Dict, Tuple

########################################################################################################################

METHODS = {'MSE': MSEModule,
           'MLE': ProbRegressorModule,
           'SPO': SPOModule,
           'Blackbox': BlackboxModule,
           'SFGE': SFGEModule,
           'SFGE+SCE': SFGEModule,
           'SCE': SCEModule,
           'DPO': DPOModule}

# Methods that rely on a probabilistic model
PROB_METHODS = ['SFGE', 'SFGE+SCE', 'MLE']

########################################################################################################################


# FIXME: too many if/else statements
def build_models(config: Dict,
                 train_dl: DataLoader,
                 optimization_problem: OptimizationProblem) -> Dict:
    """
    For each training method (e.g. SPO, MSE, etc...), create a predictive model.
    :param config: dict; dictionary with configuration parameters.
    :param optimization_problem: optimization_problems.optimization_problem.OptimizationProblem;
                                 the optimization problem instance.
    :param train_dl: torch.utils.data.DataLoader; PyTorch data loader.
    :return: dict; one model for each method and run.
    """

    # Sanity check
    assert 'method_configurations' in config.keys()
    assert 'num_runs_per_seed' in config.keys()
    assert 'output_dim' in config.keys()

    # Dictionary with one model for each method and run
    models = dict()

    # Some useful configuration parameters
    method_configurations = config['method_configurations']
    output_shape = config['output_dim']
    input_shape = config['input_dim']

    # For each method...
    for method_config in method_configurations:

        # Set PyTorch random seed
        manual_seed(config['torch_seed'])

        # Get the name of the method
        method_name = method_config['name']

        if method_name in PROB_METHODS:
            prob_model = True
        else:
            prob_model = False

        # Create the neural model and the PyTorch scaling layer
        net, scale_layer = build_net(config, train_dl)

        # Create a copy of the 'method_config' dictionary that will be used as kwargs to the 'DFLModule' constructor
        model_args = method_config.copy()

        # Build a probabilistic model
        if prob_model:

            distr_type = method_config['distr_type']

            match distr_type:
                case DistributionTypes.gaussian:

                    init_std_dev = method_config['init_log_std_dev']
                    covariance_type = method_config['covariance_type']

                    if covariance_type == 'linear_annealing':
                        # Sanity check: the end of the annealing must be defined
                        assert 'annealing_epochs' in config.keys()
                        # Sanity check: the lower bound of the annealing factor must be defined
                        assert 'min_annealing_val' in config.keys()

                        annealer = \
                            LinearAnnealer(tot_epochs=config['annealing_epochs'],
                                           min_anneal_val=config['min_annealing_val'])
                    else:
                        annealer = None

                    # These keys are only useful to build the probabilistic model but they are not argument of the
                    # 'DFLModule' class
                    del model_args['init_log_std_dev']
                    del model_args['covariance_type']

                    model_args['annealer'] = annealer

                    nn_model = \
                        MultivariateGaussianModule(net=net,
                                                   scale_layer=scale_layer,
                                                   input_shape=input_shape,
                                                   output_shape=output_shape,
                                                   init_std_dev_val=init_std_dev,
                                                   covariance_type=covariance_type)

                case DistributionTypes.poisson:
                    nn_model = PoissonModule(net=net, scale_layer=scale_layer)

                case _:
                    raise ValueError(f'Illegal distribution type {distr_type}')

            del model_args['distr_type']
        # Build a point estimator
        else:
            nn_model = nn.Sequential(net, scale_layer)

            model_args['annealer'] = None

        # Build the DFL model
        model, model_name = build_model(method_config=model_args,
                                        net=nn_model,
                                        optimization_problem_to_use=optimization_problem)
        models[model_name] = model

    return models

########################################################################################################################


def build_model(method_config: Dict,
                net: Module,
                optimization_problem_to_use: OptimizationProblem) -> Tuple[DFLModule, str]:
    """
    Create the DFL model.
    :param method_config: dict; dictionary with the method configuration.
    :param net: torch.nn.Module; PyTorch module that encapsulate the neural model.
    :param optimization_problem_to_use: optimization_problems.optimization_problem.OptimizationProblem:
           the optimization problem instance.
    :return: tuple with a DFL model and a string; the DFL model and its name.
    """

    # Get the name of the DFL method
    name = method_config['name']

    # Get the arguments for the DFL method class
    args = method_config.copy()
    del args['name']

    if name == 'SFGE+SCE':
        args['sce_loss'] = True

    # Get the DFL model class
    model_class = METHODS[name]

    # Create the DFL model object
    model = model_class(net=net, optimization_problem=optimization_problem_to_use,  **args)

    return model, name

########################################################################################################################


# FIXME: can we find a way to reduce the if-else statement length?
def build_net(config: Dict,
              train_dl: DataLoader) -> Tuple[Module, str]:
    """
    Create the predictive model.
    :param config: dict; configuration parameters.
    :param train_dl: torch.utils.data.DataLoader; the PyTorch data loader of the training set.
    :return:
    """

    output_dim = config['output_dim']

    # If the user wants to standardize back the predicted features
    if "scale_predictions" in config:

        # Get the scale configurations
        scale_config = config["scale_predictions"]

        # Optionally, the user can set the initial value of the translation term
        if "init_translation_term" in scale_config:
            init_translation_term = scale_config["init_translation_term"]
        else:
            # Use the mean target value in the training data as (initial) translation term
            # FIXME: only per-feature scaling is supported
            init_translation_term = train_dl.dataset.y.mean(axis=0)

        # Optionally, the user can set the initial value of the multiplication term
        if "init_multiplication_factor" in scale_config:
            init_multiplication_factor = scale_config["init_multiplication_factor"]
        else:
            # Use the standard deviation of the target value in the training data as
            # (initial) multiplication factor
            # FIXME: only per-feature scaling is supported
            init_multiplication_factor = train_dl.dataset.y.std(axis=0)

        # Optionally, the user can set the standardization term trainable
        scale_layer = ScaleLayer(translation_term=init_translation_term,
                                 multiplication_factor=init_multiplication_factor)
    else:
        scale_layer = MockScaleLayer()

    # Build the predictive model (simple linear layer)
    net = nn.Linear(config['input_dim'], output_dim)

    return net, scale_layer
