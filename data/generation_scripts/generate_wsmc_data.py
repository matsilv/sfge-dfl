import numpy as np
import inspect
import pickle
import shutil

import torch
from tqdm import tqdm
import pandas as pd
import json
import os
import random
import math
import matplotlib.pyplot as plt
import argparse

from data.generation_scripts.generate_knapsack_data import bernoulli, set_seeds
from data.generation_scripts import DATAPATH_PREFIX
from optimization_problems.set_cover_problem import StochasticWeightedSetMultiCover

from typing import List

########################################################################################################################


# FIXME: add description on how the availability matrix is generated
def generate_availability(num_sets: int,
                          num_products: int,
                          density: int) -> np.ndarray:
    """
    Generate the availability of each product in each set.
    :param num_sets: int; the number of sets (molds).
    :param num_products: int; the number of products (items).
    :param density:
    :return: numpy.array of shape (num_products, num_sets); 0-1 matrix for each product-set pair.
    """

    # Sanity checks
    assert isinstance(num_sets, int)
    assert isinstance(num_products, int)
    assert 0 < density < 1, "Density must be in ]0,1["

    # Initialize the availability matrix
    availability = np.zeros(shape=(num_products, num_sets), dtype=np.int8)

    # For each product (item)...
    for row in range(num_products):
        first_col = -1
        second_col = -1

        while first_col == second_col:
            first_col = np.random.randint(low=0, high=num_sets, size=1)
            second_col = np.random.randint(low=0, high=num_sets, size=1)
            availability[row, first_col] = 1
            availability[row, second_col] = 1

    for col in range(num_sets):
        row = np.random.randint(low=0, high=num_products, size=1)
        availability[row, col] = 1

    # Check that all the products are available in at least two sets
    available_products = np.sum(availability, axis=1) > 1

    # Check that all the sets have at least one product
    at_least_a_prod = np.sum(availability, axis=0) > 0

    density = np.clip(density - np.mean(availability), a_min=0, a_max=None)
    availability += np.random.choice([0, 1], size=(num_products, num_sets), p=[1 - density, density])
    availability = np.clip(availability, a_min=0, a_max=1)

    # print(f'[MinSetCover] - True density: {np.mean(availability)}')

    assert available_products.all(), "Not all the products are available"
    assert at_least_a_prod.all(), "Not all set cover at least a product"

    return availability

########################################################################################################################


def set_prod_cost(set_costs: np.ndarray,
                  num_products: int,
                  availability: np.ndarray,
                  penalty_factor: float) -> np.ndarray:
    """
    The product costs are set according to the min cost of among the sets that cover them multiplied by 10.
    :param set_costs: numpy.ndarray; the cost of each set (mold).
    :param num_products: int; the number of products (items).
    :param penalty_factor: float; this is the coefficient that (together with the sets cost) define the penalty term.
    :return: numpy.ndarray; the products costs.
    """

    assert isinstance(num_products, int), "set_costs must be integer"

    prod_costs = np.zeros(shape=(num_products, ))

    for idx in range(num_products):
        prod_availability = availability[idx]

        # First of all, we check the costs of the set that cover the current product (possible_prod_cost)
        possible_prod_cost = prod_availability * set_costs
        possible_prod_cost = possible_prod_cost[np.nonzero(possible_prod_cost)]

        # Then we choose the max among them
        max_cost = np.max(possible_prod_cost)
        prod_costs[idx] = max_cost * penalty_factor

    return prod_costs

########################################################################################################################


class MinSetCover:
    """
    Minimum Set Cover class.
    """

    def __init__(self,
                 num_sets: int,
                 num_products: int,
                 density: float,
                 availability: np.ndarray = None,
                 demands: np.ndarray = None,
                 set_costs: np.ndarray = None,
                 prod_costs: np.ndarray = None,
                 observables: np.ndarray = None,
                 lmbds: np.ndarray = None):

        self._num_sets = num_sets
        self._num_products = num_products
        self._density = density
        self._observables = observables
        self._lmbds = lmbds

        # You can choose to give some MSC parameters as input or create them from scratch

        if set_costs is None:
            # Uniform random generation of the costs in the interval [1, 100]
            self._set_costs = np.random.randint(low=1, high=100, size=num_sets)
        else:
            self._set_costs = set_costs

        assert demands is not None, "demands must be initialized"
        self._demands = demands

        if availability is None:
            self._availability = generate_availability()
        else:
            self._availability = availability

        if prod_costs is None:
            set_prod_cost()
        else:
            self._prod_costs = prod_costs

    @property
    def num_sets(self):
        return self._num_sets

    @num_sets.setter
    def num_sets(self, value):
        self._num_sets = value

    @property
    def num_products(self):
        return self._num_products

    @num_products.setter
    def num_products(self, value):
        self._num_products = value

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        self._density = value

    @property
    def set_costs(self):
        return self._set_costs

    @set_costs.setter
    def set_costs(self, value):
        self._set_costs = value

    @property
    def prod_costs(self):
        return self._prod_costs

    @prod_costs.setter
    def prod_costs(self, value):
        self._prod_costs = value

    @property
    def availability(self):
        return self._availability

    @availability.setter
    def availability(self, value):
        self._availability = value

    @property
    def demands(self):
        return self._demands

    @demands.setter
    def demands(self, value):
        self._demands = value

    @property
    def observables(self):
        return self._observables

    @observables.setter
    def observables(self, value):
        self._observables = value

    @property
    def lmbds(self):
        return self._lmbds

    @lmbds.setter
    def lmbds(self, value):
        self._lmbds = value

    def new(*args):
        raise NotImplementedError()

    def dump(self, filepath: str):
        """
        Save the MSC instance in a pickle.
        :param filepath: str; where the instance is saved to.
        :return:
        """
        msc_dict = dict()

        # We save all the readable properties
        for member_name, member_value in inspect.getmembers(self):
            if not member_name.startswith('_') and not inspect.ismethod(member_value):
                msc_dict[member_name] = member_value

        with open(filepath, 'wb') as file:
            pickle.dump(msc_dict, file)


########################################################################################################################


def get_datapath(prefix: str,
                 num_prods: int,
                 num_sets: int,
                 penalty_factor: float,
                 seed: int) -> str:
    """
    Get the full path where WSMC data are saved to.
    :param prefix: str; where data of all the problems are saved to.
    :param num_prods: int; number of products of the WSMC.
    :param num_sets: int; number of sets of the WSMC.
    :param penalty_factor: float; the coefficient used to compute the penalty.
    :param seed: int; the numpy random seed.
    :return:
    """

    data_path = os.path.join(prefix,
                             'wsmc',
                             f'{num_prods}x{num_sets}',
                             f'penalty-{penalty_factor}',
                             f'seed-{seed}')

    return data_path

########################################################################################################################


def generate_wsmc_instances(input_dim: int,
                            deg: int,
                            multiplicative_noise: float,
                            additive_noise: float,
                            num_instances: int,
                            num_sets: int,
                            num_products: int,
                            density: float,
                            penalty_factor: float) -> List[MinSetCover]:
    """
    Generate a random set of Weighted Set Multi-cover instances.
    :param input_dim: int; the number of input features.
    :param deg: int; the degree of the polynomial relationship between the input features and the Poisson rates.
    :param multiplicative_noise: float; the multiplicative noise of the input-Poisson rate relationship.
    :param additive_noise: float; the multiplicative noise of the input-Poisson rate relationship.
    :param num_instances: int; the number of instances to be generated.
    :param num_sets: int; number of sets of the MSC.
    :param num_products: int; number of products (elements) of the MSC.
    :param density: float; the density of the availability matrix.
    :param penalty_factor: float; this is the coefficient that (together with the sets cost) define the penalty term.
    :return:
    """

    # Uniform random generation of the costs in the interval [1, 100]
    set_costs = np.random.randint(low=1, high=100, size=num_sets)
    # Generate the availability matrix
    availability = generate_availability(num_sets=num_sets,
                                         num_products=num_products,
                                         density=density)
    # Compute the product costs based on the set costs
    prod_costs = set_prod_cost(set_costs=set_costs,
                               num_products=num_products,
                               availability=availability,
                               penalty_factor=penalty_factor)

    # Create one model for the entire instance
    B = np.array([[bernoulli(0.5) for _ in range(input_dim)] for _ in range(num_products)])

    # Keep track of all the instances
    instances = list()

    # For each instance...
    for _ in tqdm(range(num_instances), total=num_instances, desc='Generating instances'):

        # Keep track of Poisson rate for each product
        lmbd_list = list()

        # The input features are random values in [0, 1] generated according to a Gaussian distributions
        x = np.array([round(random.gauss(0, 1), 3) for _ in range(input_dim)])

        # Only a subset of input features affects the targets
        B_matmul_x = np.matmul(B, x)

        # For each target dimension...
        for j in range(num_products):

            # Generate the true model
            pred = B_matmul_x[j]

            # Noisy polinomial relationship
            lmbd_val = (1 + (pred / math.sqrt(input_dim) + 3) ** deg) * random.uniform(1 - multiplicative_noise,
                                                                                       1 + multiplicative_noise)
            lmbd_val = round(lmbd_val + additive_noise, 5)

            lmbd_list.append(lmbd_val)

        # Only the demands change among the instances
        demands = np.random.poisson(lmbd_list, size=num_products)

        inst = MinSetCover(num_sets=num_sets,
                           num_products=num_products,
                           availability=availability,
                           density=density,
                           demands=demands,
                           set_costs=set_costs,
                           prod_costs=prod_costs,
                           observables=x,
                           lmbds=lmbd_list)

        instances.append(inst)

    return instances

########################################################################################################################


def generate_dataset(data_path: str,
                     num_instances: int,
                     num_sets: int,
                     num_prods: int,
                     density: float,
                     input_dim: int,
                     deg: int,
                     multiplicative_noise: float,
                     additive_noise: float,
                     penalty_factor: float,
                     seed: int,
                     plot: bool = False):
    """
    Generate and save on a file MSC instances.
    :param data_path: string; where instances are saved to.
    :param num_instances: int; number of instances to be generated.
    :param num_sets: int; number of sets of the MSC.
    :param num_prods: int; number of products of the MSC.
    :param density: float; density of the availability matrix.
    :param input_dim: int; the number of input features.
    :param deg: int; the degree of the polynomial relationship between the input features and the Poisson rates.
    :param multiplicative_noise: float; the multiplicative noise of the input-Poisson rate relationship.
    :param additive_noise: float; the multiplicative noise of the input-Poisson rate relationship.
    :param penalty_factor: float; this is the coefficient that (together with the sets cost) define the penalty term.
    :param seed: int; random seed used to generate data.
    :param plot; bool; if True, plot the relationship between input features and demands.
    :return:
    """

    # This is the datapath where data are saved to
    data_path = get_datapath(prefix=data_path,
                             num_prods=num_prods,
                             num_sets=num_sets,
                             penalty_factor=penalty_factor,
                             seed=seed)

    # Generate the observables and the MSC instances
    instances = \
        generate_wsmc_instances(input_dim=input_dim,
                                deg=deg,
                                multiplicative_noise=multiplicative_noise,
                                additive_noise=additive_noise,
                                num_instances=num_instances,
                                num_sets=num_sets,
                                num_products=num_prods,
                                density=density,
                                penalty_factor=penalty_factor)

    # Create the data folder
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    else:
        shutil.rmtree(data_path)
        os.makedirs(data_path)

    # Save the observables as a pandas.Dataframe
    observables = [inst.observables for inst in instances]
    observables = np.asarray(observables)
    columns = [f'at_{idx+1}' for idx in range(input_dim)]
    features = pd.DataFrame(data=observables, columns=columns)
    features.to_csv(os.path.join(data_path, 'features.csv'))

    # Save the demands a pandas.Dataframe
    index = pd.Index(data=np.arange(num_instances), name='index')
    targets = [inst.demands for inst in instances]
    targets = np.asarray(targets)
    targets = pd.DataFrame(index=index,
                           data=targets,
                           columns=[f'd_{idx}' for idx in range(num_prods)])
    targets.to_csv(os.path.join(data_path, 'targets.csv'))

    # Save the optimal solutions as a pandas.Dataframe
    solutions = pd.DataFrame(index=index,
                             columns=[f'x_{idx}' for idx in range(num_sets)])

    # If required, plot the relationship between each pair input-target features
    if plot:
        fig, axis = plt.subplots(nrows=input_dim,
                                 ncols=num_prods,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(10, 10))
        plt.subplots_adjust(wspace=0, hspace=0)

        for i, at in enumerate(features.columns):
            for j in range(targets.shape[1]):
                axis[i, j].scatter(features[at], targets.values[:, j])

        fig.supxlabel('Input features')
        fig.supylabel('Demands')
        plt.show()

    # Save the shared optimization model parameters
    wsmc_dim = {'num_sets': num_sets, 'num_prods': num_prods}

    with open(os.path.join(data_path, 'wsmc_dim.json'), 'w') as file:
        json.dump(wsmc_dim, file)

    # Check that the availability matrix, the sets and products costs are the same among all the instances
    availability = instances[0].availability
    set_costs = instances[0].set_costs
    prod_costs = instances[0].prod_costs

    opt_problem = StochasticWeightedSetMultiCover(num_sets=num_sets, num_products=num_prods)
    opt_prob_params = {'set_costs': torch.as_tensor(set_costs),
                       'prod_costs': torch.as_tensor(prod_costs),
                       'availability': torch.as_tensor(availability)}

    for idx, inst in tqdm(enumerate(instances), total=len(instances), desc='Solving instances'):
        assert np.array_equal(availability, inst.availability), \
            "The availability matrix must be the same for all the instances"

        assert np.array_equal(set_costs, inst.set_costs), \
            "The sets costs must be the same for all the instances"

        assert np.array_equal(prod_costs, inst.prod_costs), \
            "The products costs must be the same for all the instances"

        sol, _ = opt_problem.solve(y=inst.demands, opt_prob_params=opt_prob_params)
        solutions.iloc[idx] = sol

    np.save(os.path.join(data_path, 'availability.npy'), availability)
    np.save(os.path.join(data_path, 'set_costs.npy'), set_costs)
    np.save(os.path.join(data_path, 'prod_costs.npy'), prod_costs)
    solutions.to_csv(os.path.join(data_path, 'solutions.csv'))


########################################################################################################################

# FIXME: add argument parses
if __name__ == '__main__':
    # Script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dim", type=int, required=True, help="Number of input features")
    parser.add_argument("--output-dim", type=int, required=True, help="Number of output features")
    parser.add_argument("--num-sets", type=int, required=True,
                        help="The number of sets in the WSMC")
    parser.add_argument("--num-products", type=int, required=True,
                        help="The number of products in the WSMC")
    parser.add_argument("--density", type=float, default=0.02,
                        help="The density of the availability matrix")
    parser.add_argument("--penalty", type=float, required=True,
                        help="The penalty coefficient of the recourse action")
    parser.add_argument("--degree", type=int, required=True, help="The degree of the polynomial function")
    parser.add_argument("--num-instances", type=int, required=True, help="The number of instances to generate")
    parser.add_argument("--multiplicative-noise",
                        type=float,
                        default=0,
                        help="The multiplicative noise added to the predictions")
    parser.add_argument("--additive-noise",
                        type=float,
                        default=0,
                        help="The additive noise added to the predictions")
    parser.add_argument("--seeds", type=int, nargs='+', required=True, help="Numpy random seeds")

    # Parse the arguments
    args = parser.parse_args()
    num_instances = args.num_instances
    num_sets = args.num_sets
    num_prods = args.num_products
    density = args.density
    input_dim = args.input_dim
    deg = args.degree
    mult_noise = args.multiplicative_noise
    add_noise = args.additive_noise
    penalty = args.penalty
    seeds = args.seeds

    for seed in seeds:
        set_seeds(seed)

        generate_dataset(data_path=DATAPATH_PREFIX,
                         num_instances=num_instances,
                         num_sets=num_sets,
                         num_prods=num_prods,
                         density=density,
                         input_dim=input_dim,
                         deg=deg,
                         multiplicative_noise=mult_noise,
                         additive_noise=add_noise,
                         penalty_factor=penalty,
                         seed=seed)
