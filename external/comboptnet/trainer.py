import os.path
from abc import ABC, abstractmethod

import torch
import numpy as np

from models.models import KnapsackExtractWeightsCostFromEmbeddingMLP, baseline_mlp_dict
from models.models import KnapsackExtractWeightsFromFeatures, SetCoverExtractDemandsFromFeatures
from models.models import KnapsackExtractCapacityFromFeatures
from models.modules import get_solver_module, StaticConstraintModule, CvxpyModule, CombOptNetModule
from utils.utils import loss_from_string, optimizer_from_string, set_seed, AvgMeters, compute_metrics, \
    knapsack_round, compute_normalized_solution, compute_denormalized_solution, solve_unconstrained
from utils import *
from optimization_problems.stochastic_weights_kp import StochasticWeightsKnapsackProblem
from optimization_problems.stochastic_capacity_kp import StochasticCapacityKnapsackProblem
from optimization_problems import TOTAL_COST, PENALTY_COST, SUBOPTIMALITY_COST

from typing import Dict, List, Tuple

########################################################################################################################


def get_trainer(trainer_name, **trainer_params):
    trainer_dict = dict(MLPTrainer=MLPBaselineTrainer,
                        KnapsackConstraintLearningTrainer=KnapsackConstraintLearningTrainer,
                        RandomConstraintLearningTrainer=RandomConstraintLearningTrainer,
                        KnapsackWeightsLearningTrainer=KnapsackWeightsLearningTrainer,
                        KnapsackCapacityLearningTrainer=KnapsackCapacityLearningTrainer,
                        SetCoverDemandsLearningTrainer=SetCoverDemandsLearningTrainer)
    return trainer_dict[trainer_name](**trainer_params)


class BaseTrainer(ABC):
    def __init__(self, train_iterator, val_iterator, test_iterator, use_cuda, optimizer_name, loss_name, optimizer_params, metadata,
                 model_params, seed, penalty=None):

        self._penalty = penalty
        set_seed(seed)
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'

        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.val_iterator = val_iterator

        self.true_variable_range = metadata['variable_range']
        self.num_variables = metadata['num_variables']
        self.variable_range = self.true_variable_range

        model_params['metadata'] = metadata

        model_parameters = self.build_model(**model_params)
        self.optimizer = optimizer_from_string(optimizer_name)(model_parameters, **optimizer_params)
        self.loss_fn = loss_from_string(loss_name)

    @abstractmethod
    def build_model(self, **model_params):
        pass

    @abstractmethod
    def calculate_loss_metrics(self, **data_params):
        pass

    def train_epoch(self):
        self.train = True
        metrics = AvgMeters()

        for i, data in enumerate(self.train_iterator):
            x, y_true_norm = [dat.to(self.device) for dat in data]
            loss, metric_dct = self.calculate_loss_metrics(x=x, y_true_norm=y_true_norm)
            metrics.update(metric_dct, n=x.size(0))

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        results = metrics.get_averages(prefix='train_')
        return results

    def evaluate(self, test):
        self.train = False
        metrics = AvgMeters()

        if not test:
            iterator = self.val_iterator
            prefix = 'val_'
        else:
            iterator = self.test_iterator
            prefix = 'test_'

        for i, data in enumerate(iterator):
            x, y_true_norm = [dat.to(self.device) for dat in data]
            loss, metric_dct = self.calculate_loss_metrics(x=x, y_true_norm=y_true_norm)
            metrics.update(metric_dct, n=x.size(0))

        results = metrics.get_averages(prefix=prefix)
        return results


# FIXME: not tested
class MLPBaselineTrainer(BaseTrainer):
    def build_model(self, model_name, **model_params):
        self.model = baseline_mlp_dict[model_name](num_variables=self.num_variables, **model_params).to(
            self.device)
        return self.model.parameters()

    def calculate_loss_metrics(self, x, y_true_norm):
        y_norm = self.model(x=x)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_denorm = compute_denormalized_solution(y_norm, **self.variable_range)
        y_denorm_rounded = torch.round(y_denorm)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_rounded, y_true=y_true_denorm))
        return loss, metrics


# FIXME: not tested
class ConstraintLearningTrainerBase(BaseTrainer, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    def calculate_loss_metrics(self, x, y_true_norm):
        y_denorm, y_denorm_roudned, solutions_denorm_dict, cost_vector = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())
        y_uncon_denorm = solve_unconstrained(cost_vector=cost_vector, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))
        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))
        return loss, metrics


# FIXME: not tested
class RandomConstraintLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, constraint_module_params, solver_module_params):
        self.static_constraint_module = StaticConstraintModule(variable_range=self.variable_range,
                                                               num_variables=self.num_variables,
                                                               **constraint_module_params).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        self.ilp_solver_module = CombOptNetModule(variable_range=self.variable_range).to(self.device)
        model_parameters = list(self.static_constraint_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):
        cost_vector = x
        cost_vector = cost_vector / torch.norm(cost_vector, p=2, dim=-1, keepdim=True)
        constraints = self.static_constraint_module()

        y_denorm = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        y_denorm_rounded = torch.round(y_denorm)
        solutions_dict = {}

        if not self.train and isinstance(self.solver_module, CvxpyModule):
            y_denorm_ilp = self.ilp_solver_module(cost_vector=cost_vector, constraints=constraints)
            update_dict = dict(ilp_postprocess=y_denorm_ilp)
            solutions_dict.update(update_dict)

        return y_denorm, y_denorm_rounded, solutions_dict, cost_vector


# FIXME: not tested
class KnapsackConstraintLearningTrainer(ConstraintLearningTrainerBase):
    def build_model(self, solver_module_params, backbone_module_params, metadata):
        self.backbone_module = KnapsackExtractWeightsCostFromEmbeddingMLP(**backbone_module_params).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)
        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())
        return model_parameters

    def forward(self, x):
        cost_vector, constraints = self.backbone_module(x)
        cost_vector = cost_vector / torch.norm(cost_vector, p=2, dim=-1, keepdim=True)

        y_denorm = self.solver_module(cost_vector=cost_vector, constraints=constraints)
        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm
        return y_denorm, y_denorm_rounded, {}, cost_vector


########################################################################################################################

class KnapsackWeightsLearningTrainer(ConstraintLearningTrainerBase):
    """
    Manager of the training routine for the KP with unknown item weights.
    """

    # FIXME: repeated code
    def build_model(self,
                    solver_module_params: Dict,
                    backbone_module_params: Dict,
                    metadata: Dict) -> List[torch.Tensor]:
        """
        Instantiate the backbone and the solver modules.
        :param solver_module_params: dict; a subset of the solver module constructor.
        :param backbone_module_params: dict; a subset of the backbone module constructor.
        :param metadata: dict; additional metadata of the optimization problem to solve.
        :return: list of torch.Tensor; the parameters of the resulting architecture.
        """

        # Where the backbone and solver models will be saved, inside the working directory.
        self._backbone_path = 'backbone'
        self._solver_path = 'solver'

        self._in_features = backbone_module_params['in_features']
        self._kp_dim = backbone_module_params['out_features']

        # Whether the problem is a minimization or maximation problem
        self._opt_prob_dir = metadata['opt_prob_dir']

        self.backbone_module = \
            KnapsackExtractWeightsFromFeatures(kp_dim=self._kp_dim,
                                               input_dim=backbone_module_params['in_features'],
                                               knapsack_capacity=metadata['capacity'],
                                               weight_min=metadata['min_weight'],
                                               weight_max=metadata['max_weight'],
                                               out_features=backbone_module_params['out_features']).to(self.device)

        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)

        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())

        return model_parameters

    def save(self, filepath: str):
        """
        Save the models on a file. Backbone and solver will be saved in separated folders.
        :param filepath: str; the base savepath.
        :return:
        """

        backbone_full_path = os.path.join(filepath, self._backbone_path)
        solver_full_path = os.path.join(filepath, self._solver_path)

        torch.save(self.backbone_module.state_dict(), backbone_full_path)
        torch.save(self.solver_module.state_dict(), solver_full_path)

    def load(self, filepath: str):
        """
        Load the models from file.
        :param filepath: str; the base loadpath.
        :return:

        """
        backbone_full_path = os.path.join(filepath, self._backbone_path)
        solver_full_path = os.path.join(filepath, self._solver_path)

        self.backbone_module.load_state_dict(torch.load(backbone_full_path))
        self.solver_module.load_state_dict(torch.load(solver_full_path))

    def forward(self, x: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given are the input features and the optimization problem parameters, compute the solution.
        :param x: torch.Tensor; the input features.
        :return: the predicted solution, the predicted solution rounded to the closest integer, an empty dict (why?),
                 the learnt constraints, and the true cost vector, weights and capacity
        """

        # FIXME: this should not be hardcoded
        # Sanity check: the input features, the cost vector, the item weights and the capacity are concatenated in a
        # single tensor; check whether the dimension is the expected one
        assert x.shape[1] == self._in_features + (self._kp_dim * 2) + 1

        # Unpack the input features and the optimization problem parameters
        in_features = x[:, :self._in_features]
        cost_vector = x[:, self._in_features:self._in_features+self._kp_dim]
        weights = x[:, self._in_features+self._kp_dim:-1]
        capacity = x[:, -1]

        # Predict the constraints matrix
        constraints = self.backbone_module(in_features)

        y_denorm = self.solver_module(cost_vector=-cost_vector, constraints=constraints)

        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm

        # FIXME: too many return values and why and empty dict?
        return y_denorm, y_denorm_rounded, {}, constraints, cost_vector, weights, capacity

    # FIXME: repeated code
    def calculate_loss_metrics(self, x: torch.Tensor, y_true_norm: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the loss function and the evaluation metrics.
        :param x: torch.Tensor; the input features and the optimization problem parameters.
        :param y_true_norm: torch.Tensor; the true normalized optimal solution.
        :return: torch.Tensor, dict; the loss value and the evaluation metrics.
        """

        y_denorm, y_denorm_roudned, solutions_denorm_dict, constraints, cost_vector, weights, capacity = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())

        y_uncon_denorm = solve_unconstrained(cost_vector=cost_vector, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))

        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))

        # Compute the evaluation metrics
        y = y_denorm.cpu().detach().numpy()
        y_true = y_true_denorm.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        values = cost_vector.cpu().detach().numpy()
        capacity = capacity.cpu().detach().numpy()

        rel_regret_list = list()
        feas_sol_rel_regret_list = list()
        regret_list = list()
        sol_mse_list = list()
        weights_mse_list = list()
        cost_list = list()
        opt_cost_list = list()
        num_infeas_sol = 0

        # FIXME: unpacking is too long
        for y_i, y_true_i, weights_i, values_i, capacity_i, constraints_i in zip(y, y_true, weights, values, capacity, constraints):

            # Compute the post-hoc regret given the predicted solution \hat{z} (y_i)
            opt_prob = StochasticWeightsKnapsackProblem(dim=self._kp_dim, penalty=self._penalty)
            opt_prob_params = {'values': torch.as_tensor(values_i),
                               'capacity': torch.as_tensor(capacity_i)}

            pred_sol_cost, feasible = \
                opt_prob.get_objective_values(y=torch.as_tensor(weights_i),
                                              sols=torch.as_tensor(y_i),
                                              opt_prob_params=opt_prob_params)

            opt_sol_cost, _ = \
                opt_prob.get_objective_values(y=torch.as_tensor(weights_i),
                                              sols=torch.as_tensor(y_true_i),
                                              opt_prob_params=opt_prob_params)

            post_hoc_regret = pred_sol_cost[TOTAL_COST] - opt_sol_cost[TOTAL_COST]
            if not opt_prob.is_minimization_problem:
                post_hoc_regret *= -1
            # NOTE: the post-hoc regret must be greater or equal than 0 but due to numerical error it could still be a
            # small negative number
            assert post_hoc_regret >= -1e-5
            rel_post_hoc_regret = post_hoc_regret / opt_sol_cost[TOTAL_COST]

            # If the solution is feasible, keep track of the relative regret...
            if feasible:
                feas_sol_rel_regret_list.append(rel_post_hoc_regret)
            # ...otherwise keep track of the number of infeasible solutions
            else:
                num_infeas_sol += 1

            cost_list.append(pred_sol_cost[TOTAL_COST])
            opt_cost_list.append(opt_sol_cost[TOTAL_COST])
            regret_list.append(post_hoc_regret)
            rel_regret_list.append(rel_post_hoc_regret)

            # Compute the MSE of the solution
            sol_mse = np.mean(np.square(y_i - y_true_i))
            sol_mse_list.append(sol_mse)

            # Compute MSE of the items' weights
            pred_weights = constraints_i.detach().numpy()
            # CombOptNet encode the whole set of constraints; constraints is the matrix [A|b], where | is the
            # concatenation operation; here, we only care about the A matrix (the items' weights)
            pred_weights = np.squeeze(pred_weights)[:-1]
            assert pred_weights.shape == weights_i.shape == (self._kp_dim, )
            mse_weights = np.mean(np.square(pred_weights - weights_i))
            weights_mse_list.append(mse_weights)

        # Save average metrics in a dictionary
        metrics[COST_STR] = np.mean(cost_list)
        metrics[OPTIMAL_COST_STR] = np.mean(opt_cost_list)
        metrics[REGRET_STR] = np.mean(regret_list)
        metrics[REL_REGRET_STR] = np.mean(rel_regret_list)
        metrics[FEAS_SOL_REL_REGRET_STR] = np.mean(feas_sol_rel_regret_list)
        metrics[MSE_SOL_STR] = np.mean(sol_mse_list)
        metrics[MSE_PARAMS_STR] = np.mean(weights_mse_list)
        metrics[NUM_INFEAS_SOL_STR] = num_infeas_sol / len(y)

        return loss, metrics

########################################################################################################################


class KnapsackCapacityLearningTrainer(ConstraintLearningTrainerBase):
    """
    Manager of the training routine for the KP with unknown item weights.
    """

    # FIXME: repeated code
    def build_model(self,
                    solver_module_params: Dict,
                    backbone_module_params: Dict,
                    metadata: Dict) -> List[torch.Tensor]:
        """
        Instantiate the backbone and the solver modules.
        :param solver_module_params: dict; a subset of the solver module constructor.
        :param backbone_module_params: dict; a subset of the backbone module constructor.
        :param metadata: dict; additional metadata of the optimization problem to solve.
        :return: list of torch.Tensor; the parameters of the resulting architecture.
        """

        self._metadata = metadata

        # Where the backbone and solver models will be saved, inside the working directory.
        self._backbone_path = 'backbone'
        self._solver_path = 'solver'

        self._in_features = backbone_module_params['in_features']
        self._kp_dim = metadata['num_variables']

        # Whether the problem is a minimization or maximation problem
        self._opt_prob_dir = metadata['opt_prob_dir']

        self.backbone_module = \
            KnapsackExtractCapacityFromFeatures(kp_dim=self._kp_dim,
                                                input_dim=backbone_module_params['in_features'],
                                                capacity_min=metadata['min_capacity'],
                                                capacity_max=metadata['max_capacity'],
                                                out_features=backbone_module_params['out_features']).to(self.device)

        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)

        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())

        return model_parameters

    def save(self, filepath: str):
        """
        Save the models on a file. Backbone and solver will be saved in separated folders.
        :param filepath: str; the base savepath.
        :return:
        """

        backbone_full_path = os.path.join(filepath, self._backbone_path)
        solver_full_path = os.path.join(filepath, self._solver_path)

        torch.save(self.backbone_module.state_dict(), backbone_full_path)
        torch.save(self.solver_module.state_dict(), solver_full_path)

    def load(self, filepath: str):
        """
        Load the models from file.
        :param filepath: str; the base loadpath.
        :return:

        """
        backbone_full_path = os.path.join(filepath, self._backbone_path)
        solver_full_path = os.path.join(filepath, self._solver_path)

        self.backbone_module.load_state_dict(torch.load(backbone_full_path))
        self.solver_module.load_state_dict(torch.load(solver_full_path))

    def forward(self, x: torch.Tensor):
        """
        Given are the input features and the optimization problem parameters, compute the solution.
        :param x: torch.Tensor; the input features.
        :return:
        """

        # FIXME: this should not be hardcoded
        # Sanity check: the input features, the cost vector, the item weights and the capacity are concatenated in a
        # single tensor; check whether the dimension is the expected one
        assert x.shape[1] == self._in_features + (self._kp_dim * 2) + 1

        # Unpack the input features and the optimization problem parameters
        in_features = x[:, self._metadata['features_start_idx']:self._metadata['features_end_idx']]
        cost_vector = x[:, self._metadata['values_start_idx']:self._metadata['values_end_idx']]
        weights = x[:, self._metadata['weights_start_idx']:self._metadata['weights_end_idx']]
        capacity = x[:, self._metadata['capacity_start_idx']:self._metadata['capacity_end_idx']]

        # Predict the constraints matrix
        constraints = self.backbone_module(in_features, weights)

        y_denorm = self.solver_module(cost_vector=-cost_vector, constraints=constraints)

        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm

        return y_denorm, y_denorm_rounded, {}, constraints, cost_vector, weights, capacity

    # FIXME: repeated code
    def calculate_loss_metrics(self, x: torch.Tensor, y_true_norm: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the loss function and the evaluation metrics.
        :param x: torch.Tensor; the input features and the optimization problem parameters.
        :param y_true_norm: torch.Tensor; the true normalized optimal solution.
        :return: torch.Tensor, dict; the loss value and the evaluation metrics.
        """

        y_denorm, y_denorm_roudned, solutions_denorm_dict, constraints, cost_vector, weights, capacity = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        metrics = dict(loss=loss.item())

        y_uncon_denorm = solve_unconstrained(cost_vector=cost_vector, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))

        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))

        # Compute the evaluation metrics
        y = y_denorm.cpu().detach().numpy()
        y_true = y_true_denorm.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        values = cost_vector.cpu().detach().numpy()
        capacity = capacity.cpu().detach().numpy()

        rel_regret_list = list()
        feas_sol_rel_regret_list = list()
        regret_list = list()
        sol_mse_list = list()
        cost_list = list()
        opt_cost_list = list()
        capacity_mse_list = list()
        num_infeas_sol = 0

        # FIXME: the unpacking is too long
        for y_i, y_true_i, weights_i, values_i, capacity_i, constraints_i in zip(y, y_true, weights, values, capacity, constraints):

            # Compute the post-hoc regret given the predicted solution \hat{z} (y_i)
            opt_prob = StochasticCapacityKnapsackProblem(dim=self._kp_dim, penalty=self._penalty)
            opt_prob_params = {'values': torch.as_tensor(values_i),
                               'weights': torch.as_tensor(weights_i)}

            pred_sol_cost, feasible = \
                opt_prob.get_objective_values(y=torch.as_tensor(capacity_i),
                                              sols=torch.as_tensor(y_i),
                                              opt_prob_params=opt_prob_params)

            opt_sol_cost, _ = \
                opt_prob.get_objective_values(y=torch.as_tensor(capacity_i),
                                              sols=torch.as_tensor(y_true_i),
                                              opt_prob_params=opt_prob_params)

            # FIXME: repeated code from KnapsackWeightsLearningTrainer.calculate_loss_metrics
            post_hoc_regret = pred_sol_cost[TOTAL_COST] - opt_sol_cost[TOTAL_COST]
            if not opt_prob.is_minimization_problem:
                post_hoc_regret *= -1
            assert post_hoc_regret >= 0
            rel_post_hoc_regret = post_hoc_regret / opt_sol_cost[TOTAL_COST]

            # If the solution is feasible, keep track of the relative regret...
            if feasible:
                feas_sol_rel_regret_list.append(rel_post_hoc_regret)
            # ...otherwise keep track of the number of infeasible solutions
            else:
                num_infeas_sol += 1

            # Keep track of the metrics we care about
            cost_list.append(pred_sol_cost[TOTAL_COST])
            opt_cost_list.append(opt_sol_cost[TOTAL_COST])
            regret_list.append(post_hoc_regret)
            rel_regret_list.append(rel_post_hoc_regret)

            # Compute the MSE of the solution
            sol_mse = np.mean(np.square(y_i - y_true_i))
            sol_mse_list.append(sol_mse)

            # Compute MSE of the items' weights
            pred_capacity = constraints_i.detach().numpy()
            # CombOptNet encode the whole set of constraints; constraints is the matrix [A|b], where | is the
            # concatenation operation; here, we only care about the b matrix (the capacity)
            pred_capacity = np.squeeze(pred_capacity)[-1]
            pred_capacity = np.array([pred_capacity])
            assert pred_capacity.shape == capacity_i.shape == (1,)
            mse_capacity = np.mean(np.square(pred_capacity - capacity_i))
            capacity_mse_list.append(mse_capacity)

            # Save average metrics in a dictionary
        metrics[COST_STR] = np.mean(cost_list)
        metrics[OPTIMAL_COST_STR] = np.mean(opt_cost_list)
        metrics[REGRET_STR] = np.mean(regret_list)
        metrics[REL_REGRET_STR] = np.mean(rel_regret_list)
        metrics[FEAS_SOL_REL_REGRET_STR] = np.mean(feas_sol_rel_regret_list)
        metrics[MSE_SOL_STR] = np.mean(sol_mse_list)
        metrics[MSE_PARAMS_STR] = np.mean(capacity_mse_list)
        metrics[NUM_INFEAS_SOL_STR] = num_infeas_sol / len(y)

        return loss, metrics

########################################################################################################################


class SetCoverDemandsLearningTrainer(ConstraintLearningTrainerBase):
    """
    Manager of the training routine for the weighted set multi-cover with unknown demands (aka the coverage
    requirements).
    """

    # FIXME: repeated code
    def build_model(self,
                    solver_module_params: Dict,
                    backbone_module_params: Dict,
                    metadata: Dict) -> List[torch.Tensor]:
        """
        Instantiate the backbone and the solver modules.
        :param solver_module_params: dict; a subset of the solver module constructor.
        :param backbone_module_params: dict; a subset of the backbone module constructor.
        :param metadata: dict; additional metadata of the optimization problem to solve.
        :return: list of torch.Tensor; the parameters of the resulting architecture.
        """

        self._metadata = metadata

        # Where the backbone and solver models will be saved, inside the working directory.
        self._backbone_path = 'backbone'
        self._solver_path = 'solver'

        # Whether the problem is a minimization or maximation problem
        self._opt_prob_dir = metadata['opt_prob_dir']

        self._in_features = backbone_module_params['in_features']
        self._out_features = backbone_module_params['out_features']

        # The number of decision variables involved in the optimization problem
        self._num_vars = metadata['num_variables']

        self.backbone_module = \
            SetCoverExtractDemandsFromFeatures(input_dim=backbone_module_params['in_features'],
                                               demand_min=metadata['min_demand'],
                                               demand_max=metadata['max_demand'],
                                               out_features=backbone_module_params['out_features']).to(self.device)
        self.solver_module = get_solver_module(variable_range=self.variable_range,
                                               **solver_module_params).to(self.device)

        model_parameters = list(self.backbone_module.parameters()) + list(self.solver_module.parameters())

        return model_parameters

    def save(self, filepath):
        """
        Save the models on a file. Backbone and solver will be saved in separated folders.
        :param filepath: str; the base savepath.
        :return:
        """

        backbone_full_path = os.path.join(filepath, self._backbone_path)
        solver_full_path = os.path.join(filepath, self._solver_path)
        torch.save(self.backbone_module.state_dict(), backbone_full_path)
        torch.save(self.solver_module.state_dict(), solver_full_path)

    def load(self, filepath):
        """
        Load the models from file.
        :param filepath: str; the base loadpath.
        :return:
        """

        backbone_full_path = os.path.join(filepath, self._backbone_path)
        solver_full_path = os.path.join(filepath, self._solver_path)
        self.backbone_module.load_state_dict(torch.load(backbone_full_path))
        self.solver_module.load_state_dict(torch.load(solver_full_path))

    def forward(self, x):
        """
        Given are the input features and the optimization problem parameters, compute the solution.
        :param x: torch.Tensor; the input features.
        :return:
        """

        batch_size = len(x)
        avlbty_len = self._out_features * self._num_vars
        prod_costs_len = self._out_features
        set_costs_len = self._num_vars
        demands_len = self._out_features

        tot_len = self._in_features + avlbty_len + prod_costs_len + set_costs_len + demands_len

        # FIXME: this should not be hardcoded
        assert x.shape[1] == tot_len

        in_features = x[:, self._metadata["features_start_idx"]:self._metadata["features_end_idx"]]
        availability = x[:, self._metadata["availability_start_idx"]:self._metadata["availability_end_idx"]]
        availability = availability.reshape(batch_size, self._out_features, self._num_vars)
        prod_costs = x[:, self._metadata["prod_costs_start_idx"]:self._metadata["prod_costs_end_idx"]]
        set_costs = x[:, self._metadata["set_costs_start_idx"]:self._metadata["set_costs_end_idx"]]
        true_demands = x[:, self._metadata["demands_start_idx"]:self._metadata["demands_end_idx"]]

        availability = availability.float()
        set_costs = set_costs.float()
        prod_costs = prod_costs.float()
        true_demands = true_demands.float()

        demands = self.backbone_module(in_features)
        demands = torch.unsqueeze(demands, -1)

        constraints = torch.cat((-availability, demands), dim=-1)

        y_denorm = self.solver_module(cost_vector=set_costs, constraints=constraints)
        if isinstance(self.solver_module, CvxpyModule):
            y_denorm_rounded = knapsack_round(y_denorm=y_denorm, constraints=constraints,
                                              knapsack_capacity=self.backbone_module.knapsack_capacity)
        else:
            y_denorm_rounded = y_denorm

        params = dict()
        params['set_costs'] = set_costs
        params['prod_costs'] = prod_costs
        params['availability'] = availability
        params['true_demands'] = true_demands

        return y_denorm, y_denorm_rounded, {}, params

    # FIXME: repeated code
    def calculate_loss_metrics(self, x: torch.Tensor, y_true_norm: torch.Tensor):
        """
        Compute the loss function and the evaluation metrics.
        :param x: torch.Tensor; the input features and the optimization problem parameters.
        :param y_true_norm: torch.Tensor; the true normalized optimal solution.
        :return: torch.Tensor, dict; the loss value and the evaluation metrics.
        """
        y_denorm, y_denorm_roudned, solutions_denorm_dict, params = self.forward(x)
        y_norm = compute_normalized_solution(y_denorm, **self.variable_range)
        loss = self.loss_fn(y_norm.double(), y_true_norm)

        set_costs = params['set_costs']
        prod_costs = params['prod_costs']
        availability = params['availability']
        true_demands = params['true_demands']

        metrics = dict(loss=loss.item())
        y_uncon_denorm = solve_unconstrained(cost_vector=set_costs, **self.variable_range)
        y_true_denorm = compute_denormalized_solution(y_true_norm, **self.true_variable_range)
        metrics.update(compute_metrics(y=y_denorm_roudned, y_true=y_true_denorm, y_uncon=y_uncon_denorm))
        for prefix, solution in solutions_denorm_dict.items():
            metrics.update(
                compute_metrics(y=solution, y_true=y_true_denorm, y_uncon=y_uncon_denorm, prefix=prefix + "_"))

        y = y_denorm.cpu().detach().numpy()
        y_true = y_true_denorm.cpu().detach().numpy()
        availability = availability.cpu().detach().numpy()
        set_costs = set_costs.cpu().detach().numpy()
        prod_costs = prod_costs.cpu().detach().numpy()
        true_demands = true_demands.cpu().detach().numpy()

        rel_regret_list = list()
        feas_sol_rel_regret_list = list()
        regret_list = list()
        mse_list = list()
        cost_list = list()
        opt_cost_list = list()
        num_infeas_sol = 0

        for y_i, y_true_i, a_i, set_c_i, prod_c_i, dem_i in zip(y, y_true, availability, set_costs, prod_costs, true_demands):

            pred_demands = a_i @ y_i
            not_covered_demands = np.clip(dem_i - pred_demands, a_min=0, a_max=np.inf)

            suboptimality_cost = set_c_i @ y_i
            penalty_cost = self._penalty * not_covered_demands @ prod_c_i

            # Compute the metrics we care about
            cost_i = penalty_cost + suboptimality_cost
            true_cost_i = y_true_i @ set_c_i
            regret_i = self._opt_prob_dir * (true_cost_i - cost_i)
            rel_regret_i = regret_i / true_cost_i
            feas_sol_rel_regret_i = self._opt_prob_dir * (true_cost_i - suboptimality_cost) / true_cost_i
            mse_i = np.mean(np.square(y_i - y_true_i))

            cost_list.append(cost_i)
            opt_cost_list.append(true_cost_i)
            regret_list.append(regret_i)
            rel_regret_list.append(rel_regret_i)
            feas_sol_rel_regret_list.append(feas_sol_rel_regret_i)
            mse_list.append(mse_i)

            if penalty_cost != 0:
                num_infeas_sol += 1

        # Save average metrics in a dictionary
        metrics[COST_STR] = np.mean(cost_list)
        metrics[OPTIMAL_COST_STR] = np.mean(opt_cost_list)
        metrics[REGRET_STR] = np.mean(regret_list)
        metrics[REL_REGRET_STR] = np.mean(rel_regret_list)
        metrics[FEAS_SOL_REL_REGRET_STR] = np.mean(feas_sol_rel_regret_list)
        metrics[MSE_STR] = np.mean(mse_list)
        metrics[NUM_INFEAS_SOL_STR] = num_infeas_sol / len(y)

        return loss, metrics

