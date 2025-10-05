import torch
import torch.nn as nn

########################################################################################################################


class LinearModel(torch.nn.Module):
    """
    Linear regression model.
    """
    def __init__(self,
                 out_features: int,
                 in_features: int):

        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor):
        """
        Make predictions given the input features.
        :param x: torch.Tensor; the input features.
        :return: torch.Tensor; the predictions.
        """

        x = self.fc1(x.float())
        return x


class MLP(torch.nn.Module):
    def __init__(self, out_features, in_features, hidden_layer_size, output_nonlinearity):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_layer_size)
        self.fc2 = nn.Linear(in_features=hidden_layer_size, out_features=out_features)
        self.output_nonlinearity_fn = nonlinearity_dict[output_nonlinearity]

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))
        x = self.fc2(x)
        x = self.output_nonlinearity_fn(x)
        return x


# FIXME: not tested
class KnapsackMLP(MLP):
    """
    Predicts normalized solution y (range [-0.5, 0.5])
    """

    def __init__(self, num_variables, reduced_embed_dim, embed_dim=4096, **kwargs):
        super().__init__(in_features=num_variables * reduced_embed_dim, out_features=num_variables,
                         output_nonlinearity='sigmoid', **kwargs)
        self.reduce_embedding_layer = nn.Linear(in_features=embed_dim, out_features=reduced_embed_dim)

    def forward(self, x):
        bs = x.shape[0]
        x = self.reduce_embedding_layer(x.float())
        x = x.reshape(shape=(bs, -1))
        x = super().forward(x)
        y_norm = x - 0.5
        return y_norm


# FIXME: not tested
class RandomConstraintsMLP(MLP):
    """
    Predicts normalized solution y (range [-0.5, 0.5])
    """

    def __init__(self, num_variables, **kwargs):
        super().__init__(in_features=num_variables, out_features=num_variables,
                         output_nonlinearity='sigmoid', **kwargs)

    def forward(self, x):
        x = super().forward(x)
        y_norm = x - 0.5
        return y_norm


# FIXME: not tested
class KnapsackExtractWeightsCostFromEmbeddingMLP(MLP):
    """
    Extracts weights and prices of vector-embedding of Knapsack instance

    @return: torch.Tensor of shape (bs, num_variables) with negative extracted prices,
             torch.Tensor of shape (bs, num_constraints, num_variables + 1) with extracted weights and negative knapsack capacity
    """

    def __init__(self, num_constraints=1, embed_dim=4096, knapsack_capacity=1.0, weight_min=0.15, weight_max=0.35,
                 cost_min=0.10, cost_max=0.45, output_nonlinearity='sigmoid', **kwargs):
        self.num_constraints = num_constraints

        self.knapsack_capacity = knapsack_capacity
        self.weight_min = weight_min
        self.weight_range = weight_max - weight_min
        self.cost_min = cost_min
        self.cost_range = cost_max - cost_min

        super().__init__(in_features=embed_dim, out_features=num_constraints + 1,
                         output_nonlinearity=output_nonlinearity, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        batch_size = x.shape[0]
        cost, As = x.split([1, self.num_constraints], dim=-1)
        cost = -(self.cost_min + self.cost_range * cost[..., 0])
        As = As.transpose(1, 2)
        As = self.weight_min + self.weight_range * As
        bs = -torch.ones(batch_size, self.num_constraints).to(As.device) * self.knapsack_capacity
        constraints = torch.cat([As, bs[..., None]], dim=-1)
        return cost, constraints

########################################################################################################################


# FIXME: only linear models are supported
class KnapsackExtractWeightsFromFeatures(LinearModel):
    """
    The backbone model that predicts the KP item weights from features.
    """

    def __init__(self,
                 kp_dim: int,
                 input_dim: int,
                 knapsack_capacity: int,
                 weight_min: float,
                 weight_max: float,
                 out_features: int, **kwargs):

        # Problem dimension
        self._kp_dim = kp_dim

        # The only item weights are unknown so the capacity has to be given
        self.knapsack_capacity = knapsack_capacity
        # Min and max values of the item weights used for normalization
        self.weight_min = weight_min
        self.weight_range = weight_max - weight_min

        super().__init__(in_features=input_dim, out_features=out_features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions given the input features.
        :param x: torch.Tensor; the input features.
        :return: torch.Tensor; the predictions.
        """

        # Predict the unknown parameters
        x = super().forward(x)

        batch_size = x.shape[0]

        # Sanity check: since the unknown parameters are the item weights, the second dimension of the predicted tensor
        # must be the number of items (the first dimension is the batch size)
        assert x.shape[1] == self._kp_dim

        # Create the constraints tensor in the form Ax <= b
        As = x
        # Invert the normalization
        As = self.weight_min + self.weight_range * As

        # FIXME: the size of b should not be hardcoded
        bs = -torch.ones(batch_size, 1).to(As.device) * self.knapsack_capacity

        constraints = torch.cat([As, bs], dim=-1)
        # Add a fake dimension for the number of constraints (1 constraint for the KP)
        constraints = constraints.unsqueeze(dim=1)

        return constraints

########################################################################################################################


# FIXME: only linear models are supported
class KnapsackExtractCapacityFromFeatures(LinearModel):
    """
    The backbone model that predicts the KP item weights from features.
    """

    def __init__(self,
                 kp_dim: int,
                 input_dim: int,
                 capacity_min: float,
                 capacity_max: float,
                 out_features: int, **kwargs):

        # Problem dimension
        self._kp_dim = kp_dim

        # Min and max values of the item weights used for normalization
        self.capacity_min = capacity_min
        self.capacity_range = capacity_max - capacity_min

        super().__init__(in_features=input_dim, out_features=out_features, **kwargs)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Make predictions given the input features.
        :param x: torch.Tensor; the input features.
        :return: torch.Tensor; the predictions.
        """

        # Predict the unknown parameters
        x = super().forward(x)

        # Sanity check: since the unknown parameters is the capacity, the second dimension of the predicted tensor
        # must be the number of items (the first dimension is the batch size)
        assert x.shape[1] == 1

        # Create the constraints tensor in the form Ax <= b
        As = weights.float()

        bs = x
        # Invert the normalization
        bs = self.capacity_min + self.capacity_range * bs

        constraints = torch.cat([As, bs], dim=-1)
        # Add a fake dimension for the number of constraints (1 constraint for the KP)
        constraints = constraints.unsqueeze(dim=1)

        return constraints

########################################################################################################################


# FIXME: only linear models are supported
class SetCoverExtractDemandsFromFeatures(LinearModel):
    """
    Backbone model that predicts the demands (aka coverage requirements) for the Weighted Set Multi-cover problem.
    """

    def __init__(self,
                 input_dim: int,
                 demand_min: float,
                 demand_max: float,
                 out_features: int,
                 **kwargs):

        # Min, max and demands range used for normalization
        self._demand_min = demand_min
        self._demand_max = demand_max
        self._demand_range = demand_max - demand_min

        super().__init__(in_features=input_dim, out_features=out_features, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the demands given the input features.
        :param x: torch.Tensor; the input features.
        :return: torch.Tensor; the predictions.
        """

        demands = super().forward(x)
        demands = self._demand_min + self._demand_range * demands

        return demands

########################################################################################################################


nonlinearity_dict = dict(tanh=torch.tanh, relu=torch.relu, sigmoid=torch.sigmoid, identity=lambda x: x)
baseline_mlp_dict = dict(RandomConstraintsMLP=RandomConstraintsMLP, KnapsackMLP=KnapsackMLP)
