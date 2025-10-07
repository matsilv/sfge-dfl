You can find examples of configuration files in the `experiments` folder.

In the configuration file you define the parameters of the experiment. 
Some of them are valid for all the methods and problems:

* `optimization_problem`. A string identifier of the optimization problem. Available are `knapsack`, `quadratic_kp`, 
`fractional_kp`, `stochastic_capacity_kp`, `stochastic_weights_kp` and `wsmc`.
* `relative_capacity`. The KP capacity is computed as `capacity=relative_capacity*sum(weights)` (only for linear and 
                       quadratic KP and for the KP with stochastic weights).
* `correlate_values_and_weights`. See the dataset generation process below for a more detailed description (only for the 
                                  KP problems).
* `rho`. See the dataset generation process below for a more detailed description (only for the 
                                  KP problems).
* `penalty_factor`. The penalty factor as a string (for the only fractional KP) or float (for the WSMC).
* `penalty`. The penalty factor as an int (for the only KP with stochastic capacity or weights).
* `capacity`. The capacity value as integer (for the only fractional KP).
* `num_prods`. The number of products (a.k.a. items) as int (for the only WSMC).
* `num_sets`. The number of sets as int (for the only WSMC).
* `n`. Number of instances to generate (this parameter is used to load the data since it is expected to be in the 
filename).
* `input_dim`. Number of input features.
* `output_dim`. Size of optimization problem parameters to predict, e.g. the knapsack values, the covering demands, 
the KP weights.
* `mult_noise`. Between the input and target features, a multiplicative noise is introduced to make the problem harder 
(this parameter is used to load the data since it is expected to be in the filename).
* `add_noise`. Between the input and target features, an additive noise is introduced to make the problem harder (this 
parameter is used to load the data since it is expected to be in the filename).
* `deg`. The degree of the polynomial relationship between the input and the target (this parameter is used to load the 
data since it is expected to be in the filename).
* `epochs`. Maximum number of training epochs.
* `annealing_epochs`. The number of epochs required to anneal some model parameters from the initial to the final value 
                      (required only if an annealing procedure is used).
* `min_annealing_val`. The min allowed annealing factor (required only if an annealing procedure is used).
* `patience`. How many epochs to wait if the monitored metric is not improved before stopping the training.
* `batch_size`. The training batch size.
* `num_runs_per_seed`. Number of training runs for each method on the same dataset and training/validation/test split.
* `prop_training`. The fraction of the dataset to use for training.
* `prop_validation`. The fraction of the dataset to use for validation.
* `torch_seed`. The PyTorch random seed.
* `numpy_seed`. The Numpy random seeds used to generate the datasets (this parameter is used to load the data since it is 
expected to be in the filename).
* `rnd_split_seed`. The seeds used to split the dataset in training/validation/test sets.
* `scale_predictions`. Predictions are standardized to have 0 mean and unit variance. Optionally, you can set the 
standardization layer as trainable: the mean and standard deviation of the rescaling operation become trainable 
parameters. Remove this key to skip standardization.
* `method_configurations`. List of configurations for each of the method. Each configuration should be defined as 
                           follow:
    * `name`. Method's name. Available are:
        1. `SFGE`: score function gradient estimation.
        2. `SFGE+SCE`: score function gradient estimation to minimize the regret and the self-contrastive term; it only 
        works for problems with linear or quadratic objective. It is referred as SFGE-MAP in the paper.
        3. `SCE`: self-contrastive estimation.
        4. `SPO`: smart predict-then-optimize.
        5. `Blackbox`: blackbox.
        6. `MSE`: prediction-focused learning for MSE minimization.
        7. `MLE`: prediction-focused learning for MLE.
        8. `DPO`: differentiable perturbed optimizers with Fenchel-Young loss.
    * `lr`. Learning rate for Adam optimizer.
    * `monitor`. The metric to monitor for early stopping (e.g. `val_mse`).
    * `min_delta`. The minimum improvement required to avoid early stopping.
    * `covariance_type`. The available methods are:
      1. `static`: the value is static and does not change during training.
      2. `trainable`: the value is trainable but does not depend on the input features.
      3. `contextual`: the value is parametrized by predictive model.
      4. `linear_annealing`: the value is linearly annealed from an initial to a final value in a given number of epochs.
    * `init_log_std_dev`. The initial value of the logarithm of the standard deviation. This parameter is only used for 
                          probabilistic model.
    * `alpha`. The $\alpha$ parameter of SPO (for SPO only).
    * `lmbd`. The $\lambda$ parameter of Blackbox (for Blackbox only).
    * `loss`. The loss function (for Blackbox only). Available are `Regret` and `Hamming`.
    * `distr_type`. The distribution modeled by the predictive model (only for MLE). The only available value is `gaussian`.
    * `std_batch_vals`. If `true` the regret values within a batch are standardized as described in the appendix of the paper.