You can generate your own synthetic data for the linear and quadratic KP, stochastic KP with unknown item weights and 
the WSMC with stochastic coverage requirements:

* **Linear KP**. You can generate the data for the linear KP by running the script 
`data.generation_scripts.generate_knapsack_data.py`. You can customize the generation process with the following 
arguments:
    * `--input-dim`. The number of input features correlated with the target ones.
    * `--output-dim`. The number of target features that the model has to predict.
    * `--relative-capacity`. Referring to $\gamma$ as the relative capacity, the KP capacity is computed as 
    $b = \gamma \sum_{i=1}^d w_i$.
    * `--degree`. The degree of the polynomial relationship between the input and target features.
    * `--num-instances`. The number of instances to generate (the dataset set size).
    * `--multiplicative-noise`. We add a multiplicative noise as described in shortest-path experiments of [[1]](#1).
    * `--additive-noise`. We further complicate the relationship between input and target features by introducing 
    additive noise.
    * `--correlate_values_and_weights`. This parameter is used to introduce a correlation between the values and the 
    weights of the knapsack items. Value 0 does not introduce any explicit correlation. Value 1 denotes that after 
    generating values from features in a manner analogous to the shortest path datageneration process described in the 
    SPO paper, each generated item value gets multiplied with the associated weight. Value 2 denotes that the generated 
    item value vector will be linearly transformed introduce a correlation between the item value vector and the cost 
    value vector as follows: $c \leftarrow \rho  w + (1 - \rho) * c$.
    * `--rho`. An increasing value of $\rho$ increases the correlation coefficient between c and w, but $\rho$ does not 
    itself denote the exact value of that correlation coefficient. The value of $\rho$ is only relevant when 
    `correlate_values_and_weights` has value 2. In other cases the value of $\rho$ is ignored).
    * `--seeds`. You can generate multiple datasets with different random initialization of Numpy.

* **Quadratic KP**. You can customize the data generation process the same as for the linear version.

* **WSMC**. You can generate the data for the linear KP by running the script 
`data.generation_scripts.generate_knapsack_data.py`. You can customize the generation process with the following 
arguments:
    * `--input-dim`. The number of input features correlated with the target ones.
    * `--output-dim`. The number of target features that the model has to predict.
    * `--num-sets`. The number of sets in the WSMC.
    * `--num-products`. The number of products in the WSMC.
    * `--density`. The density of the availability matrix.
    * `--penalty`. The penalty associated with the recourse action.
    * `--degree`. The degree of the polynomial relationship between the input and target features.
    * `--num-instances`. The number of instances to generate (the dataset set size).
    * `--multiplicative-noise`. We add a multiplicative noise as described in shortest-path experiments of [[1]](#1).
    * `--additive-noise`. We further complicate the relationship between input and target features by introducing 
    additive noise.
    * `--seeds`. You can generate multiple datasets with different random initialization of Numpy.

* **KP with stochastic weights**. You can customize the data generation process the same as for the KP with unknown costs 
                               by running the scripts `generate_stochastic_kp_data.py`.
* **KP with stochastic capacity**. You can customize the data generation process the same as for the KP with unknown costs 
                                     by running the scripts `generate_stochastic_capacity_kp_data.py`.