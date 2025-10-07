In the main paper, we use the PFL model to sample scenarios for the Sample Average Approximation (SAA) algorithm. You 
can reproduce the results by running the script `experiments.pfl_plus_saa.py` with the following arguments:

* `config_dir`. The configuration file where the optimization problem, the datasets, the splits and the model to load 
                are defined, in the same way as for the training/testing procedure. Please, note that the **model must 
                be already trained and must be a probabilistic model**.
* `res_dir`. The root filepath where the results for the different datasets, splits and models are saved.
* `--num-scenarios`. A list of number of scenarios to sample.

In each run folder you will find results are saved in file named `saa-results.pkl`.

For example: 

`experiments/stochastic_capacity_kp/configs/50-items/penalty-1/config.json experiments/stochastic_capacity_kp/50-items/penalty-1/ --num-scenarios 1 10 50`

where the content of the config file is:

```
{
	"optimization_problem": "stochastic_capacity_kp",
	"n": 1000,
	"input_dim": 5,
	"problem_dim": 50,
	"output_dim": 1,
	"mult_noise": 0.1,
	"add_noise": 0.03,
	"deg": 5,
	"correlate_values_and_weights": 1,
	"rho": 0,
	"penalty": 1,

	"epochs": 5,
	"patience": 10,
	"batch_size": 32,

	"log_every_n_steps": 50,
	"num_runs_per_seed": 1,
	"prop_training": 0.8,
	"prop_validation": 0.1,

	"torch_seed": 0,
	"numpy_seed": [0],
	"rnd_split_seed": [0],

	"parallelize_runs": false,

	"scale_predictions": {
		"learnable": false
	},

	"method_configurations": [

		{
			"name": "MLE",
			"lr": 0.005,
			"covariance_type": "trainable",
			"init_log_std_dev": 0,
			"monitor": "val_regret",
			"min_delta": 1
		}

	],

	"plot_with_log_scale": true
}
```

The models (for each dataset, split) are saved inside the `experiments/stochastic_capacity_kp/50-items/penalty-1/` and
we report results when sampling 1, 10 and 50 scenarios.
Results are saved in `experiments/stochastic_capacity_kp/50-items/penalty-1/seed-0/rnd-split-seed-1/MLE/run_0` (for each run, split seed and dataset seed)
