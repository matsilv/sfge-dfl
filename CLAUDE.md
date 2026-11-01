# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research codebase for the paper "Score Function Gradient Estimation to Widen the Applicability of Decision-focused Learning". It trains predictive models end-to-end through combinatorial/stochastic optimization problems (DFL) and compares SFGE against baselines (MSE/MLE prediction-focused learning, SPO, Blackbox, SCE, DPO).

## Setup & commands

- Python 3.10 (uses `match` statements). Dependencies are in `requirements.txt` (PyTorch Lightning, Gurobi via `gurobipy`, cvxpy, wandb). A Gurobi license is needed to solve problem instances.
- Every script must be run from the repo root with the root on `PYTHONPATH` (e.g. `export PYTHONPATH=.`). Imports are absolute from the root (`from utils...`, `from data.generation_scripts...`).
- `tests/` holds `unittest` tests. Run them from the repo root with `python -m unittest discover -s tests -t .`. Each module generates its own small dataset in `data/data/`, except `test_fractional_kp`, which is skipped unless its dataset is copied in by hand. See `docs/usage/testing.md`. There is no linter or build step.

### Running code inside the Claude Code Docker container

The repo is a Windows checkout mounted at `/workspace`, so keep these in mind:
- `.venv` is the **Windows host** venv (`Scripts/`, `C:\...\Python310`). Don't use or modify it from the container.
- Use `.claudevenv` instead. It's a Linux venv on the container's Python 3.11, with CPU-only torch and without the CombOptNet extras (jax/jaxlib/ray/cvxpylayers). It's git- and docker-ignored.
  ```bash
  PYTHONPATH=. .claudevenv/bin/python experiments/experiment_handler.py <config.json> <res_dir> --mode train
  PYTHONPATH=. .claudevenv/bin/python -m unittest discover tests
  ```
- To recreate it:
  ```bash
  python3 -m venv .claudevenv && .claudevenv/bin/pip install --upgrade pip
  .claudevenv/bin/pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
  .claudevenv/bin/pip install -r <(sed '/comboptnet/,$d' requirements.txt)
  ```
- The pip `gurobipy` comes with a size-limited license (small models only). For full-size instances, set `GRB_LICENSE_FILE` to a valid license.
- Git: the repo is owned by another user and was checked out with CRLF line endings. Plain `git` refuses to run ("dubious ownership"), and if you work around that, every file shows as modified. Run git as `git -c safe.directory=/workspace -c core.autocrlf=true -c core.filemode=false ...`. Tracked text files use CRLF, so keep CRLF when editing them.

```bash
# Generate data (written to data/data/<problem>/seed-<k>/...); see data/bat_scripts/*.bat for the exact args used in the paper
python data/generation_scripts/generate_knapsack_data.py --input-dim 5 --output-dim 50 --relative-capacity 0.5 --degree 5 \
  --num-instances 1000 --multiplicative-noise 0.1 --additive-noise 0.03 --correlate-values-weights 1 --rho 0 --seeds 0 1 2 3 4

# Train, then evaluate (test mode reloads the best checkpoints from res_dir and writes test-res.pkl)
python experiments/experiment_handler.py experiments/knapsack/configs/50-items/config.json results/knapsack/50-items --mode train
python experiments/experiment_handler.py experiments/knapsack/configs/50-items/config.json results/knapsack/50-items --mode test
#   optional: --wandb --wandb-user <user>

# PFL + Sample Average Approximation (needs an already-trained probabilistic model, e.g. MLE)
python experiments/pfl_plus_saa.py <config.json> <res_dir> --num-scenarios 1 10 50

# Docs
mkdocs build && mkdocs serve

# Docker
docker build -t dflrepo . && docker run -e PYTHONPATH=/app -it dflrepo /bin/bash
```

To run one method quickly, copy a config and keep a single entry in `method_configurations`, with a single value in `numpy_seed`/`rnd_split_seed` and a small `epochs` value.

External baselines run separately: CombOptNet with `cd external/comboptnet && python main.py <yaml>` (Linux only, needs jax/jaxlib/ray), and IntOptC with `cd external/popackingcovering && python runModel.py --capacity .. --penalty .. --rnd-seed ..`.

## Architecture

Everything goes through `experiments/experiment_handler.py`, which reads a JSON config and loops over `numpy_seed` (dataset), then `rnd_split_seed` (train/val/test split), then `num_runs_per_seed`, then each method:

1. `utils/build_data_loaders.py::build_data_loaders` dispatches on `config["optimization_problem"]` (`knapsack`, `quadratic_kp`, `fractional_kp`, `stochastic_capacity_kp`, `stochastic_weights_kp`, `wsmc`). It loads pre-generated CSVs from `data/data/` and returns the dataloaders plus an `OptimizationProblem` instance. **Data filenames are built from config values** (`n`, `input_dim`, `output_dim`, `mult_noise`, `add_noise`, `deg`, `rho`, ...), so a config has to match the generation arguments exactly or the files won't be found.
2. `utils/build_models.py::build_models` maps each method name to its module class (`METHODS` dict). Methods in `PROB_METHODS` (`SFGE`, `SFGE+SCE`, `MLE`) wrap the net in `utils/probabilistic_models.py` (`MultivariateGaussianModule`/`PoissonModule`, using `covariance_type`/`init_log_std_dev`/`distr_type`). The others use `nn.Sequential(net, ScaleLayer)`. `SFGE+SCE` is `SFGEModule` with `sce_loss=True`. Keys in a method config are passed as kwargs to the module constructor, after the probabilistic-model keys are removed.
3. Training uses PyTorch Lightning with early stopping and checkpointing on `model.monitor`. Results go to `<res_dir>/seed-<s>/rnd-split-seed-<r>/<METHOD>/run_<k>/` (CSV logs plus `best-model/` checkpoint).

Key abstractions:
- `dfl_module.py::DFLModule` (a `pl.LightningModule`) is the base class for every method in `methods/`. It provides the shared validation/test step: it solves each instance with the predicted parameters and logs `{val,test}_{mse,regret,relative_regret,...}`. Subclasses mainly override `training_step`. A batch is `(x, y, scaled_y, sol_true, solve_params, opt_model_params)`.
- `optimization_problems/optimization_problem.py::OptimizationProblem` is the abstract solver interface: `solve`, `get_objective_values` (returns a dict keyed by `TOTAL_COST`/`SUBOPTIMALITY_COST`/`PENALTY_COST` plus feasibility), and `obj_type` (`linear`/`quadratic`/`nonlinear`; some methods like SPO/SCE need linear/quadratic objectives). `solve_from_torch` converts tensors and calls `solve`. Stochastic problems (stochastic capacity/weights KP, WSMC) score infeasible solutions with a recourse penalty.
- `utils/datawrapper.py` holds one `Dataset` wrapper per problem family.
- `losses/` holds the losses used by Blackbox (`LOSS_CLASSES`: `Regret`, `Hamming`).

To add a new problem, you need a subclass in `optimization_problems/`, a generation script in `data/generation_scripts/`, a `DataWrapper`, a `build_data_loaders_<problem>` function registered in the dispatch dict, and configs under `experiments/<problem>/configs/`. To add a new method, subclass `DFLModule` in `methods/` and register it in `METHODS` (and `PROB_METHODS` if it needs a probabilistic model).

`plotting.py` (repo root) holds the analysis and plotting code for result folders. The full description of config keys is in `docs/usage/configuration.md`.
