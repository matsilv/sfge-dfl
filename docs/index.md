## Methods

These are the DFL methods available:

- [x] **Prediction-focused learning**. Actually, this is not a DFL method and it involves training a ML model for maximum 
accuracy. The model trained is then used to make predictions in the optimization model.
- [x] **Smart Predict-then-optimize**. Implementation of the method described in the paper [[1]](#1).
- [x] **Blackbox**. Implementation of the method described in the paper [[2]](#2).
- [x] **Self-contrastive estimation**. Implementation of the Self-contrastive estimation method described in the paper 
[[3]](#3).
- [x] **Differentiable perturbed optimizer** (Fenchel-Young loss) [[4]](#4).
- [x] **Score function gradient estimation**. The method we are proposing in the current paper.

## Usage

In the main project folder, launch the `experiment/experiment_handler.py` script. Launch the script as follows:

`python experiment_handler.py path/to/config.json path/to/results --mode [train|test]` 

where:

* `path/to/config.json` is the experiment configuration file in JSON format.
* `path/to/results` is the folder where results will be saved.
* `--mode [train|test]` is an argument used to specify whether you want to train or test the models defined in the 
configuration file.

The experiment scripts are designed to run from the project main folder and you need to add it to the PYTHONPATH 
environment variable.
In `data/configs` folder you can find the configuration for the data generation routine. If you find it useful, you can 
find Windows .bat scripts to generate the data. 

In the `experiments` folder you can find the configuration files for each problem.

## References

<a id="1">[1]</a> [Smart “Predict, then Optimize”](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2020.3922>).

<a id="2">[2]</a> [Differentiation of blackbox combinatorial solvers](https://openreview.net/pdf?id=BkevoJSYPB).

<a id="3">[3]</a> [Contrastive Losses and Solution Caching for Predict-and-Optimize](https://www.ijcai.org/proceedings/2021/0390.pdf)

<a id="4">[4]</a> [Learning with Differentiable Perturbed Optimizers](https://proceedings.neurips.cc/paper_files/paper/2020/file/6bb56208f672af0dd65451f869fedfd9-Paper.pdf)

