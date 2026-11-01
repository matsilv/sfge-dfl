This repository is the codebase for the paper "Score Function Gradient Estimation to Widen the Applicability of 
Decision-focused Learning". 

To visualize the full documentation run the command `mkdocs build` and then `mkdocs serve`.

## Setup

The latest Python version tested is 3.10. Project requirements are listed in requirements.txt

Alternatively, you can build a docker image with the command `docker build -t dflrepo .`. This command will create a 
docker image called `dflrepo`. To run the image `docker run -e PYTHONPATH=/app -it dflrepo /bin/bash`. The 
command will run the `dflrepo` image setting the environment variable to the root folder of the repo. At default you 
should use `python3` to run the scripts.

Optimization problems are modeled and solved with Gurobi (https://www.gurobi.com/). 
More instructions to get an academic 
license here: https://www.gurobi.com/academia/academic-program-and-licenses/.

## Tests

The `tests/` folder holds a `unittest` suite. Run it from the repo root:

```
python -m unittest discover -s tests -t . -v
```

A Gurobi license is required. Each problem's test module generates its own small dataset in `data/data/` and trains a
few methods on it, so the full suite takes a while. See the [documentation](docs/usage/testing.md) for details.

## Citation

This code accompanies the paper [Score Function Gradient Estimation to Widen the Applicability of Decision-Focused
Learning](https://doi.org/10.1613/jair.1.19498) (Journal of Artificial Intelligence Research, 2026). If you use it,
please cite:

```bibtex
@article{silvestri2026score,
  title     = {Score Function Gradient Estimation to Widen the Applicability of Decision-Focused Learning},
  author    = {Silvestri, Mattia and Berden, Senne and Signorelli, Gaetano and Mahmuto{\u{g}}ullar{\i}, Ali {\.I}rfan and
               Mandi, Jayanta and Amos, Brandon and Guns, Tias and Lombardi, Michele},
  journal   = {Journal of Artificial Intelligence Research},
  volume    = {85},
  year      = {2026},
  publisher = {AI Access Foundation},
  doi       = {10.1613/jair.1.19498},
  url       = {https://doi.org/10.1613/jair.1.19498}
}
```
