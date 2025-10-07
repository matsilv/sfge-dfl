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
