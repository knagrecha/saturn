# Saturn Installation Guide


## Requirements

Saturn has been tested with Python 3.8 on an AWS p4 instance. Package requirements are listed in requirements.txt

## Installation

To install Saturn, start by cloning the git repository. You can install requirements by running ``pip install -r requirements.txt``.
A different set of requirements is needed to run our examples. You can install those by running ``pip install -r tests/requirements.txt``.
Then, from the main folder, run ``pip install .``.

You will also need to specify where Saturn should store its Library of techniques. You can do this by setting the environment
variable SATURN_LIBRARY_PATH to the appropriate directory.

At the moment, Saturn relies on [Gurobi](https://www.gurobi.com/) to produce its execution plan. You will have to follow
the procedure [here](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-) to install Gurobi
and acquire a license for usage. You will also need to follow any additional requirements for PuLP solver configuration
listed [here](https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html). 

IMPORTANT: It is highly recommended to use the conda installation of Gurobi instead
rather than the pip installation. The conda version integrates better with APIs like PuLP to record variables in Python
for use in Saturn's executor.

We are actively working on building a minimal-dependency version of Saturn that will have its own solver rather than relying on Gurobi.
We welcome contributions, so feel free to make a PR!

## Verification
To verify your installation, you can run the ``wikitext2/simple-verification.py`` script in the examples directory. 
This will run a toy HPO sweep to confirm that everything is working. You will need to specify the environment
variable SATURN_LIBRARY_PATH so that Saturn knows where to store registered execution techniques.