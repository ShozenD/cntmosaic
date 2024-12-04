# Contact Mosaic

## What is Contact Mosaic?
Contact Mosaic (`cntmosaic`) is a Python package for analysing social contact patterns from 
social contact data. It provides a set of tools to process, analyse, simulate, and visualise social contact data.
It also provides a set of models to infer social contact matrices from real world social contact data.
The models in `cntmosaic` are implemented using the probabilistic programming language [Numpyro](https://num.pyro.ai/en/stable/index.html) which allows for
both Hamiltonian Monte Carlo (HMC) based full Bayesian inference and fast stochastic variational inference (SVI).

## Setup
### Local Setup

To install the package locally, create a new virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Then, install the dependencies
```bash
pip install -r requirements.txt
```

### Imperial College HPC
The following instructions are for setting up the analysis environment on the [Imperial College HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/). There are two ways to setup the environment depending on the need. For basic usage, the Easybuild Python environment is sufficient. However, if you wish to work interactively on the server (i.e., use JupyterHub), a custom conda environment must be created.

#### Easybuild Python
Login to the HPC via SSH and cd into the project directory. Load a recent version of the SciPy-bundle.
```bash
cd high_res_brc
module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0
```
Create a virtual environment and activate it
```bash
virtualenv .venv
source .venv/bin/activate
```
Install the dependencies using pip
```bash
pip install -r requirements.txt
```

#### Enabling JupyterHub
To enable JupyterHub, a custom conda environment must be created. First, login to the HPC via SSH and load the module
```bash
eval "$(~/miniforge3/bin/conda shell.bash hook)"
```
Setup a new conda environment
```bash
conda create -n cntmosaic python=3.12 ipykernel jupyter_client
```
Activate the environment
```bash
conda activate cntmosaic
```
Install the required packages using pip
```bash
pip install -r requirements.txt
```
Install the python kernel for Jupyter
```bash
python -m ipykernel install --user --name python312_cntmosaic --display-name "Python3.12 (cntmosaic)"
```
Now, you can start a new [Jupyter Hub](https://jupyter.rcs.imperial.ac.uk/) session and select the new ```Python3.12 (cntmosaic)``` kernel icon in the Jupyter Launcher.