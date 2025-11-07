[![CI](https://github.com/ShozenD/cntmosaic/workflows/CI/badge.svg)](https://github.com/ShozenD/cntmosaic/actions)
[![codecov](https://codecov.io/gh/ShozenD/cntmosaic/graph/badge.svg?token=9U271V3D3H)](https://codecov.io/gh/ShozenD/cntmosaic)

# Contact Mosaic

## What is Contact Mosaic?
Contact Mosaic (`cntmosaic`) is a Python package for analysing social contact patterns from 
social contact data. It provides a set of tools to process, analyse, simulate, and visualise social contact data.
It also provides a set of models to infer social contact matrices from real world social contact data.
The models in `cntmosaic` are implemented using the probabilistic programming language [Numpyro](https://num.pyro.ai/en/stable/index.html) which allows for
both Hamiltonian Monte Carlo (HMC) based full Bayesian inference and fast stochastic variational inference (SVI).

## Installation Guide
#### Step 1: Clone the repository
```bash
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```
#### Step 2: Create a virtual environment and activate it
**On MacOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` appear in your terminal prompt, indicating the virtual environment is active.

#### Step 3: Install the package

Install in **editable mode** (recommended for development or staying up-to-date):
```bash
pip install -e .
```

This will automatically install all required dependencies listed in `pyproject.toml`.

Or install normally:
```bash
pip install .
```

#### Step 4: Verify installation
Test that the package is installed correctly:
```bash
python -c "import cntmosaic; print('Installation successful!')"
```

---

## Imperial College HPC Setup

Special instructions for users of the [Imperial College HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/).

### Option A: Using Easybuild Python (Recommended for Batch Jobs)

#### Step 1: Connect and Navigate

```bash
ssh username@login.hpc.ic.ac.uk
cd /path/to/your/workspace
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```

#### Step 2: Load Python Module

```bash
module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0
```

#### Step 3: Create Virtual Environment

```bash
virtualenv .venv
source .venv/bin/activate
```

#### Step 4: Install Dependencies

```bash
pip install -r requirements_hpc.txt
pip install -e .
```

#### Step 5: Enable GPU (Optional)

For GPU support on HPC, explicitly install JAX with CUDA:
```bash
pip install --upgrade "jax[cuda12]"
```

### Option B: Using Conda with JupyterHub (For Interactive Work)

#### Step 1: Initialize Conda

```bash
eval "$(~/miniforge3/bin/conda shell.bash hook)"
```

If you don't have miniforge installed:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

#### Step 2: Create Conda Environment

```bash
conda create -n cntmosaic python=3.12 ipykernel jupyter_client
conda activate cntmosaic
```

#### Step 3: Install Package

```bash
cd /path/to/cntmosaic
pip install -e .
```

#### Step 4: Register Jupyter Kernel

```bash
python -m ipykernel install --user --name cntmosaic --display-name "Python 3.12 (cntmosaic)"
```

#### Step 5: Launch JupyterHub

1. Navigate to [jupyter.rcs.imperial.ac.uk](https://jupyter.rcs.imperial.ac.uk/)
2. Log in with your Imperial credentials
3. Start a new server
4. Select the **"Python 3.12 (cntmosaic)"** kernel from the launcher

---


## Testing
To run all unit tests in the package, use pytest from the root directory:
```bash
pytest
```

To run tests with coverage report:
```bash
pytest --cov=cntmosaic --cov-report=html
```

To run tests in a specific module:
```bash
pytest cntmosaic/datasets/tests/
pytest cntmosaic/models/tests/
pytest cntmosaic/preprocess/tests/
pytest cntmosaic/sim/tests/
pytest cntmosaic/utils/tests/
```

To run a specific test file:
```bash
pytest cntmosaic/datasets/tests/test_load_polymod_germany.py
```
