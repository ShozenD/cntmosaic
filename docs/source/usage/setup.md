# Setup

## Local Setup

To install the package locally, create a new virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Then, install the dependencies
```bash
pip install -r requirements.txt
```

## Imperial College HPC
The following instructions are for setting up the analysis environment on the [Imperial College HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/). There are two ways to setup the environment depending on the need. For basic usage, the Easybuild Python environment is sufficient. However, if you wish to work interactively on the server (i.e., use JupyterHub), a custom conda environment must be created.

### Easybuild Python
Login to the HPC via SSH and cd into the project directory. Load a recent version of the SciPy-bundle.
```bash
cd high_res_brc
module load SciPy-bundle/2023.07-gfbf-2023a
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

### Enabling JupyterHub
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