# high_res_brc

## Installation

### Local  
First, create a virtual environment and activate it.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install the requirements.
```bash
pip install -r requirements.txt
```

### Imperial College HPC
#### First time setup
Follow up to date instruction from the [ICL RCS User Guide](https://icl-rcs-user-guide.readthedocs.io/en/latest/hpc/applications/guides/jupyter/).

Load the anaconda module
```bash
module load anaconda3/personal
```

Create a new conda environment
```bash
conda create -n brc python=3.12
```

Create an environment from the environment.yml file
```bash
conda env create -f environment.yml
```