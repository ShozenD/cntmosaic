# Installation Guide

**Contact Mosaic** (`cntmosaic`) is currently under active development and is not yet available on PyPI. To use the package, you'll need to install it from source.

## Prerequisites

Before installing Contact Mosaic, ensure you have:

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **pip** (Python package installer)
- **git** (for cloning the repository)

### Checking Your Python Version

```bash
python --version
# or
python3 --version
```

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/) or use your system's package manager.

---

## Installation Methods

Choose the installation method that best fits your use case:

### Method 1: Standard Installation (Recommended for Most Users)

This is the simplest method for general use on your local machine.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```

#### Step 2: Create a Virtual Environment

**On macOS/Linux:**
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

#### Step 3: Install the Package

Install in **editable mode** (recommended for development or staying up-to-date):
```bash
pip install -e .
```

This will automatically install all required dependencies listed in `pyproject.toml`.

Or install normally:
```bash
pip install .
```

> **Note:** All core dependencies (JAX, NumPyro, NumPy, Pandas, etc.) will be installed automatically.

#### Step 4: Verify Installation

Test that the package is installed correctly:
```bash
python -c "import cntmosaic; print('Installation successful!')"
```

---

### Method 2: Installation with Conda

If you prefer using Conda for package management, use this method.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```

#### Step 2: Create Conda Environment

Create a new environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate brc
```

Or create a minimal environment manually:
```bash
conda create -n cntmosaic python=3.12
conda activate cntmosaic
```

#### Step 3: Install the Package

```bash
pip install -e .
```

#### Step 4: Verify Installation

```bash
python -c "import cntmosaic; print('Installation successful!')"
```

---

### Method 3: Development Installation

For contributors or those who want to modify the source code.

#### Step 1: Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/cntmosaic.git
cd cntmosaic
```

#### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step 3: Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

#### Step 4: Install Pre-commit Hooks (Optional)

```bash
pre-commit install
```

#### Step 5: Run Tests

Verify everything works:
```bash
pytest
```

---

## GPU Support (Optional)

Contact Mosaic uses **JAX** for numerical computations, which can leverage GPU acceleration for significant performance improvements.

### Installing JAX with CUDA Support

For NVIDIA GPUs with CUDA 12:
```bash
pip install --upgrade "jax[cuda12]"
```

For NVIDIA GPUs with CUDA 11:
```bash
pip install --upgrade "jax[cuda11]"
```

### Verifying GPU Support

```python
import jax
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

If GPU is available, you should see `gpu` in the device list.

> **Note:** GPU support requires appropriate NVIDIA drivers and CUDA toolkit installed on your system. See the [JAX installation guide](https://github.com/google/jax#installation) for detailed instructions.

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

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'jax'`

**Solution:** Install JAX explicitly:
```bash
pip install jax jaxlib
```

#### Issue: `ImportError: cannot import name 'XXX' from 'cntmosaic'`

**Solution:** Reinstall in editable mode:
```bash
pip install -e . --force-reinstall
```

#### Issue: Virtual environment activation fails on Windows

**Solution:** Use PowerShell and enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

#### Issue: NumPyro MCMC runs very slowly

**Solution:** 
1. Ensure JAX is using XLA compilation (should be automatic)
2. For large models, consider using GPU acceleration
3. Try using SVI instead of MCMC for faster inference

#### Issue: Out of memory errors during inference

**Solution:**
1. Reduce batch size in your model
2. Use fewer MCMC samples
3. Enable JAX's memory preallocation:
   ```python
   import os
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
   ```

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/ShozenD/cntmosaic/issues) page
2. Search for similar problems in closed issues
3. Open a new issue with:
   - Your Python version (`python --version`)
   - Your OS and version
   - Full error message and traceback
   - Minimal reproducible example

---

## Next Steps

Now that you have Contact Mosaic installed, proceed to:

- **[Quickstart Guide](quickstart.rst)** - Get started with a simple example
- **[Key Concepts](concepts.md)** - Understand the fundamentals
- **[Tutorials](../tutorials/index.rst)** - Explore detailed examples

---

## Keeping Your Installation Up to Date

Since the package is under active development, you may want to update periodically:

```bash
cd /path/to/cntmosaic
git pull origin main
pip install -e . --force-reinstall
```

Or if you installed without editable mode:
```bash
cd /path/to/cntmosaic
git pull origin main
pip install . --upgrade
```
