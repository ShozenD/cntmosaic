# Installation Guide

**Contact Mosaic** (`cntmosaic`) is available on [PyPI](https://pypi.org/project/cntmosaic/).

## Prerequisites

Before installing Contact Mosaic, ensure you have:

- **Python 3.8 or higher** (Python 3.10+ recommended)
- **pip** (Python package installer)

### Checking Your Python Version

```bash
python --version
# or
python3 --version
```

---

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
pip install cntmosaic
```

### Method 2: Install from Source

Use this method to get the latest unreleased changes.

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

#### Step 3: Install the Package

Install in **editable mode** (recommended for development):
```bash
pip install -e .
```

Or install normally:
```bash
pip install .
```

#### Step 4: Verify Installation

```bash
python -c "import cntmosaic; print(cntmosaic.__version__)"
```

---

### Method 3: Installation with Conda

#### Step 1: Clone the Repository

```bash
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```

#### Step 2: Create Conda Environment

Create a new environment from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate cntmosaic
```

Or create a minimal environment manually:
```bash
conda create -n cntmosaic python=3.12
conda activate cntmosaic
```

#### Step 3: Install the Package

```bash
pip install cntmosaic
# or from source
pip install -e .
```

---

### Method 4: Development Installation

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

```bash
pytest
```

---

## GPU Support (Optional)

Contact Mosaic uses **JAX** for numerical computations, which supports GPU acceleration.

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

> **Note:** GPU support requires appropriate NVIDIA drivers and CUDA toolkit. See the
> [JAX installation guide](https://github.com/google/jax#installation) for details.

---

## Imperial College HPC Setup

Special instructions for users of the
[Imperial College HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/).

### Option A: Using Easybuild Python (Recommended for Batch Jobs)

```bash
ssh username@login.hpc.ic.ac.uk
module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0
virtualenv .venv
source .venv/bin/activate
pip install cntmosaic
```

For GPU support:
```bash
pip install --upgrade "jax[cuda12]"
```

### Option B: Using Conda with JupyterHub (For Interactive Work)

```bash
conda create -n cntmosaic python=3.12 ipykernel jupyter_client
conda activate cntmosaic
pip install cntmosaic
python -m ipykernel install --user --name cntmosaic --display-name "Python 3.12 (cntmosaic)"
```

Then navigate to [jupyter.rcs.imperial.ac.uk](https://jupyter.rcs.imperial.ac.uk/) and
select the **"Python 3.12 (cntmosaic)"** kernel.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'jax'`

```bash
pip install jax jaxlib
```

### `ImportError: cannot import name 'XXX' from 'cntmosaic'`

Reinstall in editable mode:
```bash
pip install -e . --force-reinstall
```

### NumPyro MCMC runs very slowly

1. Ensure JAX is using XLA compilation (automatic by default)
2. For large models, use GPU acceleration
3. Consider SVI instead of MCMC for faster approximate inference

### Out of memory errors during inference

1. Reduce the number of MCMC samples
2. Enable JAX's lazy memory allocation:
   ```python
   import os
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
   ```

### Getting Help

If you encounter issues, please open an issue on
[GitHub](https://github.com/ShozenD/cntmosaic/issues) with:
- Your Python version (`python --version`)
- Your OS and version
- Full error message and traceback
- A minimal reproducible example

---

## Next Steps

- **[Quickstart Guide](quickstart.rst)** — Get started with a simple example
- **[API Reference](../api/index.rst)** — Full module and class documentation
