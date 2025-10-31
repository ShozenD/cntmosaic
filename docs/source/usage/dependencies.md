# Dependencies & Requirements

This page provides detailed information about Contact Mosaic's dependencies and system requirements.

## Core Dependencies

Contact Mosaic relies on several key scientific Python packages:

### Essential Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **JAX** | ≥0.4.0 | Numerical computing with automatic differentiation |
| **JAXlib** | ≥0.4.0 | JAX's CPU/GPU/TPU backends |
| **NumPyro** | ≥0.13.0 | Probabilistic programming and Bayesian inference |
| **NumPy** | ≥1.20.0 | Array operations and numerical computing |
| **Pandas** | ≥1.3.0 | Data manipulation and analysis |
| **SciPy** | ≥1.7.0 | Scientific computing utilities |

### Visualization Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **Matplotlib** | ≥3.4.0 | Plotting and visualization |
| **Seaborn** | ≥0.11.0 | Statistical visualization |

### Analysis Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **ArviZ** | ≥0.11.0 | Exploratory analysis of Bayesian models |
| **xarray** | ≥0.19.0 | Labeled multi-dimensional arrays |

---

## Python Version Support

Contact Mosaic officially supports:

- **Python 3.8** (minimum)
- **Python 3.9**
- **Python 3.10**
- **Python 3.11**
- **Python 3.12** (recommended)

> **Note:** We recommend using Python 3.10 or later for the best performance and compatibility with JAX.

---

## System Requirements

### Minimum Requirements

- **RAM:** 8 GB
- **CPU:** Multi-core processor (4+ cores recommended)
- **Disk Space:** 2 GB free space
- **OS:** macOS, Linux, or Windows 10+

### Recommended for Large-Scale Analysis

- **RAM:** 16 GB or more
- **CPU:** 8+ cores
- **GPU:** NVIDIA GPU with 8+ GB VRAM (for GPU acceleration)
- **Disk Space:** 10 GB free space

---

## Optional Dependencies

### GPU Acceleration

For GPU support, you'll need:

1. **NVIDIA GPU** with CUDA Compute Capability 3.5+
2. **CUDA Toolkit** (11.0+ or 12.0+)
3. **cuDNN** (compatible with your CUDA version)
4. **JAX with CUDA support** (see [Installation Guide](installation.md#gpu-support-optional))

**Benefits:**
- 10-100x speedup for MCMC inference
- Faster model fitting for large datasets
- Enables analysis of high-resolution models

### Development Tools

For contributing to Contact Mosaic:

```bash
pip install -e ".[dev]"
```

This installs additional tools:
- **pytest** - Testing framework
- **pytest-cov** - Coverage reporting
- **black** - Code formatting
- **flake8** - Linting
- **mypy** - Type checking
- **pre-commit** - Git hooks

### Documentation Building

For building documentation locally:

```bash
cd docs
pip install -r requirements.txt
```

This installs:
- **Sphinx** - Documentation generator
- **sphinx-rtd-theme** - ReadTheDocs theme
- **sphinx-autodoc** - API documentation
- **nbsphinx** - Jupyter notebook integration

---

## Platform-Specific Notes

### macOS

**Apple Silicon (M1/M2/M3):**
- JAX has native support for Apple Silicon
- Install via pip as usual: `pip install jax jaxlib`
- GPU acceleration not available (yet)

**Intel Macs:**
- Full compatibility with all features
- GPU acceleration possible with eGPU setups

### Linux

**Ubuntu/Debian:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv
```

**RHEL/CentOS:**
```bash
# Install system dependencies
sudo yum install python3-devel python3-pip
```

**GPU Support:**
- Follow NVIDIA's instructions for CUDA installation
- Ensure driver version matches CUDA version

### Windows

**WSL2 Recommended:**
For the best experience on Windows, use Windows Subsystem for Linux 2 (WSL2):

```powershell
wsl --install
```

Then follow Linux installation instructions within WSL2.

**Native Windows:**
- All features work on native Windows
- GPU support requires NVIDIA drivers and CUDA Toolkit
- Use PowerShell or Command Prompt for installation

---

## Dependency Management

### Using pip

The simplest approach for most users:

```bash
pip install -e .
```

Dependencies are automatically installed from PyPI.

### Using Conda

For a more isolated environment:

```bash
conda env create -f environment.yml
conda activate brc
```

This creates an environment with all dependencies pre-configured.

### Using Poetry (Advanced)

For reproducible environments:

```bash
poetry install
poetry shell
```

---

## Checking Installed Versions

To check your installed versions:

```python
import jax
import numpyro
import numpy as np
import pandas as pd

print(f"JAX version: {jax.__version__}")
print(f"NumPyro version: {numpyro.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Check available devices
print(f"\nAvailable devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
```

---

## Upgrading Dependencies

### Upgrading JAX

```bash
pip install --upgrade jax jaxlib
```

For GPU support:
```bash
pip install --upgrade "jax[cuda12]"
```

### Upgrading NumPyro

```bash
pip install --upgrade numpyro
```

### Upgrading All Dependencies

```bash
pip install --upgrade -r requirements.txt
```

---

## Compatibility Matrix

### JAX + NumPyro Compatibility

| JAX Version | NumPyro Version | Status |
|-------------|-----------------|--------|
| 0.4.20+ | 0.14.0+ | ✅ Recommended |
| 0.4.10-0.4.19 | 0.13.0-0.13.2 | ✅ Supported |
| 0.3.x | 0.12.x | ⚠️ Legacy |
| < 0.3 | < 0.12 | ❌ Not supported |

### Python + JAX Compatibility

| Python Version | JAX Support | Notes |
|----------------|-------------|-------|
| 3.12 | ✅ Full | Recommended |
| 3.11 | ✅ Full | Recommended |
| 3.10 | ✅ Full | Recommended |
| 3.9 | ✅ Full | Supported |
| 3.8 | ⚠️ Limited | Minimum version |
| 3.7 | ❌ No | End of life |

---

## Known Issues

### JAX on Windows

**Issue:** JAX performance may be suboptimal on Windows.

**Solution:** Use WSL2 for better performance, or accept slightly slower execution.

### NumPyro with Old JAX Versions

**Issue:** Some NumPyro features require recent JAX versions.

**Solution:** Ensure you have JAX ≥ 0.4.10.

### Memory Issues with Large Models

**Issue:** Out-of-memory errors during MCMC.

**Solution:** 
1. Use SVI instead of MCMC
2. Reduce the number of chains/samples
3. Enable JAX's memory efficiency mode:
   ```python
   import os
   os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
   ```

---

## Getting Help with Dependencies

If you encounter dependency-related issues:

1. **Check versions:** Ensure you meet minimum requirements
2. **Update packages:** Try upgrading to the latest versions
3. **Search issues:** Look for similar problems on [GitHub Issues](https://github.com/ShozenD/cntmosaic/issues)
4. **Ask for help:** Open a new issue with your environment details

### Reporting Environment Information

When reporting issues, include:

```python
import sys
import platform
import jax
import numpyro

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"JAX: {jax.__version__}")
print(f"NumPyro: {numpyro.__version__}")
print(f"JAX backend: {jax.default_backend()}")
```
