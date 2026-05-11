# Contact Mosaic (cntmosaic) - AI Agent Guide

## Project Overview

Contact Mosaic is a Bayesian inference framework for estimating social contact matrices from survey data. The package uses **NumPyro** (JAX-based probabilistic programming) for both MCMC and SVI inference, supporting GPU acceleration.

**Core Domain**: Epidemiological modeling of age-structured contact patterns for infectious disease research.

## Architecture

### Module Organization

```
cntmosaic/
├── dataloader/        # Contact survey data preprocessing & validation
├── models/            # Bayesian contact matrix models (BRC family, Prem, SocialMix)
│   ├── priors/        # 2D spatial priors (Splines, IGMRF, HSGP, vdKassteele)
│   └── mcmc/          # Custom MCMC kernels (Polya-Gamma, MVN samplers)
├── distributions/     # Custom NumPyro distributions (IGMRF2D, SymIGMRF2D, QuasiPoisson)
├── sim/               # Synthetic contact data generation
├── analysis/          # Model evaluation & summarization
├── preprocess/        # Raw survey data cleaning utilities
├── datasets/          # Bundled datasets & template contact patterns
├── utils/             # Age binning, matrix utilities
└── vis/               # Altair-based visualization
```

### Key Design Patterns

**1. BRC Model Family (Abstract Base Class)**
- Base class: `BRC` (abstract, in `models/_BRC.py`)
- Concrete implementations: `BRCfine`, `BRCrefine`, `HiBRCfine`, `HiBRCrefine`
- All models follow this workflow:
  ```python
  model = BRCfine(dataloader, priors={'rate': Spline2D(...)}, likelihood='negbin')
  model.set_age_dims(0, 85)  # Define age range
  model.run_inference_mcmc(rng_key, num_samples=1000)  # or run_inference_svi()
  samples = model._mcmc_result.get_samples()
  ```

**2. Prior2D Hierarchy**
- All priors inherit from `models/priors/_Prior2D.py`
- Support three modes: `prior_type='global'` (shared), `'partial'` (per-row/column), `'full'` (diagonal/off-diagonal)
- Support compositional transformations: `transform='alr'/'clr'/'ilr'` or `None`
- Examples: `Spline2D`, `PSpline2D`, `IGMRF2D`, `HSGP2D`, `vdKassteele`

**3. DataLoader Pipeline**
- `DataLoader` requires column mapping via `CoordToColumns` dataclass
- Converts pandas DataFrames → xarray Datasets → JAX arrays
- Handles stratification (e.g., gender, setting), repeat interviews, population weighting

**4. Custom Distributions**
- Located in `distributions/`, all inherit from `numpyro.distributions.Distribution`
- `IGMRF2D`: 2D Intrinsic Gaussian Markov Random Field with Kronecker structure
- `SymIGMRF2D`: Symmetric variant with single precision parameter
- Implement efficient sampling via eigendecomposition

## Development Workflow

### Testing
```bash
# Run all tests (use pytest, not unittest)
pytest

# Test specific module
pytest cntmosaic/models/tests/
pytest cntmosaic/models/priors/tests/test_vdKassteele.py

# With coverage
pytest --cov=cntmosaic --cov-report=html
```

**Test Organization**: Each module has `tests/` subdirectory with `test_*.py` files. Tests use pytest fixtures and parametrization.

### Environment Setup

**Local**: Use virtual environment with `pip install -e .` for editable install.

**HPC (Imperial College)**:
- Load Python module: `module load Python/3.10.8-GCCcore-12.2.0`
- For GPU: Install JAX with CUDA support explicitly
- For JupyterHub: Use conda environment with custom kernel

### Jupyter Notebooks
- Located in `tutorials/`
- **Critical**: Notebooks use `nbstripout` to prevent committing outputs (configured in `.gitattributes`)
- Save figures for documentation using: `save_tutorial_figure(chart, "filename")` from `utils._tutorial_utils`

### Package Structure
- Editable install: `pip install -e .`
- Configuration: `pyproject.toml` (hatchling build system)
- Optional extras: `pip install -e ".[dev]"` (testing), `".[cuda12]"` (GPU), `".[viz]"` (advanced plotting)

## JAX/NumPyro Conventions

### Key Patterns
1. **Random number generation**: Always use `jax.random.PRNGKey` and split keys explicitly
2. **NumPyro models**: Defined as instance methods (e.g., `BRCfine.model()`) with `@abstractmethod` in base class
3. **Inference helpers**: Centralized in `models/_numpyro.py`:
   - `run_inference_mcmc()`: NUTS sampler with divergence checking
   - `run_inference_svi()`: SVI with linear onecycle LR schedule
   - `posterior_predictive_mcmc/svi()`: Predictive sampling

### Common Gotchas
- **Array shapes**: Models expect flattened age matrices (e.g., `(A*A,)` not `(A, A)`)
- **Log-space offsets**: Contact intensities use additive log-scale terms (`log_N`, `log_P`, `log_S`)
- **Rate consistency**: Models enforce bidirectional contact balance via population age distribution

## Data Requirements

### Input Format
Contact data requires three DataFrames:
1. **Participants** (`df_part`): `id`, `age_part`, stratification vars
2. **Contacts** (`df_cnt`): `id`, `age_cnt` (or `age_grp_cnt` as IntervalIndex), `y` (count)
3. **Population** (`df_age_dist`): `age`, `size` (proportions/counts)

### Column Mapping
```python
from cntmosaic.dataloader import CoordToColumns, DataLoader

col_map = CoordToColumns(
    age_part='participant_age',
    age_cnt='contact_age',  # Use age_grp_cnt for interval-based ages
    id_var='pid',
    age_pop='age',
    size_pop='population',
    strat_vars_part=['gender'],  # Optional stratification
    repeat_part='wave'  # Optional repeat interview tracking
)
dataloader = DataLoader(df_part, df_cnt, df_age_dist, col_map=col_map)
```

## Common Tasks

### Adding a New Prior
1. Inherit from `Prior2D` in `models/priors/`
2. Implement abstract methods: `sample_single()`, `sample_partial()`, `sample_full()`
3. Add to `models/priors/__init__.py`
4. Create tests in `models/priors/tests/`

### Adding a New Model
1. Inherit from `BRC` in `models/`
2. Override `model()` method (NumPyro generative model)
3. Override `_validate_inputs()` for data validation
4. Add to `models/__init__.py`

### Simulation Workflow
```python
from cntmosaic.sim import Subgroup, ParticipantGenerator, MatrixGenerator, ContactGenerator
from cntmosaic.datasets import load_template_patterns

# 1. Load template patterns (household, school, work, community)
templates = load_template_patterns('USA')

# 2. Define subgroup
subgroup = Subgroup(n=500, age_dist=np.array([...]), mean_cint_margin=15.0)

# 3. Generate synthetic data
part_gen = ParticipantGenerator(subgroup)
participants = part_gen.generate(seed=42)
mat_gen = MatrixGenerator(templates)
matrix = mat_gen.generate_single(subgroup, seed=42)
contact_gen = ContactGenerator(participants, matrix)
contacts = contact_gen.generate(seed=42)
```

## Important Files

- `models/_BRC.py`: Abstract base class for all BRC models (read this first)
- `dataloader/_dataloader.py`: Contains `CoordToColumns` docstring with full field explanations
- `models/_numpyro.py`: Inference wrappers and ArviZ conversion
- `distributions/_IGMRF2D.py`: Core spatial prior implementation
- `tutorials/README.md`: Notebook workflow and figure saving conventions
