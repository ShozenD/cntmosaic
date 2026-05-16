# cntmosaic Models

This directory contains Bayesian contact matrix models for estimating social contact
patterns from survey data.

## Model Family Overview

All models share the Bayesian Rate Consistency framework: contact rates are modelled
as smooth 2D functions of participant and contact ages, and the model enforces
consistency between forward and reciprocal contact rates weighted by population
age distributions.

The naming convention encodes the data resolution and stratification structure:

| Model | Participant age | Contact age | Stratification |
|-------|-----------------|-------------|----------------|
| `AgeMixFF` | 1-year | 1-year | Age only |
| `AgeMixFC` | 1-year | Coarse groups | Age only |
| `AgeMixCC` | Coarse groups | Coarse groups | Age only — *to be implemented* |
| `GenMixFF` | 1-year | 1-year | Age + other features |
| `GenMixFC` | 1-year | Coarse groups | Age + other features |
| `GenMixCC` | Coarse groups | Coarse groups | Age + other features — *to be implemented* |

**F** = Fine (1-year age resolution), **C** = Coarse (age ranges / groups)

---

## Model Descriptions

### `AgeMixFF` — Age Mixing, Fine-Fine

Estimates a social contact matrix at 1-year age resolution from contact survey data
where **both** participant and contact ages are recorded at single-year resolution.

- **Use when:** Your survey records exact ages (or near-exact) for both participants
  and their contacts.
- **Key prior:** Smooth 2D function over the (participant age, contact age) surface
  via B-splines (`Spline2D`, `PSpline2D`) or Gaussian processes (`HSGP2D`, `IGMRF2D`).

**Reference:** Dan et al. (2023), *PLoS Computational Biology*

---

### `AgeMixFC` — Age Mixing, Fine-Coarse

Estimates a social contact matrix at 1-year age resolution from contact survey data
where participant ages are at 1-year resolution but **contact ages are reported in
coarse age groups** (e.g., 0–4, 5–9, …).

- **Use when:** Your survey records exact participant ages but contact ages only in
  broad brackets (common in diary-based surveys).
- **Key mechanism:** An age aggregation step (log-sum-exp over the coarse age group)
  recovers fine-age contact rates from the grouped observations.

**Reference:** Dan et al. (2023), *PLoS Computational Biology*

---

### `GenMixFF` — Generalised Mixing, Fine-Fine

Extends `AgeMixFF` with **hierarchical priors** to estimate contact matrices
stratified by additional features (e.g., gender, household setting). Both participant
and contact ages are at 1-year resolution.

- **Use when:** You want to estimate separate contact matrices per stratum (e.g., by
  gender) while sharing information across strata via a common smooth baseline.
- **Key idea:** Stratum-specific deviations (`δ_s`) are modelled as multiplicative
  adjustments to the shared baseline rate, centred on the stratum's population
  age distribution.

---

### `GenMixFC` — Generalised Mixing, Fine-Coarse

Extends `AgeMixFC` with **hierarchical priors** for stratified populations. Participant
ages are at 1-year resolution and contact ages are given in coarse age groups.

- **Use when:** Your survey has coarse contact age groups *and* you want stratum-specific
  matrices (e.g., by gender or setting).
- **Combines:** The age aggregation mechanism from `AgeMixFC` with the hierarchical
  structure from `GenMixFF`.

---

## Backward-Compatible Aliases

The old BRC-prefixed class names are preserved as aliases for backward compatibility:

| Old name | New name |
|----------|----------|
| `BRC` | `GenMix` (base class) |
| `BRCfine` | `AgeMixFF` |
| `BRCrefine` | `AgeMixFC` |
| `HiBRCfine` | `GenMixFF` |
| `HiBRCrefine` | `GenMixFC` |

These aliases will be maintained indefinitely and are safe to use.

---

## Quick Start

```python
from cntmosaic.dataloader import ContactSurveyLoader
from cntmosaic.models import AgeMixFF
from cntmosaic.models.numpyro.priors import Spline2D
from jax.random import PRNGKey

# Set up dataloader (fine ages for both participant and contact)
dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)

# Define priors
priors = {"rate": Spline2D(prior_type="global", M=30, degree=3)}

# Fit model
model = AgeMixFF(dataloader, priors, likelihood="negbin")
model.run_inference_mcmc(PRNGKey(42), num_samples=1000, num_warmup=1000, num_chains=4)

# Summarise results
from cntmosaic.analysis import ModelSummariserBRC
summariser = ModelSummariserBRC(model)
summary = summariser.summarise_cint(alpha=0.05)
```
