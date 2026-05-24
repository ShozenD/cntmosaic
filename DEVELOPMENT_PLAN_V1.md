# cntmosaic v1.0 Development Plan

This document is the canonical checklist for the v1.0 release. Tasks are grouped into five phases ordered by dependency. Phases 1–2 must be completed before any documentation or PyPI packaging work begins.

Each task lists: **affected files**, **effort** (S = hours, M = 1–2 days, L = 3+ days), and a **brief rationale**.

---

## Phase 1 — Architecture: Blocking Extensibility Issues

These issues directly prevent external researchers from adding new models. They must be resolved first because later refactoring depends on the interfaces introduced here.

### ~~1.1 Define a unified `ContactModel` abstract base class~~ ✅ DONE
- **Files**: `cntmosaic/models/__init__.py`, `_BRC.py`, `_Prem.py`, `_vdKassteele.py`, `_SocialMix.py`
- **Effort**: M
- **Rationale**: `BRC`, `Prem`, `vdKassteele`, and `SocialMix` are completely independent classes with no common interface. A researcher cannot add a new model without reading all four implementations and guessing the expected API. A `ContactModel` ABC should define at minimum `model()`, `run_inference_mcmc()`, `run_inference_svi()`, and `posterior_predictive()`.
- **Notes**: `SocialMix` is deterministic (no MCMC) and must be moved to a new `cntmosaic.models.classical` subpackage. Bayesian models remain in `cntmosaic.models`. A `DeterministicContactModel` ABC should govern the `classical` subpackage. The public `__init__.py` must clearly separate both namespaces so researchers know immediately which interface to implement.

### ~~1.3 Define a `DataContainer` protocol to decouple models from the concrete `DataLoader`~~
- **Files**: `cntmosaic/models/_BRC.py:115`, `_Prem.py:211–215`, `cntmosaic/dataloader/_BaseLoader.py`
- **Effort**: M
- **Rationale**: `BRC` hard-depends on `DataLoader`; `Prem` accepts raw `ParticipantData`/`ContactData`. No researcher can feed a custom data structure to any model without subclassing the wrong thing. A `DataContainer` protocol (providing `.strat_data`, `.contact_counts`, `.participant_counts`) lets both paths converge and allows custom data pipelines.

### ~~1.4 Standardise prior construction across all models~~ ✅ DONE
- **Files**: `cntmosaic/models/_vdKassteele.py:106–154`, `_BRCfine.py:150–172`, `_Prem.py`
- **Effort**: S
- **Rationale**: `BRCfine` receives pre-constructed priors; `vdKassteele` builds them internally in `_set_prior()`. This inconsistency forces researchers to read each model's source to understand prior injection. Preferred pattern: all models receive constructed priors as a `Dict[str, Prior2D]` argument.

### ~~1.5 Fix `SymIGMRF2D` JAX/NumPy type mismatch (GPU correctness bug)~~ ✅ DONE
- **Files**: `cntmosaic/distributions/_SymIGMRF2D.py:429, 583`
- **Effort**: S
- **Rationale**: `self.L = L1 + L2` (line 429) stores a NumPy array, which is then used in JAX computation at line 583. This silently works on CPU but will fail under JIT compilation or on GPU. Fix: `self.L = jnp.asarray(L1 + L2)`.

### ~~1.6 Fix phantom exports in `sim/__init__.py`~~ ✅ DONE
- **Files**: `cntmosaic/sim/__init__.py:8–17`
- **Effort**: S
- **Rationale**: `__all__` exports `ModelEvaluatorSVI` and `ModelEvaluatorMCMC`, which do not exist in the `sim` module. This causes an `ImportError` for any user who does `from cntmosaic.sim import *`.

### ~~1.7 Decouple the inference engine behind a pluggable `InferenceBackend` protocol~~ ✅ DONE *(consolidates former item 1.2)*

**Goal**: Remove all top-level `import numpyro` from model files so that importing `cntmosaic.models` does not require NumPyro to be installed. Eliminate ~300 lines of duplicated inference boilerplate across `_BRC.py`, `_Prem.py`, and `_vdKassteele.py` by introducing a single `NumPyroBackend` that all concrete models delegate to.

---

### ~~1.8 Decouple the post-processing pipeline (`analysis/`) from NumPyro~~ ✅ DONE

**Goal**: Remove all top-level `import numpyro` from `analysis/_arviz.py` so that `import cntmosaic.analysis` does not require NumPyro to be installed. Eliminate the `BRC`-specific type annotation and the private `numpyro.infer.util._predictive` call in the same file. Unify the scattered `has_mcmc` / `has_svi` branching in both summarisers by introducing a `ContactModel.get_posterior_samples()` convenience method that hides all inference-method detection and guide handling.

**Effort**: S–M (2 stages, separable into two PRs)

---

## Phase 2 — Design Refactoring: Maintainability and SOLID Compliance

### ~~2.1 Redesign the `dataloader` pipeline (replaces BaseLoader + DataLoader)~~ ✅ DONE
- **Files**: Entire `cntmosaic/dataloader/` module
- **Effort**: L
- **Rationale**: `BaseLoader` (~744 lines) performs data validation, column mapping, stratification inference, Cartesian grid construction, population log-proportion computation, caching, and model data assembly — six unrelated concerns in one class. The coupling makes every concern untestable in isolation and blocks external survey support without subclassing the entire class.

### ~~2.2 Fix circular imports blocking type hints in `_DataLoader.py`~~ ✅ DONE
- **Files**: `cntmosaic/dataloader/_DataLoader.py:172–176`
- **Effort**: S
- **Rationale**: Type hints for `part_data`, `cnt_data`, `pop_data` are removed with a comment about circular imports. Use `from __future__ import annotations` or a `TYPE_CHECKING` guard to restore the annotations without the import cycle. This is a quick prerequisite for 2.1 Stage 4.

### ~~2.3 Create `BaseModelEvaluator` abstract class~~ ✅ DONE
- **Files**: `cntmosaic/analysis/evaluator/_ModelEvaluatorBRC.py`, `_ModelEvaluatorPrem.py`, `_ModelEvaluatorSocialMix.py`
- **Effort**: M
- **Rationale**: All three evaluators duplicate `validate_alpha()`, `interval_score()`, and `compute_metrics()`. A `BaseModelEvaluator` ABC housing shared logic allows evaluators for new models to be added by only implementing model-specific overrides. Evaluators should also accept a `ModelSummariser` protocol rather than concrete summariser classes (current tight coupling via direct import).

### ~~2.4 Delete `QuasiPoisson` and `QuasiNegBin`~~ ✅ DONE
- **Files**: `cntmosaic/distributions/_QuasiPoisson.py`, `_QuasiNegBin.py`, `cntmosaic/distributions/__init__.py`
- **Effort**: S
- **Rationale**: These distributions are no longer used anywhere in the codebase. Shipping dead code with silent TODO stubs in `sample()` as part of a v1.0 PyPI release creates a misleading API surface and maintenance burden. Delete both files, remove them from `distributions/__init__.py`, and confirm no remaining imports exist (`grep -r "QuasiPoisson\|QuasiNegBin" cntmosaic/`).

### 2.5 Replace string-based matrix type detection in `ContactGenerator`
- **Files**: `cntmosaic/sim/_ContactGenerator.py:214–367`
- **Effort**: M
- **Rationale**: `_validate_matrices()` infers matrix structure by parsing key strings like `"Urban->All"`. Adding new matrix types requires modifying validation logic. Replace with a `MatrixStructure` enum or dataclass so the structure is explicit and extensible.

### 2.6 Define a `Prior2D` `Protocol` with `@runtime_checkable`
- **Files**: `cntmosaic/models/priors/_Prior2D.py`
- **Effort**: S
- **Rationale**: The current `Prior2D` ABC exists but has loose `Union[int, float, np.ndarray, jnp.ndarray]` parameter types and no runtime-checkable contract. Defining a `@runtime_checkable Protocol` lets models do `isinstance(prior, Prior2D)` at construction time, giving researchers immediate feedback when they pass a malformed prior.

---

## Phase 3 — API Polish: Production-Quality Public Surface

### ~~3.1 Audit and fix all `__init__.py` public exports~~ ✅ DONE
- **Files**: `cntmosaic/__init__.py`, all submodule `__init__.py` files
- **Effort**: S
- **Rationale**: Root `__all__` omits `dataloader` and `datasets` — the primary entry points for new users. Audit every `__all__` list against actual definitions to eliminate phantom exports and expose the intended public API. `DataLoader` should be importable as `from cntmosaic import DataLoader`.

### ~~3.2 Rename cryptic `BaseLoader` property names~~ ✅ DONE
- **Files**: `cntmosaic/dataloader/_BaseLoader.py:218–285`
- **Effort**: S
- **Rationale**: `df_V` (contact offsets), `df_y` (contact counts), and `df_n` (participant counts) are mathematically motivated names that are opaque to users unfamiliar with the underlying model notation. Rename to `df_contact_offsets`, `df_contact_counts`, `df_participant_counts`.

### 3.3 Add comprehensive type annotations (Python 3.8-compatible)
- **Files**: `cntmosaic/dataloader/`, `cntmosaic/datasets/_base.py`, `cntmosaic/preprocess/_preprocess.py`, `cntmosaic/utils/_AgeBins.py`
- **Effort**: M
- **Rationale**: Many public methods lack return type hints. Several files use Python 3.9+ syntax (`list[str]`, `X | Y`) inconsistent with the declared `python_requires = ">=3.8"`. Standardise to `from typing import List, Union, Optional, Dict` throughout, or bump minimum Python to 3.10.

### ~~3.4 Define typed return structures for `datasets` loaders~~ ✅ DONE
- **Files**: `cntmosaic/datasets/_base.py`
- **Effort**: S
- **Rationale**: `load_template_patterns()` returns an untyped `dict`. Define a `TypedDict` (e.g. `ContactPatterns`) with keys `household`, `school`, `work`, `other` so IDEs and type checkers can validate downstream usage.

### ~~3.5 Consolidate `plot_mosaic` parameter list~~ ✅ DONE
- **Files**: `cntmosaic/vis/_visuals.py`
- **Effort**: S
- **Rationale**: `plot_mosaic()` takes 20+ parameters, making it unusable without consulting docs. Group related parameters into a `MosaicPlotConfig` dataclass with sensible defaults.

### ~~3.6 Remove Altair global state mutation on import~~ ✅ DONE
- **Files**: `cntmosaic/vis/_visuals.py:9`
- **Effort**: S
- **Rationale**: `alt.data_transformers.disable_max_rows()` is called at module import time, silently mutating global Altair state for any user who imports `cntmosaic.vis`. Either remove it, call it lazily inside plot functions, or expose it as `cntmosaic.vis.configure(disable_max_rows=True)`.

### ~~3.7 Define `StratumLabel` type and use consistently across `sim`~~ ✅ DONE
- **Files**: `cntmosaic/sim/_ContactGenerator.py`, `_MatrixGenerator.py`
- **Effort**: S
- **Rationale**: `ContactGenerator` uses string keys (`"Urban->All"`); `MatrixGenerator` uses tuple indices `(s, t)`. There is no unified abstraction. Define `StratumLabel = Union[str, Tuple[str, str]]` in `cntmosaic/_types.py` and adopt it consistently.

### ~~3.8 Clarify internal vs. public class boundaries~~ ✅ DONE
- **Files**: `cntmosaic/dataloader/_BaseLoader.py`, `_CoordToColumns.py`
- **Effort**: S
- **Rationale**: Classes prefixed with `_` imply internal-only, yet `BaseLoader` and `CoordToColumns` are referenced in docstrings as extension points. Rename them (drop `_` prefix) and commit to their stability, or explicitly mark them as private in the module docstring.

---

## Phase 4 — Test Suite: Fixing and Expanding Coverage

### 4.1 Fix broken test imports (release blocker)
- **Files**: `cntmosaic/analysis/tests/test_evaluator_brc.py:39, 56`; `cntmosaic/preprocess/tests/test_make_train_data.py:57, 69`
- **Effort**: S
- **Rationale**: `test_evaluator_brc.py` references undefined `Subgroup` and `CoordToColumns`; test fixtures cannot instantiate. `test_make_train_data.py` uses a wrong `RuntimeWarning` regex pattern that will never match. Both files must pass before CI can be trusted.

### 4.2 Rewrite `sim` test skeletons
- **Files**: `cntmosaic/sim/tests/`
- **Effort**: M
- **Rationale**: `ContactGenerator`, `MatrixGenerator`, `ParticipantGenerator`, and `PopulationConstructor` have skeleton test files with fixtures but no assertions. Write tests covering: correct output shapes, data validity (non-negative counts), random seed reproducibility, and generate_full vs generate_partial paths.

### 4.3 Rewrite `distributions` test skeletons
- **Files**: `cntmosaic/distributions/tests/test_IGMRF.py`
- **Effort**: M
- **Rationale**: `test_IGMRF.py` is a skeleton with no assertions. Tests should verify: `log_prob()` returns correct scalar shape, values are finite for valid inputs, JAX JIT compilation succeeds, and (once implemented) `sample()` produces correctly shaped output.

### 4.4 Add model integration tests (BRC, Prem, vdKassteele)
- **Files**: `cntmosaic/models/tests/`
- **Effort**: L
- **Rationale**: No tests verify end-to-end model execution with real or simulated data. Add integration tests that: instantiate each model with a small synthetic dataset, run a short MCMC chain (100 samples), and assert posterior shape and basic diagnostics (no NaNs, R-hat computable).

### 4.5 Add missing preprocessor unit tests
- **Files**: `cntmosaic/preprocess/tests/`
- **Effort**: M
- **Rationale**: `test_make_full_grid.py`, `test_expand_age_interval.py`, and `test_add_grp_cnt_offsets.py` are either missing or empty. These functions are critical data pipeline steps; they need coverage for both happy paths and edge cases (empty input, mismatched age bins).

### 4.6 Strengthen `datasets` tests
- **Files**: `cntmosaic/datasets/tests/test_load_covimod.py`, `test_load_polymod_germany.py`
- **Effort**: S
- **Rationale**: `test_load_covimod.py` only checks that dictionary keys exist — it passes even if all DataFrames are empty. Tests should validate data shapes, non-negative counts, and consistent column schema. `test_load_polymod_germany.py` is missing entirely.

### 4.7 Add `dataloader` container and pipeline tests
- **Files**: `cntmosaic/dataloader/tests/`
- **Effort**: M
- **Rationale**: `ModelData` class has no tests. Stratification mode detection logic (partial vs full) is untested. Add unit tests for container construction and property access, and an integration test for the full `DataLoader.load()` pipeline.

---

## Phase 5 — Documentation

### 5.1 Write contributor guide for adding new models
- **Files**: `docs/` (new page)
- **Effort**: M
- **Rationale**: The primary value proposition for the community is extensibility. A short guide showing how to implement `ContactModel`, provide a `DataContainer`, and register a prior is the most important documentation missing for v1.0.

### 5.2 Add module-level docstrings to all `__init__.py` files
- **Files**: All submodule `__init__.py` files, especially `cntmosaic/analysis/evaluator/__init__.py`
- **Effort**: S
- **Rationale**: Several submodules (especially `analysis/evaluator/`) have no module-level docstring explaining their purpose or relationship to sibling modules.

### 5.3 Update API reference docs
- **Files**: `docs/`
- **Effort**: M
- **Rationale**: Docs are described as out-of-date. Once Phase 1–3 refactoring is complete, regenerate the Sphinx autodoc output and review all hand-written pages against the new public API surface.

### 5.4 Add quickstart tutorial
- **Files**: `tutorials/` (new notebook)
- **Effort**: M
- **Rationale**: There is currently no end-to-end example showing a new user how to: load data → preprocess → fit a model → inspect posteriors → visualise. This is the first thing any potential user or reviewer will look for.

---

## Priority Summary

| Phase            | Blockers for v1.0                                             | Owner suggestion                             |
|------------------|---------------------------------------------------------------|----------------------------------------------|
| 1 — Architecture | ✅ All items are release blockers                              | Core maintainer (you + co-author)            |
| 2 — Design       | Items 2.1, 2.4, 2.6 are important; rest can slip to v1.1      | Contributor familiar with statistical models |
| 3 — API Polish   | 3.1 (exports), 3.6 (Altair), 3.3 (types) are release blockers | Any contributor                              |
| 4 — Tests        | 4.1 is an immediate blocker; 4.4 required before PyPI         | Any contributor; 4.4 needs domain knowledge  |
| 5 — Docs         | 5.1 (contributor guide) critical for community adoption       | Core maintainer                              |

**Minimum viable set for a credible v1.0 PyPI release**: Phase 1 complete + Phase 3.1/3.3/3.6 + Phase 4.1 + Phase 5.1/5.4.
