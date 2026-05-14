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

### 1.3 Define a `DataContainer` protocol to decouple models from the concrete `DataLoader`
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

**Affected files**:
- New: `cntmosaic/models/_backend.py` — `InferenceBackend` Protocol
- New: `cntmosaic/models/numpyro/__init__.py`, `cntmosaic/models/numpyro/_backend.py` — `NumPyroBackend` class
- New: `cntmosaic/models/numpyro/_BRCfine.py`, `_BRCrefine.py`, `_HiBRCfine.py`, `_HiBRCrefine.py`, `_Prem.py`, `_vdKassteele.py` — NumPyro model mixins
- Modified: `cntmosaic/models/_base.py`, `_BRC.py`, `_BRCfine.py`, `_BRCrefine.py`, `_HiBRCfine.py`, `_HiBRCrefine.py`, `_Prem.py`, `_vdKassteele.py`, `_numpyro.py`, `__init__.py`

**Effort**: L (3–5 days; split into two sequential PRs as described below)

**Rationale**: Every model hard-codes NumPyro at module level and duplicates `run_inference_mcmc`, `run_inference_svi`, `posterior_predictive_mcmc`, `posterior_predictive_svi`, `_log_mcmc_diagnostics`, and `print_model_shape` across `_BRC.py:393–778`, `_Prem.py:795–1024`, and `_vdKassteele.py:244–690`. Both problems — tight coupling and code duplication — are resolved together in two sequential PRs.

---

#### ~~Stage 1~~ ✅ — Introduce `InferenceBackend` Protocol (no behaviour change)
*PR-a: strictly additive, zero behaviour change, all existing tests must pass unchanged.*

**New file `cntmosaic/models/_backend.py`**:

```python
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

@runtime_checkable
class InferenceBackend(Protocol):
    def run_mcmc(
        self,
        model: Callable,
        prng_key: Any,
        *,
        num_samples: int,
        num_warmup: int,
        num_chains: int,
        target_accept_prob: float,
        max_tree_depth: int,
        **model_kwargs: Any,
    ) -> Any: ...

    def run_svi(
        self,
        model: Callable,
        guide: Callable,
        prng_key: Any,
        *,
        num_steps: int,
        peak_lr: float,
        **model_kwargs: Any,
    ) -> Any: ...

    def get_mcmc_samples(self, mcmc_result: Any) -> Dict[str, Any]: ...

    def get_mcmc_extra_fields(self, mcmc_result: Any) -> Dict[str, Any]: ...

    def get_svi_params(self, svi_result: Any) -> Dict[str, Any]: ...

    def get_svi_samples(
        self,
        prng_key: Any,
        guide: Callable,
        svi_result: Any,
        num_samples: int,
        **guide_kwargs: Any,
    ) -> Dict[str, Any]: ...

    def posterior_predictive_mcmc(
        self,
        prng_key: Any,
        model: Callable,
        mcmc_result: Any,
        **model_kwargs: Any,
    ) -> Dict[str, Any]: ...

    def posterior_predictive_svi(
        self,
        prng_key: Any,
        model: Callable,
        guide: Callable,
        svi_result: Any,
        num_samples: int,
        **model_kwargs: Any,
    ) -> Dict[str, Any]: ...
```

**Design notes for Stage 1**:
- Use `@runtime_checkable Protocol`, not an ABC. This keeps the contract structural (duck-typed) so a future PyMC backend does not need to import this file at all — it just needs to satisfy the interface.
- `format_model_shapes` / `print_model_shape` is intentionally **excluded** from the Protocol. It is a NumPyro diagnostic utility (`numpyro.util.format_shapes`), not an inference operation. It belongs on `NumPyroBackend` as a concrete method, not on the protocol surface. The `print_model_shape()` method on model classes will call it directly on `NumPyroBackend` (accessed via `self._backend`) rather than through the protocol.
- `get_mcmc_samples`, `get_mcmc_extra_fields`, `get_svi_params`, `get_svi_samples` are explicit methods rather than having the caller reach into the opaque result object. This is the key encapsulation boundary — the `analysis/` layer and summarisers currently call `model._mcmc_result.get_samples()` directly; after Stage 4 they will call `model._backend.get_mcmc_samples(model._mcmc_result)`. The raw `_mcmc_result` attribute remains on the model (as `Optional[Any]`) so `analysis/_arviz.py` and `_ModelSummariserBRC/Prem.py` do not break during transition.

---

#### ~~Stage 2~~ ✅ — Create `NumPyroBackend` in `models/numpyro/`
*Still part of PR-a. All existing code paths continue to work; `_numpyro.py` free functions are not deleted.*

**New `cntmosaic/models/numpyro/_backend.py`**:
- `NumPyroBackend` class, implementing `InferenceBackend`.
- All methods delegate to the existing free functions in `cntmosaic/models/_numpyro.py` — no logic is duplicated or moved yet.
- `print_model_shape(model_fn: Callable) -> None` as an extra concrete method (not on the Protocol).
- `_build_default_guide(model_fn: Callable) -> Callable` as an extra concrete method returning `AutoNormal(model_fn)` — this is where `Prem` and `vdKassteele`'s default-guide construction will delegate (Stage 4).

**`cntmosaic/models/numpyro/__init__.py`** — exports `NumPyroBackend` only. Add a docstring explaining how to add a non-NumPyro backend (the doc stub requested in Stage 6 of the original plan is merged here).

**`cntmosaic/models/_numpyro.py`** is **not touched** in PR-a. It remains the implementation; `NumPyroBackend` wraps it.

---

#### ~~Stage 3~~ ✅ — Extract NumPyro model mixins
*Still part of PR-a. Adds new mixin files; does not modify any existing model files.*

**New files** in `cntmosaic/models/numpyro/`:
- `_BRCfine.py`, `_BRCrefine.py`, `_HiBRCfine.py`, `_HiBRCrefine.py`, `_Prem.py`, `_vdKassteele.py`

Each contains one `*NumPyroMixin` class with a single `model(self, y=None)` method whose body is copied verbatim from the current concrete model class. No other methods. No imports of `_base.py` or `ContactModel` — the mixin is a pure behaviour carrier.

**HiBRC special case**: `sample_log_delta()` uses `numpyro.deterministic` and `scope`. Move it into `HiBRCfineNumPyroMixin` / `HiBRCrefineNumPyroMixin` alongside `model()`. The abstract `sample_log_delta()` declaration in `_BRC.py` stays; the mixin satisfies it.

**Why mixins rather than composition or strategy objects?**  
A strategy/composition approach (e.g., `self._model_fn = NumPyroModelStrategy()`) would require the `ContactModel` interface to expose a `model_fn` callable rather than a `model()` method, which breaks the existing API used by NumPyro's `NUTS(model.model)`, `AutoNormal(model.model)`, and `svi_to_inference_data`. The mixin pattern adds no indirection to the public API — `model.model` still resolves directly to a bound method — and requires no changes to `analysis/` code.

---

#### ~~Stage 4~~ ✅ — Thread `NumPyroBackend` into model constructors and delete boilerplate
*PR-b: the deletion pass. Stages 1–3 (PR-a) must be merged first.*

**`ContactModel._base.py`** gains:
- `__init__(self, backend: Optional[InferenceBackend] = None)` with lazy default: `self._backend: Optional[InferenceBackend] = backend`
- `_get_backend() -> InferenceBackend`: returns `self._backend` if set, otherwise does `from .numpyro._backend import NumPyroBackend; self._backend = NumPyroBackend(); return self._backend` (lazy import — this is the key guard that prevents NumPyro from loading at model-class import time).
- Concrete implementations of `run_inference_mcmc`, `run_inference_svi`, `posterior_predictive_mcmc`, `posterior_predictive_svi` — all delegating through `self._get_backend()`. These replace all three duplicated copies in `_BRC.py`, `_Prem.py`, `_vdKassteele.py`.
- `print_model_shape()` — calls `self._get_backend().print_model_shape(self.model)` (concrete `NumPyroBackend` method, not on the Protocol).
- `_log_mcmc_diagnostics()` — calls `self._get_backend().get_mcmc_extra_fields(self._mcmc_result)` to extract divergence info.

**Each concrete model class** (`BRC`, `Prem`, `vdKassteele`):
- Constructor gains `backend: Optional[InferenceBackend] = None` parameter and calls `super().__init__(backend=backend)`.
- Inherits the new mixin: `class BRCfine(BRCfineNumPyroMixin, BRC)` — MRO places the mixin's `model()` first; the abstract `model()` in `ContactModel` is satisfied.
- `run_inference_mcmc`, `run_inference_svi`, `posterior_predictive_mcmc`, `posterior_predictive_svi`, `print_model_shape`, `_log_mcmc_diagnostics` are **deleted** from `_BRC.py`, `_Prem.py`, `_vdKassteele.py`.

**`_mcmc_result` / `_svi_result` / `_guide` attributes**: move to `ContactModel.__init__` as `Optional[Any]`. Type annotations change from `Optional[numpyro.infer.MCMC]` / `Optional[numpyro.infer.SVI]` to `Optional[Any]`. This is safe: `analysis/summariser/_ModelSummariserBRC.py:216` calls `model._mcmc_result.get_samples()` and `analysis/_arviz.py:34` accesses `model._svi_result.params` and `model._guide` — all of these attribute names are preserved and the duck-typed `Optional[Any]` annotation does not break runtime behaviour. The only change is that mypy loses the specific type; a `# type: ignore` comment is acceptable here during transition.

**`Prem.get_samples_svi`**: This method exists on `Prem` but not on `vdKassteele` or `BRC`, and it was omitted from the original Protocol definition. It stays as a **concrete method on `Prem`** (not on the Protocol or `ContactModel`), delegating to `self._get_backend().get_svi_samples(...)`. The `InferenceBackend` Protocol does include `get_svi_samples` (see Stage 1 above), so `NumPyroBackend` already implements the underlying operation. Prem-specific callers use `model.get_samples_svi(...)` directly — this is intentional as not all models need this method.

**`Prem` and `vdKassteele` default-guide construction**: Both currently auto-construct `AutoNormal(self.model)` when `guide=None` in `run_inference_svi`. In the unified `ContactModel.run_inference_svi`, the default is `None` and the backend provides `_build_default_guide(model_fn)` which returns `AutoNormal(model_fn)`. This is called lazily inside `run_inference_svi` when guide is `None`. BRC subclasses do not use a default guide (they require the caller to provide one), but since `BRC.run_inference_svi` inherits from `ContactModel` and the default-guide code path simply short-circuits if guide is provided, this introduces no behavioural change for BRC. The `guide=None` default previously absent from BRC's `run_inference_svi` signature is preserved at the `ContactModel` level — the check `if guide is None: raise ValueError(...)` can be added in `BRC.__init__` if BRC-family models must require an explicit guide; alternatively, a `_requires_explicit_guide: bool = False` class variable governs this.

---

#### ~~Stage 5~~ ✅ — Clean up `models/__init__.py` and verify zero top-level numpyro imports
*Merged into PR-b (not a separate stage).*

**`models/__init__.py`** currently imports `to_inference_data` from `_numpyro.py`, which pulls in NumPyro at `cntmosaic.models` import time. Fix: make `to_inference_data` a lazy import using `__getattr__`:

```python
# In models/__init__.py — replace the direct import:
# from ._numpyro import to_inference_data   ← REMOVE

def __getattr__(name: str):
    if name == "to_inference_data":
        from ._numpyro import to_inference_data
        return to_inference_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

This preserves the public export in `__all__` while deferring the import until first use. Alternatively — and more explicitly — `to_inference_data` can be moved to `cntmosaic.models.numpyro` and the entry in `models/__init__.__all__` can be removed; this is a minor API break and should be noted in the changelog.

**Verification test** (add to `cntmosaic/models/tests/test_lazy_imports.py`):
```python
def test_models_import_does_not_load_numpyro():
    import importlib, sys
    for mod in list(sys.modules):
        if "numpyro" in mod:
            del sys.modules[mod]
    importlib.import_module("cntmosaic.models")
    assert "numpyro" not in sys.modules, "numpyro was imported at cntmosaic.models import time"
```

---

#### PR split and sequencing

| PR | Stages | Description | Risk |
|---|---|---|---|
| PR-a | 1, 2, 3 | Add `InferenceBackend` Protocol, `NumPyroBackend`, and six NumPyro mixins — purely additive | Low |
| PR-b | 4, 5 | Thread backend into constructors, delete ~300 lines of boilerplate, clean up lazy imports | Medium |

Stages 3 and 4 are **not** parallelisable: Stage 4 deletes the `model()` body from each concrete class and relies on the mixin from Stage 3 to satisfy the abstract method. PR-a must be merged and CI green before PR-b begins.

---

#### What is explicitly out of scope for this item

- `analysis/_arviz.py` and `analysis/summariser/` are **not** changed. They access `model._mcmc_result`, `model._svi_result`, `model._guide`, and call `model._mcmc_result.get_samples()` — all of which remain structurally identical after this refactor.
- `_numpyro.py` free functions are preserved; `NumPyroBackend` wraps them without inlining or moving logic.
- No changes to the `ContactModel.model()` abstract method signature — it must remain `def model(self, y=None)` for NumPyro's handler machinery and for the existing `analysis/` code that calls `model.model`.
- PyMC or INLA backend implementations are out of scope; this item only makes them *possible*.

### ~~1.8 Decouple the post-processing pipeline (`analysis/`) from NumPyro~~ ✅ DONE

**Goal**: Remove all top-level `import numpyro` from `analysis/_arviz.py` so that `import cntmosaic.analysis` does not require NumPyro to be installed. Eliminate the `BRC`-specific type annotation and the private `numpyro.infer.util._predictive` call in the same file. Unify the scattered `has_mcmc` / `has_svi` branching in both summarisers by introducing a `ContactModel.get_posterior_samples()` convenience method that hides all inference-method detection and guide handling.

**Affected files**:
- `cntmosaic/analysis/_arviz.py` — remove top-level numpyro imports; replace private `_predictive` with public `Predictive`; broaden type annotation; defer numpyro calls to lazy import inside function body
- `cntmosaic/models/_base.py` — add `inference_method()` and `get_posterior_samples()` to `ContactModel`
- `cntmosaic/models/_BRC.py` — drop `guide` parameter from `posterior_predictive_svi` abstract override (store `self._guide` implicitly, already done at storage site); update `posterior_predictive_svi` to use `self._guide` internally
- `cntmosaic/models/_vdKassteele.py` — same as `_BRC.py`
- `cntmosaic/models/_Prem.py` — already omits `guide` parameter; no change needed beyond inheriting `get_posterior_samples`
- `cntmosaic/analysis/summariser/_ModelSummariserBRC.py` — replace `hasattr`/`_mcmc_result`/`_guide` access with `model.inference_method()` and `model.get_posterior_samples()`
- `cntmosaic/analysis/summariser/_ModelSummariserPrem.py` — same

**Effort**: S–M (2 stages, separable into two PRs)

**Rationale**: `analysis/__init__.py` imports `_arviz.py` at the top level, which unconditionally loads NumPyro (`import numpyro`, `from numpyro.handlers import substitute`, `from numpyro.infer import log_likelihood`, `from numpyro.infer.util import _predictive`). This means any user who runs `import cntmosaic.analysis` installs a hard NumPyro dependency even if they only want summariser utilities. After item 1.7 establishes the `InferenceBackend` protocol on the model side, this item applies the equivalent separation to the analysis side. It also removes a private-API call (`_predictive`) that will break without warning on any NumPyro version bump.

---

#### Stage A — Replace private NumPyro internals in `analysis/_arviz.py` and align `posterior_predictive_svi` signatures

*Strictly additive/corrective; no behaviour change. Can ship as its own PR.*

**`analysis/_arviz.py`**:
- Replace `from numpyro.infer.util import _predictive` (private) with the public `Predictive` class: `from numpyro.infer import Predictive`. Rewrite the two `_predictive(...)` calls using `Predictive(model_or_guide, posterior_samples, ...).call(prng_key, ...)`. The existing xarray assembly logic (manual `dict_to_dataset` for posterior, log_likelihood, posterior_predictive, and observed_data groups) must be **preserved** — `arviz.from_numpyro` is **not** a valid replacement here because it expects a fitted `MCMC` or `SVI` object, not raw sample dicts, and does not produce the posterior-predictive and observed-data groups that the current implementation assembles.
- Broaden the type annotation: `model: BRC` → `model: ContactModel` (import from `..models._base`).
- Do **not** remove the top-level numpyro imports yet (that is Stage B); the goal of Stage A is correctness only.

**`cntmosaic/models/_base.py`**:
- Add `inference_method(self) -> Optional[Literal["mcmc", "svi"]]` as a concrete method: returns `"mcmc"` if `getattr(self, "_mcmc_result", None) is not None`, `"svi"` if `getattr(self, "_svi_result", None) is not None`, else `None`.

**`posterior_predictive_svi` signature alignment** (prerequisite for Stage B):
- Remove the explicit `guide: Callable` parameter from `ContactModel.posterior_predictive_svi`, `BRC.posterior_predictive_svi`, and `vdKassteele.posterior_predictive_svi`. All three already store `self._guide` at `run_inference_svi` call time (lines 598 in `_BRC.py`, 514 in `_vdKassteele.py`). `Prem.posterior_predictive_svi` already omits `guide` — this change brings BRC and vdKassteele into alignment with Prem's existing pattern. No overrides are needed after this: the default `get_posterior_samples` on `ContactModel` can call `self.posterior_predictive_svi(prng_key, num_samples)` uniformly.
- Add `ContactModel.get_posterior_samples(self, prng_key, num_samples) -> Dict[str, Any]` as a concrete method: calls `self._mcmc_result.get_samples()` for MCMC or `self.posterior_predictive_svi(prng_key, num_samples)` for SVI, guarded by `self.inference_method()`.

---

#### Stage B — Lazy-import all numpyro references in `analysis/_arviz.py`; update summarisers

*Behaviour change: `import cntmosaic.analysis` no longer loads NumPyro. PR-b.*

**`analysis/_arviz.py`**:
- Move all top-level numpyro imports (`import numpyro`, `from numpyro.handlers import substitute`, `from numpyro.infer import log_likelihood, Predictive`, `from arviz import dict_to_dataset`) inside the `svi_to_inference_data` function body. The module-level imports reduce to `from typing import Dict, Optional` and `import numpy as np`.
- The `from ..models._base import ContactModel` annotation import can remain at module level (it carries no NumPyro dependency).

**`analysis/summariser/_ModelSummariserBRC.py`** — `__init__` and `_load_posterior`:
- Replace the `hasattr(model, "_mcmc_result") and model._mcmc_result is not None` / `hasattr(model, "_svi_result") ...` pair with a single `model.inference_method()` call.
- Replace `_load_posterior`'s branching (`model._mcmc_result.get_samples()` / `model.posterior_predictive_svi(PRNGKey(0), model._guide, ...)`) with `model.get_posterior_samples(PRNGKey(0), self.num_samples)`.
- Store `self.inference_method: Literal["mcmc", "svi"] = model.inference_method()`.

**`analysis/summariser/_ModelSummariserPrem.py`** — same structural change as BRC summariser.

**Verification test** (add to `cntmosaic/analysis/tests/test_lazy_imports.py`):
```python
def test_analysis_import_does_not_load_numpyro():
    import importlib, sys
    for mod in list(sys.modules):
        if "numpyro" in mod:
            del sys.modules[mod]
    importlib.import_module("cntmosaic.analysis")
    assert "numpyro" not in sys.modules, \
        "numpyro was imported at cntmosaic.analysis import time"
```

---

#### What is explicitly out of scope for this item

- `NumPyroSVIConverter` class and `to_inference_data` free function in `models/_numpyro.py` are not touched (they are already in the right module and carry no `analysis/` coupling).
- Moving `svi_to_inference_data` logic to the `InferenceBackend` or `NumPyroBackend` is deferred. The function stays in `analysis/_arviz.py` as a NumPyro-specific utility; the lazy import is sufficient to break the unconditional coupling.
- The sample-site names `log_cint`, `log_rate`, `log_delta` used in `_ModelSummariserBRC._compute_contact_intensities` are NumPyro-specific. A PyMC backend would produce different site names. A `sample_site_map: Dict[str, str]` attribute on `ModelSummariserBRC` (defaulting to current NumPyro names) is the intended future extension point but is **not** implemented here — it is only relevant once a second backend exists.

---

## Phase 2 — Design Refactoring: Maintainability and SOLID Compliance

### ~~2.1 Redesign the `dataloader` pipeline (replaces BaseLoader + DataLoader)~~ ✅ DONE
- **Files**: Entire `cntmosaic/dataloader/` module
- **Effort**: L
- **Rationale**: `BaseLoader` (~744 lines) performs data validation, column mapping, stratification inference, Cartesian grid construction, population log-proportion computation, caching, and model data assembly — six unrelated concerns in one class. The coupling makes every concern untestable in isolation and blocks external survey support without subclassing the entire class.

**Final state** (all 5 stages complete, 2026-05-13):
- Stage 1: `DataValidator.validate()` returns validated container copies; no mutation of caller's objects.
- Stage 2: `_observation.py` — six free functions (`build_participant_counts`, `build_contact_offsets`, `build_contact_counts`, `build_observation_grid`, `construct_log_P`, `align_age_range`); `BaseLoader` delegates.
- Stage 3: `_stratification.py` — eight free functions (`infer_strat_modes`, `infer_strat_dims`, `infer_strat_labels`, `infer_full_strat_labels`, `infer_strat_ixs`, `infer_strat_pixs`, `make_flat_ix`, `assemble_strat_kwargs`); `BaseLoader` delegates.
- Stage 4: `CoordToColumns` frozen (`frozen=True`) + `from_containers` classmethod + `ColumnSpec` alias; `DataFrameSurveySource` bundles validated containers, merged data, and col spec.
- Stage 5: `ContactSurveyLoader` with `from_containers()` factory; exports (`ColumnSpec`, `ContactSurveyLoader`, `DataFrameSurveySource`) added to `cntmosaic.dataloader.__init__`.

**Refined architecture** (synthesised from design, critical, and UX review):

The following components replace `BaseLoader` + `DataLoader`. Abstractions are introduced only where real polymorphism exists; everything else becomes a free function or a refactored method returning immutable values.

| Component | Type | Replaces | Responsibility |
|---|---|---|---|
| `SurveySource` | `Protocol` | Implicit `DataLoader` contract | Exposes validated containers: `participants → ParticipantData`, `contacts → ContactData`, `population → PopulationData`, `strat_vars → list[str]`. Exposes *containers*, not raw DataFrames — preserves type safety. |
| `DataFrameSurveySource` | concrete class | `DataLoader.__init__` | Default `SurveySource` impl: wraps three raw DataFrames + column kwargs. This is what 90 % of researchers use without knowing the Protocol exists. |
| `ColumnSpec` | frozen dataclass | `CoordToColumns` | Carries all column-name bindings; no behaviour. Assembled internally via `ColumnSpec.from_containers(part_data, cnt_data, pop_data)` — researchers never construct it directly. |
| `DataValidator` (refactored) | concrete class | `DataValidator` (mutating) | Returns normalised container *copies* rather than mutating inputs in-place. No new ABC — just change the return type and remove the side effects. |
| `_observation.py` | module of free functions | `BaseLoader._build_df_full`, `df_participant_counts`, `df_contact_counts`, `df_contact_offsets` | `build_observation_grid(merged, col_spec, age_min, age_max) → pd.DataFrame`. Free function, not a class — no `ObservationBuilder` ABC (there will never be a second implementation needing substitution). |
| `_stratification.py` | module of free functions | `BaseLoader.infer_strat_*` (7 methods) | Keep the existing seven functions as module-level free functions. They are already modular with clear inputs/outputs. No `StratificationAssembler` class needed. |
| `ContactSurveyLoader` | concrete class | `BaseLoader` orchestration | Thin orchestrator: accepts a `SurveySource` and calls the above free functions in sequence. Primary entry point is a class method (see below). |

**Primary public API** (what researchers actually call):
```python
# 90 % case — identical mental model to today
loader = ContactSurveyLoader.from_dataframes(
    participants_df, contacts_df, population_df,
    id_col="id", part_age_col="age", cnt_age_col="age",
    pop_age_col="age", pop_size_col="N",
    strat_var_cols=["setting"],
    smooth_amb_cnt_offsets=True,   # must survive migration
)
model_data = loader.load()   # returns ModelData — contract unchanged
```

**Non-negotiable UX contracts that must be preserved:**
- Per-container column specification (`id_col`, `age_col`, `strat_var_cols`) as keyword arguments
- Automatic categorical dtype handling for stratification variables after merge
- `strat_prop_data` as a top-level argument (demographic stratification)
- `.load()` returns a flat `ModelData` dataclass. Key fields that downstream model code depends on: `y`, `aid`, `log_N`, `log_P`, `age_min`, `age_max`; optional fields `log_V`, `bid`, `rid`, `flat_ix`, `flat_pixs`. These field names must not change.
- `UserWarning` on dropped NaN rows (silent data loss is unacceptable in epidemiology)
- Eager validation fires at `ContactSurveyLoader.__init__`, not lazily in `.load()`, with column-level error messages

**Migration path (5 stages):**

1. **Refactor `DataValidator` to be non-mutating** — return container copies rather than mutating in-place; wire `DataLoader.__init__` to use new return values. No API change.
2. **Extract `_observation.py`** — move `_build_df_full`, `df_participant_counts`, `df_contact_counts`, `df_contact_offsets` to free functions. `BaseLoader` delegates to them; full unit-test coverage unlocked.
3. **Extract `_stratification.py`** — move the seven `infer_strat_*` / `make_*` methods to module-level free functions. `BaseLoader` delegates.
4. **Introduce `ColumnSpec.from_containers()`** and `DataFrameSurveySource`. `DataLoader._create_col_map` constructs `ColumnSpec` internally. No public API change.
5. **Introduce `ContactSurveyLoader` with `.from_dataframes()` factory**. Deprecate `BaseLoader` and `DataLoader` as thin shims for one release cycle. Audit for direct `BaseLoader` subclasses before removal (it is documented as a subclassing surface).

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

| Phase | Blockers for v1.0 | Owner suggestion |
|---|---|---|
| 1 — Architecture | ✅ All items are release blockers | Core maintainer (you + co-author) |
| 2 — Design | Items 2.1, 2.4, 2.6 are important; rest can slip to v1.1 | Contributor familiar with statistical models |
| 3 — API Polish | 3.1 (exports), 3.6 (Altair), 3.3 (types) are release blockers | Any contributor |
| 4 — Tests | 4.1 is an immediate blocker; 4.4 required before PyPI | Any contributor; 4.4 needs domain knowledge |
| 5 — Docs | 5.1 (contributor guide) critical for community adoption | Core maintainer |

**Minimum viable set for a credible v1.0 PyPI release**: Phase 1 complete + Phase 3.1/3.3/3.6 + Phase 4.1 + Phase 5.1/5.4.
