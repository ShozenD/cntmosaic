"""
NumPyro backend for cntmosaic Bayesian models.

This subpackage contains the NumPyro implementation of the
``InferenceBackend`` protocol defined in ``cntmosaic.models._backend``.

Contents
--------
NumPyroBackend
    Concrete backend that wraps the NumPyro HMC/NUTS and SVI engines.
    All inference methods delegate to the free functions in
    ``cntmosaic.models._numpyro`` without duplicating logic.

Adding a new backend (e.g. PyMC or INLA)
-----------------------------------------
1. Create a new subdirectory, e.g. ``cntmosaic/models/pymc/__init__.py``.
2. Implement a class that satisfies the ``InferenceBackend`` protocol:

   .. code-block:: python

       # cntmosaic/models/pymc/_backend.py
       class PyMCBackend:
           def run_mcmc(self, model, prng_key, *, num_samples, ...): ...
           def run_svi(self, model, guide, prng_key, *, ...): ...
           def get_mcmc_samples(self, mcmc_result): ...
           # ... all remaining protocol methods

3. Users pass the backend at model construction time:

   .. code-block:: python

       from cntmosaic.models.pymc import PyMCBackend
       model = AgeMixFF(loader, priors, backend=PyMCBackend())

   No changes to ``cntmosaic.models`` or ``ContactModel`` are required.

Note: importing this package loads NumPyro. Use lazy imports in model
constructors (``_get_backend()``) so that ``import cntmosaic.models``
does not trigger a NumPyro import at module load time.
"""

from ._backend import NumPyroBackend
from ._AgeMixFF import AgeMixFFNumPyroMixin
from ._AgeMixFC import AgeMixFCNumPyroMixin
from ._GenMixFF import GenMixFFNumPyroMixin
from ._GenMixFC import GenMixFCNumPyroMixin
from ._Prem import PremNumPyroMixin
from ._vdKassteele import vdKassteeleNumPyroMixin

__all__ = [
    "NumPyroBackend",
    "AgeMixFFNumPyroMixin",
    "AgeMixFCNumPyroMixin",
    "GenMixFFNumPyroMixin",
    "GenMixFCNumPyroMixin",
    "PremNumPyroMixin",
    "vdKassteeleNumPyroMixin",
]
