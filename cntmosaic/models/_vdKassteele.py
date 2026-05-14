from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from .._types import StratMode
from ..dataloader import DataLoader
from ._base import ContactModel
from .numpyro import vdKassteeleNumPyroMixin
from .numpyro.priors import Hill, vdKassteele2D


class vdKassteele(vdKassteeleNumPyroMixin, ContactModel):
    """
    van de Kassteele model for estimating social contact matrices.

    This class implement the van de Kassteele model which is used for inferring
    social contact matrices based on age-specific contact data.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        likelihood: str,
        order: int = 2,
        tau_shape: float = 2.0,
        tau_rate: float = 0.1,
        prior: Optional[vdKassteele2D] = None,
        backend: Optional[Any] = None,
    ) -> None:
        """
        Initialise the van de Kassteele model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object containing the processed contact data.
        likelihood : str
            Observation likelihood.  Either ``'poisson'`` or ``'negbin'``.
        order : int, default=2
            P-spline order used by the vdKassteele2D prior.
        tau_shape : float, default=2.0
            Shape parameter of the Gamma prior on the smoothing precision ``tau``.
        tau_rate : float, default=0.1
            Rate parameter of the Gamma prior on the smoothing precision ``tau``.
        prior : vdKassteele2D, optional
            Pre-constructed ``vdKassteele2D`` prior object.  When supplied the
            model uses it directly and skips automatic prior construction from
            ``order``, ``tau_shape``, and ``tau_rate``.  This mirrors the
            ``priors`` argument of ``BRC`` and allows external prior sharing or
            customisation without sub-classing.  Backward compatibility is
            fully preserved: omitting this argument keeps the original behaviour.
        backend : InferenceBackend, optional
            Pluggable inference engine (default: NumPyroBackend).
        """
        super().__init__(backend=backend)
        self.data = dataloader.load()
        self.likelihood = likelihood

        # ================
        # Prior parameters
        # ================
        self.order = order
        self.tau_shape = tau_shape
        self.tau_rate = tau_rate

        self.age_min = self.data.age_min
        self.age_max = self.data.age_max
        self.A = int(self.age_max - self.age_min + 1)
        self.aid = jnp.array(self.data.aid, dtype=jnp.int8)
        self.bid = jnp.array(self.data.bid, dtype=jnp.int8)
        self.y = jnp.array(self.data.y)
        self.log_N = jnp.array(self.data.log_N)
        # Handle log_P shape: add newaxis only if 1D (unstratified case)
        log_P_raw = self.data.log_P
        if log_P_raw.ndim == 1:
            self.log_P = jnp.array(log_P_raw[jnp.newaxis, :])
        else:
            self.log_P = jnp.array(log_P_raw)

        # log_S is an optional field; default to zeros if not provided by the loader
        self.log_S = (
            jnp.array(self.data.log_S)
            if self.data.log_S is not None
            else jnp.zeros_like(self.y)
        )

        # Initialize optional attributes
        self.rid: Optional[ArrayLike] = None
        self.hill: Optional[Hill] = None

        # Optional repeat interview effect
        if self.data.rid is not None:
            self.rid = jnp.array(self.data.rid, dtype=jnp.int8)
            self.hill = Hill(max_value=int(self.data.rid.max()))

        self.prior_type: str = None
        self.prior: vdKassteele2D = None

        self._set_prior(prior)

    def _infer_prior_type(self) -> None:
        """
        Infers which type of prior to use amongst global, partial, and full

        Note: This method is used within _set_prior
        """
        if not self.data.is_stratified:
            prior_type = "global"
        else:
            modes = list(self.data.strat_modes.values())

            # If mixed type stratification
            if StratMode.PARTIAL in modes and StratMode.FULL in modes:
                # vdKassteele can only handle this pattern as partial
                prior_type = "partial"
            elif StratMode.FULL in modes:
                prior_type = "full"
            else:
                prior_type = "partial"

        self.prior_type = prior_type

    def _set_prior(self, prior: Optional[vdKassteele2D] = None) -> None:
        """
        Configure the vdKassteele2D prior for this model instance.

        If a pre-constructed *prior* object is supplied it is used directly
        (bypassing automatic construction).  Otherwise the prior is built
        from the ``order``, ``tau_shape``, and ``tau_rate`` parameters that
        were passed to ``__init__``, using the stratification mode inferred
        from the loaded dataset.

        Parameters
        ----------
        prior : vdKassteele2D, optional
            A fully-configured ``vdKassteele2D`` prior.  When provided, all
            automatic construction logic is skipped.
        """
        if prior is not None:
            self.prior = prior
            # Infer prior_type for use in model() (needed even when prior is external)
            self._infer_prior_type()
            return

        self._infer_prior_type()

        if self.prior_type == "global":
            self.prior = vdKassteele2D(
                prior_type="global",
                order=self.order,
                tau_shape=self.tau_shape,
                tau_rate=self.tau_rate,
            )
            self.prior.set_age_bounds(self.age_min, self.age_max)
            self.prior.set_event_dim(1)
            self.prior.set_loc(0.0)
        elif self.prior_type != "full":
            modes = list(self.data.strat_modes.values())
            dims = list(self.data.strat_dims.values())

            # Calculate total number of strata (product of all dimensions)
            # For example: gender (2) × setting (4) = 8 strata
            total_dims = 1
            for mode, dim in zip(modes, dims):
                if mode == StratMode.PARTIAL:
                    total_dims *= dim
                else:
                    # For FULL mode, dim is already the squared count (e.g., 4 for 2x2)
                    total_dims *= int(np.sqrt(dim))

            self.prior = vdKassteele2D(
                prior_type="partial",
                order=self.order,
                tau_shape=self.tau_shape,
                tau_rate=self.tau_rate,
            )
            self.prior.set_age_bounds(self.age_min, self.age_max)
            self.prior.set_event_dim(int(total_dims))
            self.prior.set_loc(total_dims)
        else:
            # For FULL mode with multiple variables, compute product of sqrt(dims)
            dims = np.asarray(list(self.data.strat_dims.values()))
            total_dims = int(np.prod(np.sqrt(dims)))
            self.prior = vdKassteele2D(
                prior_type="full",
                order=self.order,
                tau_shape=self.tau_shape,
                tau_rate=self.tau_rate,
            )
            self.prior.set_age_bounds(self.age_min, self.age_max)
            self.prior.set_event_dim(total_dims)
            self.prior.set_loc(0.0)

