from ._base import ContactModel
from ._BRC import BRC
from ._BRCfine import BRCfine
from ._BRCrefine import BRCrefine
from ._HiBRCfine import HiBRCfine
from ._HiBRCrefine import HiBRCrefine
from ._numpyro import to_inference_data
from ._Prem import Prem
from ._vdKassteele import vdKassteele

# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# SocialMix and related classes have moved to cntmosaic.models.classical.
# They are re-exported here so that existing code using
#   from cntmosaic.models import SocialMix
# continues to work without modification.
# ---------------------------------------------------------------------------
from .classical import BootstrapResults, SocialMix, SocialMixBootstrap

__all__ = [
    # Abstract base
    "ContactModel",
    # Bayesian Rate Consistency models
    "BRC",
    "BRCfine",
    "BRCrefine",
    "HiBRCfine",
    "HiBRCrefine",
    # Other Bayesian models
    "Prem",
    "vdKassteele",
    # Classical models (re-exported for backward compatibility)
    "SocialMix",
    "SocialMixBootstrap",
    "BootstrapResults",
    # Utilities
    "to_inference_data",
]
