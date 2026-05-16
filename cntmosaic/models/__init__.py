from ._base import ContactModel
from ._GenMix import GenMix
from ._AgeMixFF import AgeMixFF
from ._AgeMixFC import AgeMixFC
from ._GenMixFF import GenMixFF
from ._GenMixFC import GenMixFC
from ._Prem import Prem
from ._vdKassteele import vdKassteele


def __getattr__(name: str):
    if name == "to_inference_data":
        from ._numpyro import to_inference_data

        return to_inference_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    # GenMix model family
    "GenMix",
    "AgeMixFF",
    "AgeMixFC",
    "GenMixFF",
    "GenMixFC",
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
