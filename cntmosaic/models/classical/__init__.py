"""
cntmosaic.models.classical
==========================

Classical (deterministic / frequentist) contact matrix models.

The primary model in this subpackage is :class:`SocialMix`, which implements
the socialmixr algorithm (Funk et al. 2024) for estimating age-structured
contact matrices from survey data.

All models in this subpackage inherit from :class:`DeterministicContactModel`,
which provides the common ``fit()`` / ``predict()`` interface.

Public API
----------
DeterministicContactModel
    Abstract base class for deterministic contact models.
SocialMix
    Socialmixr-style contact intensity and rate matrix estimator.
SocialMixBootstrap
    Bootstrap uncertainty quantification for SocialMix.
BootstrapResults
    Container for bootstrap estimation results.
"""

from ._base import DeterministicContactModel
from ._SocialMix import SocialMix
from ._socialmix_bootstrap import BootstrapResults, SocialMixBootstrap

__all__ = [
    "DeterministicContactModel",
    "SocialMix",
    "SocialMixBootstrap",
    "BootstrapResults",
]
