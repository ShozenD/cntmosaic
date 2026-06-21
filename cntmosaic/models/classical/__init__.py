"""
cntmosaic.models.classical
==========================

Contact matrix models — both classical frequentist (SocialMix) and Bayesian
(Prem) implementations.

The primary frequentist model is :class:`SocialMix`, which implements the
socialmixr algorithm (Funk et al. 2024) for estimating age-structured contact
matrices from survey data.  :class:`Prem` implements the Bayesian methodology
from Prem et al. (2017).

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
Prem
    Bayesian contact matrix estimator (Prem et al. 2017).
"""

from ._base import DeterministicContactModel
from ._Prem import Prem
from ._SocialMix import SocialMix
from ._socialmix_bootstrap import BootstrapResults, SocialMixBootstrap

__all__ = [
    "DeterministicContactModel",
    "Prem",
    "SocialMix",
    "SocialMixBootstrap",
    "BootstrapResults",
]
