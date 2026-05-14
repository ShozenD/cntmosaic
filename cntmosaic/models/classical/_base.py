"""
Abstract base class for deterministic (non-Bayesian) contact models.

This module defines `DeterministicContactModel`, the common interface for
classical frequency-based contact matrix estimators such as SocialMix.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class DeterministicContactModel(ABC):
    """
    Abstract base class for deterministic social contact models.

    Deterministic contact models do not perform probabilistic inference.
    Instead, they fit a point estimate of the contact matrix directly from
    observed data (e.g., by computing weighted averages) and can optionally
    quantify uncertainty via non-parametric methods such as the bootstrap.

    The primary concrete implementation is :class:`SocialMix`.

    Methods
    -------
    fit()
        Prepare and load data, making the model ready for prediction.
    predict()
        Compute and return the estimated contact matrix/matrices.
    """

    @abstractmethod
    def fit(self) -> None:
        """
        Prepare and load data for matrix estimation.

        Implementations should perform all data preprocessing, validation,
        and aggregation steps required before ``predict()`` can be called.

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self) -> Dict[str, Any]:
        """
        Compute and return the estimated contact matrix.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping stratum labels to estimated contact matrices.
            Keys follow the ``"source->target"`` naming convention
            (e.g. ``"All->All"``, ``"M->F"``).

        Raises
        ------
        NotImplementedError
            Raised by default if a subclass does not override this method.
        """
        raise NotImplementedError
