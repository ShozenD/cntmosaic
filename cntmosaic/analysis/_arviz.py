from typing import Dict, Optional

import numpy as np


def svi_to_inference_data(
    model,
    num_samples: int = 1000,
    kwargs_dict: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Convert NumPyro SVI data to an ArViz InferenceData object.

    Parameters
    ----------
    model :
            A fitted ContactModel instance with SVI results.
    num_samples: int
            The number of posterior samples to draw from the variational posterior.

    Returns
    -------
    InferenceData
            An ArViz InferenceData object containing posterior and log_likelihood data.
    """
    import numpyro
    from arviz import InferenceData, dict_to_dataset
    from jax import random
    from numpyro.handlers import substitute
    from numpyro.infer import Predictive, log_likelihood

    from ..models._base import ContactModel

    if model._svi_result is None:
        raise ValueError(
            "SVI inference has not been run. Call run_inference_svi() first."
        )
    if model._guide is None:
        raise ValueError(
            "No guide found on the model. Ensure run_inference_svi() stored self._guide."
        )

    # Substitute SVI-optimised parameters into the guide
    sub_guide = substitute(model._guide, model._svi_result.params)

    # If kwargs are not provided, use model.y by default
    if not kwargs_dict:
        kwargs_dict = {"y": model.y}

    # Draw samples from the variational posterior (guide)
    posterior = Predictive(sub_guide, num_samples=num_samples, parallel=True)(
        random.PRNGKey(0), **kwargs_dict
    )

    # Filter auto-guide internal sites
    data = {k: v for k, v in posterior.items() if "_auto_" not in k}
    for key, values in data.items():
        data[key] = values[None, ...]

    posterior_xarray = dict_to_dataset(data, library=numpyro)

    # Compute log_likelihood
    ll_dict = log_likelihood(model.model, posterior, **kwargs_dict)

    ll_data = {}
    for obs_name, ll in ll_dict.items():
        shape = (1, ll.shape[0]) + ll.shape[1:]
        ll_data[obs_name] = np.reshape(np.asarray(ll), shape)
    ll_xarray = dict_to_dataset(ll_data, library=numpyro)

    # Compute posterior predictive samples using the raw model conditioned on guide samples
    posterior_pred = Predictive(model.model, posterior_samples=posterior, parallel=True)(
        random.PRNGKey(1), **kwargs_dict
    )

    # Convert posterior predictive to xarray (only keep non-internal sites)
    pp_data = {}
    for key, values in posterior_pred.items():
        if not key.startswith("_"):
            shape = (1, values.shape[0]) + values.shape[1:]
            pp_data[key] = np.reshape(np.asarray(values), shape)
    pp_xarray = dict_to_dataset(pp_data, library=numpyro)

    # Add observed data for posterior predictive checks
    obs_data = {"obs": np.asarray(kwargs_dict["y"])}
    obs_xarray = dict_to_dataset(obs_data, library=numpyro)

    return InferenceData(
        **{
            "posterior": posterior_xarray,
            "log_likelihood": ll_xarray,
            "posterior_predictive": pp_xarray,
            "observed_data": obs_xarray,
        }
    )
