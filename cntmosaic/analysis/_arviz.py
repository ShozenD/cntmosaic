from typing import Dict, Optional

import numpy as np
import numpyro
from arviz import InferenceData, dict_to_dataset
from jax import random
from numpyro.handlers import substitute
from numpyro.infer import log_likelihood
from numpyro.infer.util import _predictive

from ..models._BRC import BRC


def svi_to_inference_data(
    model: BRC,
    num_samples: int = 1000,
    kwargs_dict: Optional[Dict[str, np.ndarray]] = None,
) -> InferenceData:
    """
    Convert NumPyro SVI data to an ArViz InferenceData object.

    Parameters
    ----------
    model:
            A fitted BRC or HIBRC model instance.
    num_samples: int
            The number of posterior samples to draw from the variational posterior.

    Returns
    -------
    InferenceData
            An ArViz InferenceData object containing posterior and log_likelihood data.
    """
    sub_guide = substitute(model._guide, model._svi_result.params)

    # If kwargs are not provided, use model.y by default
    if not kwargs_dict:
        kwargs_dict = {"y": model.y}

    posterior = _predictive(
        random.PRNGKey(0),
        sub_guide,
        {},
        (num_samples,),
        return_sites="",
        parallel=True,
        model_args=(),
        model_kwargs=kwargs_dict,
        exclude_deterministic=True,
    )

    # Convert posterior to xarray
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

    # Compute posterior predictive samples
    sub_model = substitute(model.model, model._svi_result.params)
    posterior_pred = _predictive(
        random.PRNGKey(1),
        sub_model,
        posterior,
        (num_samples,),
        return_sites="",
        parallel=True,
        model_args=(),
        model_kwargs=kwargs_dict,
    )

    # Convert posterior predictive to xarray (only keep observed sites)
    pp_data = {}
    for key, values in posterior_pred.items():
        if not key.startswith("_"):  # Filter out internal variables
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
