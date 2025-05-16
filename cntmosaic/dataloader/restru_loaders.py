import itertools
from typing import Optional
from abc import ABC
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

import jax.numpy as jnp

def get_params(distr):
	common_distribution_params = [
		# Central tendency & spread
		"loc",
		"scale",
		"mean",
		"variance",
		
		# Rate & shape
		"rate",
		"concentration",
		"concentration0",
		"concentration1",
		"scale_tril",
		"precision_matrix",
		"covariance_matrix",
		
		# Discrete/multivariate
		"total_count",
		"probs",
		"logits",
		"low",
		"high",
		"df",  # degrees of freedom (StudentT)
		
		# Meta/shape
		"batch_shape",
		"event_shape",
		"support"
	]
	return {
			attr: getattr(distr, attr)
			for attr in dir(distr)
			if (not attr.startswith("_") and not callable(getattr(distr, attr) )) and (attr in common_distribution_params)
		}

@dataclass
class HyperParams:
	def __init__(self):
		self.prior = {}

	def __str__(self):
		lines = []
		for k, v in self.__dict__.items():
			if k != 'prior':
				lines.append(f"{k}: {v}")
		lines.append("prior:")
		for k, v in self.prior.items():
			lines.append(f'{k}:{v}')
			d = get_params(v)
			for k1, v1 in d.items():
				lines.append(f'{k1}:{v1}')
		return '\n'.join(lines)

