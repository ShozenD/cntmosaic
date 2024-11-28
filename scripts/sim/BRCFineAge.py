import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir/'cntmosaic'))

import pickle
import jax
import numpy as np
import pandas as pd

import numpyro
numpyro.set_platform('gpu')
numpyro.set_host_device_count(4)

from cntmosaic.sim import (
	load_contact_patterns,
	load_age_distribution,
	make_contact_pattern,
	sample_contacts
)

from cntmosaic.models import BRCFineAge


repo_path = '/Users/shozendan/Imperial/0_Research/mixing-patterns'
country = 'United_States'
level = 'country'

# Generate synthetic data
np.random.seed(0)
patterns = load_contact_patterns(repo_path, country, level, symmetrise=True, smooth=True)
age_dist = load_age_distribution(repo_path, country, level)
cnt_rate, cnt_int = make_contact_pattern(patterns, age_dist.P.values)
df_sim = sample_contacts(2500, cnt_int, age_dist.P.values)

# Initialise model
model = BRCFineAge(df_sim, age_dist.P.values, likelihood='poisson')

# Run MCMC
prng_key = jax.random.PRNGKey(0)
model.run_inference_mcmc(prng_key, num_samples=1000, num_warmup=500, num_chains=4)

# Pickle the model
output_dir = Path('/rds/general/user/sd121/home/high_res_brc_outputs')
with open(output_dir/'brc_fine_age.pkl', 'wb') as f:
  pickle.dump(model, f)