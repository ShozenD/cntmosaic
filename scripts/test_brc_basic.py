import sys
sys.path.append('../src')

import numpy as np
import pandas as pd

from numpyro.infer.initialization import init_to_value
from numpyro_utils import fit_mcmc
from models import BRCBasic

import arviz as az
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', palette='deep')

SEED = 0

df = pd.read_csv('../data/sim/basic.csv')

brc = BRCBasic(df, M=[20, 20])
mcmc = fit_mcmc(SEED,
                brc.model,
                num_warmup=500,
                num_samples=500,
                num_chains=2,
                target_accept_prob=0.80,
                init_strategy=init_to_value(
                  values = {'baseline': -(brc.log_p + brc.log_n).mean()}
                ))

idata = az.from_numpyro(mcmc)

func_dict = {
  'q025': lambda x: np.percentile(x, 2.5),
  'q50': lambda x: np.percentile(x, 50),
  'q975': lambda x: np.percentile(x, 97.5),
}
po_cint = az.summary(idata, var_names='log_cint', stat_funcs=func_dict, extend=False)
po_cint = po_cint.map(np.exp)

po_rate = az.summary(idata, var_names='log_rate', stat_funcs=func_dict, extend=False)
po_rate = po_rate.map(np.exp)

