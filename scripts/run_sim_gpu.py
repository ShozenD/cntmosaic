import sys
import time
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import jax
import numpyro
numpyro.set_platform('gpu')
numpyro.set_host_device_count(4)

from cntmosaic.sim import load_base_patterns, load_age_distribution, simulate_ses
from cntmosaic.preprocess import make_train_data
from cntmosaic.models import HiBRCfine
from cntmosaic.visuals import plot_contact_matrix, plot_contact_marginal

import logging
from omegaconf import DictConfig
import hydra

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=str(root_dir/'conf'), config_name='config')
def run(cfg: DictConfig) -> None:
  log.info('Starting experiment')
  output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

  # Load data
  log.info('Simulating data')
  patterns = load_base_patterns(repo_path=cfg.data.patterns_repo_path,
                                country=cfg.data.country,
                                level=cfg.data.level)
  age_dist = load_age_distribution(repo_path=cfg.data.patterns_repo_path,
                                   country=cfg.data.country,
                                   level=cfg.data.level)
  df_sample, age_dist_props, data_eval = simulate_ses(patterns, age_dist.P.values, dist='poisson')

  # Prepare data
  log.info('Preparing data')
  df_sample['ses'] = pd.Categorical(df_sample['ses'], categories=['low', 'mid', 'high'], ordered=True)
  df_train = make_train_data(df_sample, id_var='id', grp_vars=['ses'])

  # Initialise model
  log.info('Initialising model')
  model = HiBRCfine(df_train, age_dist.P.values, age_dist_props)
  model.set_hsgp_params(grid_type='diff-age')
  model.print_model_shape()

  log.info('Running MCMC')
  start = time.time()
  prng_key = jax.random.PRNGKey(cfg.mcmc.seed)
  model.run_inference_mcmc(
    prng_key,
    num_samples=cfg.mcmc.num_samples,
    num_warmup=cfg.mcmc.num_warmup,
    num_chains=cfg.mcmc.num_chains
  )
  end = time.time()
  elapsed = (end - start)/60
  log.info(f'MCMC run time: {elapsed:.2f} minutes')
  pd.DataFrame({'time': elapsed}, index=0).to_csv(output_dir/'mcmc_time.csv', index=False)
  
  log.info('Saving model')
  with open(output_dir/'model.pkl', 'wb') as f:
    pickle.dump(model, f)

  log.info('Evaluating model')
  # evaluator = ModelEvaluatorMCMC(mcmc, brcs, data_eval)

  # diagnosis = evaluator.diagnose()
  # diagnosis.to_csv(output_dir/'diagnosis.csv', index=False)

  # error = evaluator.evaluate_rate()
  # error.to_csv(output_dir/'error.csv', index=False)
  
  log.info('Plotting results')
  # df_eval_rates, df_eval_marginal_rates = evaluator.get_eval_dfs()
  
  # plot_rates_matrix(df_eval_rates, output_dir)
  # plot_rates_marginal(df_eval_marginal_rates, output_dir)
  
  log.info('Finished experiment')
  
if __name__ == "__main__":
  run()