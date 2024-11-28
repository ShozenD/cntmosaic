import sys
import time
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir/'cntmosaic'))

import pickle
import pandas as pd

import numpyro
from numpyro_utils import run_inference_mcmc
from models import BRCStratified
from simulation import (
  ModelEvaluatorMCMC,
  simulate_ses,
  load_contact_patterns,
  load_age_distribution,
  plot_rates_matrix,
  plot_rates_marginal
)

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
  patterns = load_contact_patterns(repo_path=cfg.data.patterns_repo_path,
                                   country=cfg.data.country,
                                   level=cfg.data.level)
  age_dist = load_age_distribution(repo_path=cfg.data.patterns_repo_path,
                                   country=cfg.data.country,
                                   level=cfg.data.level)
  data_train, data_eval = simulate_ses(patterns, age_dist.P.values)

  # Prepare data
  log.info('Preparing data')
  data_train['data']['X_a'] = data_train['data']['subgroup']
  data_train['pop_ratio'] = {'X_a': data_train['pop'] / data_train['pop'].sum()}
  data_eval['var_name'] = 'X_a'

  # Initialise model
  log.info('Initialising model')
  brcs = BRCStratified(
    data_train['data'],
    M=[20, 20],
    pratio=data_train['pop_ratio'],
    smooth_type={'X_a': 'random'}
  )

  log.info('Running MCMC')
  numpyro.set_host_device_count(cfg.mcmc.device_count)
  start = time.time()
  mcmc = run_inference_mcmc(
    seed=cfg.mcmc.seed,
    model=brcs.model,
    num_warmup=cfg.mcmc.num_warmup,
    num_samples=cfg.mcmc.num_samples,
    num_chains=cfg.mcmc.num_chains,
  )
  end = time.time()
  elapsed = (end - start)/60
  log.info(f'MCMC run time: {elapsed:.2f} minutes')
  pd.DataFrame({'time': elapsed}, index=0).to_csv(output_dir/'mcmc_time.csv', index=False)

  log.info('Evaluating model')
  evaluator = ModelEvaluatorMCMC(mcmc, brcs, data_eval)

  diagnosis = evaluator.diagnose()
  diagnosis.to_csv(output_dir/'diagnosis.csv', index=False)

  error = evaluator.evaluate_rate()
  error.to_csv(output_dir/'error.csv', index=False)
  
  log.info('Plotting results')
  df_eval_rates, df_eval_marginal_rates = evaluator.get_eval_dfs()
  
  plot_rates_matrix(df_eval_rates, output_dir)
  plot_rates_marginal(df_eval_marginal_rates, output_dir)
  
  log.info('Finished experiment')
  
if __name__ == "__main__":
  run()