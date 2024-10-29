from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import pandas as pd
from scipy.stats import multinomial, poisson, multivariate_normal
from typing import Callable

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import arviz as az

from plotnine import *

class DataGenerator():
  def __init__(self,
               n: int,
               A: int,
               pop: NDArray,
               seed: int = 0,
               C: int = 1/200):
    self.n = n
    self.A = A
    self.pop = pop
    self.seed = seed
    self.C = C

  def get_rate(self):
    """Generate a contact pattern matrix using a mixture of 3 bivariate Gaussians"""
    X = np.arange(0, self.A, 1)
    Y = np.arange(0, self.A, 1)
    grid = np.array([[i, j] for i in X for j in Y])
    p_main = multivariate_normal.pdf(grid, mean=[15, 15], cov=[[90, 90*0.90], [90*0.90, 90]])
    p_sub1 = multivariate_normal.pdf(grid, mean=[20, 5], cov=[[90, 90*0.80], [90*0.80, 90]])
    p_sub2 = multivariate_normal.pdf(grid, mean=[5, 20], cov=[[90, 90*0.80], [90*0.80, 90]])

    p = (p_main + p_sub1 + p_sub2) / 3
    p = p / p.sum() # Re-normalise

    rate = (self.C * p).reshape(self.A, self.A)

    return rate
  
class DataGeneratorBasic(DataGenerator):
  def __init__(self,
               n: int,
               A: int,
               pop: NDArray,
               seed: int = 0,
               C: int = 1/200):
    super().__init__(n=n, A=A, pop=pop, seed=seed, C=C)
    self.sample_size = multinomial.rvs(n=self.n, p=self.pop/self.pop.sum())
    self.rate = self.get_rate()

  def generate(self):
    # Index for all pairs of age groups
    aidx = np.array([[i, j] for i in range(self.A) for j in range(self.A)])

    # Contact intensity
    self.cint = self.rate * self.pop[None,:]

    # Generate contacts
    mu = (self.cint * self.sample_size[:,None])[aidx[:,0], aidx[:,1]]
    self.y = poisson.rvs(mu=mu)

    # Create dataframe
    age_part = aidx[:,0]
    age_cnt = aidx[:,1]
    self.data_train = pd.DataFrame({
      'age_part': age_part,
      'age_cnt': age_cnt,
      'n': self.sample_size[age_part],
      'p': self.pop[age_cnt],
      'rate': self.rate[aidx[:,0], aidx[:,1]],
      'cint': self.cint[aidx[:,0], aidx[:,1]],
      'y': self.y
    })
    
    self.data_eval = pd.DataFrame({
        'age_part': age_part,
        'age_cnt': age_cnt,
        'rate': self.rate[aidx[:,0], aidx[:,1]],
        'cint': self.cint[aidx[:,0], aidx[:,1]],
    })

    return self.data_train, self.data_eval

class DataGeneratorStratified(DataGeneratorBasic):
  def __init__(self,
               n: int,
               A: int,
               pop: NDArray,
               seed: int = 0,
               C: int = 1/200):
    super().__init__(n=n, A=A, pop=pop, seed=seed, C=C)

  def set_subgroup_rates(self, func: Callable[..., NDArray], **kwargs) -> NDArray:
    """
    Set the contact rate for each subgroup using a specified function which applies some form
    of transformation to the base contact rate matrix.

    :param x: Base contact rate matrix
    :type x: NDArray
    :param func: Function to apply to the base contact rate matrix. The function should take the base contact rate matrix as input and return a matrix with dimensions (n_subgroups, n_age_groups, n_age_groups)
    :type func: Callable[[NDArray], NDArray]
    :param kwargs: Additional arguments to pass to the function
    :type kwargs: Dict
    """
    self.rate_subgroups = func(self.rate, **kwargs)

  def set_subgroup_sample_sizes(self, func: Callable[..., NDArray], **kwargs) -> NDArray:
    """
    Set the sample size of each subgroup using a specified function which splits the
    base sample size into several subgroups.

    :param x: Base sample size by age group
    :type x: NDArray
    :param func: Function to split the sample size into subgroups. The function should take the base sample size as input and return a matrix with dimensions (n_subgroups, n_age_groups)
    :type func: Callable[[NDArray], NDArray]
    :param kwargs: Additional arguments to pass to the function
    :type kwargs: Dict
    """
    self.sample_size_subgroups = func(self.sample_size, **kwargs)

  def set_subgroup_populations(self, func: Callable[..., NDArray], **kwargs) -> NDArray:
    """
    Set the population size of each subgroup using a specified function which splits the
    base population into several subgroups. 

    :param x: Base population size by age group
    :type x: NDArray
    :param func: Function to split the population into subgroups. The function should take the base population as input and return a matrix with dimensions (n_subgroups, n_age_groups)
    :type func: Callable[[NDArray], NDArray]
    :param kwargs: Additional arguments to pass to the function
    :type kwargs: Dict
    """
    self.pop_subgroups = func(self.pop, **kwargs)

  def generate(self):
    K, A, B = self.rate_subgroups.shape

    idx = np.array([[k, i, j] for k in range(K) for i in range(A) for j in range(B)])
    self.cint_subgroups = self.rate_subgroups * self.pop[None,None,:]
    self.mu = self.cint_subgroups * self.sample_size_subgroups[:,:,None]
    self.y = poisson.rvs(mu=self.mu[idx[:,0], idx[:,1], idx[:,2]])

    self.data_train = pd.DataFrame({
      'subgroup': idx[:,0],
      'age_part': idx[:,1],
      'age_cnt': idx[:,2],
      'n': self.sample_size_subgroups[idx[:,0], idx[:,1]],
      'p': self.pop[idx[:,2]],
      'rate': self.rate_subgroups[idx[:,0], idx[:,1], idx[:,2]],
      'cint': self.cint_subgroups[idx[:,0], idx[:,1], idx[:,2]],
      'y': self.y
    })
    
    self.data_eval = pd.DataFrame({
        'subgroup': idx[:,0],
        'age_part': idx[:,1],
        'age_cnt': idx[:,2],
        'rate': self.rate_subgroups[idx[:,0], idx[:,1], idx[:,2]],
        'cint': self.cint_subgroups[idx[:,0], idx[:,1], idx[:,2]]
    })

    return self.data_train, self.data_eval


class ModelEvaluatorMCMC:
    def __init__(self, mcmc, model, data_eval: pd.DataFrame):
        self.mcmc = mcmc
        self.S = mcmc.num_chains * mcmc.num_samples
        self.data_eval = data_eval
        self.model = model
        self.A = model.A
        
    def diagnose(self):
        if not hasattr(self, 'idata'):
            self.idata = az.from_numpyro(self.mcmc)
            self.diag = az.summary(self.idata, kind='diagnostics')
            self.loo = az.loo(self.mcmc)
        
        return pd.DataFrame({
            'statistic': ['min_ess_bulk', 'min_ess_tail', 'max_rhat', 'elpd_loo', 'elpd_loo_se'],
            'site_name': [self.diag['ess_bulk'].idxmin(),
                          self.diag['ess_tail'].idxmin(),
                          self.diag['r_hat'].idxmax(),
                          np.nan, np.nan],
            'value': [self.diag['ess_bulk'].min(),
                      self.diag['ess_tail'].min(),
                      self.diag['r_hat'].max(),
                      np.round(self.loo['elpd_loo'], 2),
                      np.round(self.loo['se'], 2)]
        })
        
        
    def summarise_rate_bl(self):
        po_samples = self.mcmc.get_samples()
        beta0 = po_samples['baseline']
        f = po_samples['f']
        log_rate_bl = (beta0[:,None] + f).reshape((self.S, self.A, self.A))
        su_log_rate_bl = np.quantile(log_rate_bl, (0.025, 0.5, 0.975), axis=0)
        
        return np.exp(su_log_rate_bl)
    
    def summarise_rates(self, quantiles: tuple=(0.025, 0.5, 0.975)):
        samples = self.mcmc.get_samples()
        
        beta0 = samples['baseline']
        f     = samples['f']
      
        # Calculate baseline rate
        log_rate_bl = (beta0[:,None] + f).reshape((self.S, self.A, self.A))

        site_names = self.model.X_cols + '/log_delta'
        dfs = []
        for site in site_names:
            log_delta = samples[site]
            log_rates = log_rate_bl[:,None,:,:] + log_delta
            sum_rates = np.exp(np.quantile(log_rates, quantiles, axis=0))
            
            K = sum_rates.shape[-3]
            idx = np.array([[k, i, j] for k in range(K) for i in range(self.A) for j in range(self.A)])
            
            estim   = sum_rates[1, idx[:,0], idx[:,1], idx[:,2]]
            lower = sum_rates[0, idx[:,0], idx[:,1], idx[:,2]]
            upper = sum_rates[2, idx[:,0], idx[:,1], idx[:,2]]
            
            df = pd.DataFrame({'var_name': site.replace('/log_delta', ''),
                               'subgroup': idx[:,0],
                               'age_part': idx[:,1],
                               'age_cnt': idx[:,2],
                               'lower': lower,
                               'estimate': estim,
                               'upper': upper})
            dfs.append(df)
            
        return pd.concat(dfs)
    
    def summarise_marginal_rates(self, quantiles: tuple=(0.025, 0.5, 0.975)):
        samples = self.mcmc.get_samples()
        
        beta0 = samples['baseline']
        f     = samples['f']
        log_rate_bl = (beta0[:,None] + f).reshape((self.S, self.A, self.A))
        
        site_names = self.model.X_cols + '/log_delta'
        dfs = []
        for site in site_names:
            log_delta = samples[site]
            log_rates = log_rate_bl[:,None,:,:] + log_delta
            marginal_rates = np.sum(np.exp(log_rates), axis=-2)
            sum_marginal_rates = np.quantile(marginal_rates, quantiles, axis=0)
            
            K = sum_marginal_rates.shape[-2]
            idx = np.array([[k, i] for k in range(K) for i in range(self.A)])
            
            estim   = sum_marginal_rates[1, idx[:,0], idx[:,1]]
            lower = sum_marginal_rates[0, idx[:,0], idx[:,1]]
            upper = sum_marginal_rates[2, idx[:,0], idx[:,1]]
            
            df = pd.DataFrame({'var_name': site.replace('/log_delta', ''),
                               'subgroup': idx[:,0],
                               'age_part': idx[:,1],
                               'lower': lower,
                               'estimate': estim,
                               'upper': upper})
            dfs.append(df)
        
        return pd.concat(dfs)
    
    def get_eval_dfs(self):
        df_sum_rates = self.summarise_rates()
        df_sum_marginal_rates = self.summarise_marginal_rates()
        
        df_true_marginal_rates = (
            self.data_eval
            .groupby(['var_name', 'subgroup', 'age_part'])['rate']
            .sum()
            .reset_index()
        )
        
        df_eval_rates = pd.merge(self.data_eval,
                                df_sum_rates,
                                how='inner',
                                on=['var_name', 'subgroup', 'age_part', 'age_cnt'])
        df_eval_marginal_rates = pd.merge(df_true_marginal_rates,
                                         df_sum_marginal_rates,
                                         how='inner',
                                         on=['var_name', 'subgroup', 'age_part'])
        
        return df_eval_rates, df_eval_marginal_rates
    
    def evaluate_rate(self):
        df_eval, _ = self.get_eval_dfs()
            
        return pd.DataFrame({
            'rmse': root_mean_squared_error(df_eval['rate'], df_eval['estimate']),
            'mae': mean_absolute_error(df_eval['rate'], df_eval['estimate']),
            'coverage': np.mean(((df_eval['lower'] <= df_eval['rate']) & (df_eval['upper'] >= df_eval['rate']))*100)
        }, index=[0])
    
def plot_rates_matrix(x: pd.DataFrame, path: str, filename: str='rates.pdf'):
  x = pd.melt(x,
              id_vars=['subgroup', 'age_part', 'age_cnt'],
              value_vars=['rate', 'estimate'],
              var_name='type')

  plot = (
    ggplot(x, aes(x='age_part', y='age_cnt')) +
    geom_tile(aes(fill='value')) +
    facet_wrap('~ subgroup + type') +
    scale_x_continuous(expand=(0,0)) +
    scale_y_continuous(expand=(0,0)) +
    scale_fill_distiller(type='div',
                         palette='Spectral',
                         labels=scientific_format(digits=2),
                         direction=-1) +
    labs(x='Age of contacting individual',
         y='Age of contact',
         fill='rate') +
    theme_bw() +
    theme(strip_background=element_blank())
  )
    
  plot.save(Path(path) / filename, width=6.4, height=5.9)
  
def plot_rates_marginal(x: pd.DataFrame, path: str, filename: str='rates_marginal.pdf'):
  plot = (
    ggplot(x, aes(x='age_part', y='estimate')) +
    geom_line(color='#58508d') +
    geom_line(aes(y='rate'), color='#de425b') +
    geom_ribbon(aes(ymin='lower', ymax='upper'), alpha=0.3, fill='#58508d') +
    facet_wrap('~subgroup') +
    labs(x='Age of contacting individual', y='Rate') +
    scale_x_continuous(expand=(0,0)) +
    scale_y_continuous(labels=scientific_format(digits=2)) +
    theme_bw() +
    theme(strip_background=element_blank())
  )
  plot.save(Path(path) / filename, width=6.4, height=4.8)