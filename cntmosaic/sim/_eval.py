from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from jax import random
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import arviz as az
from plotnine import *
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)

def compute_metrics(y_true, y_est, y_low, y_high):
    """
    Compute RMSE, MAE, and coverage for given true values, estimates, and interval bounds.
    """
    rmse = root_mean_squared_error(y_true, y_est)
    mae = mean_absolute_error(y_true, y_est)
    mape = mean_absolute_percentage_error(y_true, y_est)
    coverage = np.mean((y_true >= y_low) & (y_true <= y_high)) * 100
    return rmse, mae, mape, coverage

def process_variable_metrics(var, data_eval, data_est):
    """
    Compute metrics for a single variable across its categories and overall.
    """
    metrics = []
    for cat, values in data_est[var].items():
        rmse, mae, mape, coverage = compute_metrics(
            data_eval[var][cat], values[1], values[0], values[2]
        )
        metrics.append({
            'var': var,
            'cat': cat,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'coverage': coverage
        })

    # Compute overall metrics for the variable
    y_true = np.vstack([data_eval[var][cat] for cat in data_est[var].keys()])
    y_est = np.vstack([values[1] for values in data_est[var].values()])
    y_low = np.vstack([values[0] for values in data_est[var].values()])
    y_high = np.vstack([values[2] for values in data_est[var].values()])

    rmse, mae, mape, coverage = compute_metrics(y_true, y_est, y_low, y_high)
    metrics.append({
        'var': var,
        'cat': 'all',
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'coverage': coverage
    })

    return metrics

def aggregate_metrics(data_eval, data_est):
    """
    Aggregate metrics for all variables and categories, and compute overall metrics.
    """
    all_metrics = []
    for var in data_est.keys():
        all_metrics.extend(process_variable_metrics(var, data_eval, data_est))

    # Compute overall metrics across all variables and categories
    y_true = np.vstack([
        data_eval[var][cat] for var in data_est.keys() for cat in data_est[var].keys()
    ])
    y_est = np.vstack([values[1] for var in data_est.keys() for values in data_est[var].values()])
    y_low = np.vstack([values[0] for var in data_est.keys() for values in data_est[var].values()])
    y_high = np.vstack([values[2] for var in data_est.keys() for values in data_est[var].values()])

    rmse, mae, mape, coverage = compute_metrics(y_true, y_est, y_low, y_high)
    all_metrics.append({
        'var': 'all',
        'cat': 'all',
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'coverage': coverage
    })

    # Combine into a DataFrame
    return pd.DataFrame(all_metrics)

class ModelEvaluator(ABC):
    def __init__(self, model, data_eval: tuple):
        self.model = model
        self.cint_eval, self.mcint_eval = data_eval
        self.prng_key = random.PRNGKey(0)
        
    @abstractmethod
    def get_predictive(self):
        pass
    
    @abstractmethod
    def get_pred_cint(self):
        pass
    
    @abstractmethod
    def summary_pred_cint(self):
        pass
    
    @abstractmethod
    def summary_pred_mcint(self):
        pass
    
    
class ModelEvaluatorSVI(ModelEvaluator):
    def __init__(self, model, data_eval: pd.DataFrame):
        super().__init__(model, data_eval)
        
    def get_predictive(self):
        self.pred = self.model.posterior_predictive_svi(self.prng_key, self.model.guide)
        
    def get_pred_cint(self):
        if not hasattr(self, 'post'):
            self.get_predictive()

        log_rate = self.pred['log_rate']
        pred_cint = {}
        for name, site in self.pred.items():
            if 'log_delta' in name:
                var = name.split('/')[0]
                cat = self.model.data[var].cat.categories
                pred_cint[var] = {
                    cat[i]: np.exp(log_rate[:,None,:,:] + site + self.model.log_P[None,None,:,:])[:,i,:,:]
                    for i in range(len(cat))
                }
        self.pred_cint = pred_cint
    
    def summary_rate(self, probs: tuple=(0.025, 0.5, 0.975)):
        if not hasattr(self, 'pred'):
            self.pred = self.get_predictive()
            
        if not hasattr(self, 'sum_pred_rate'):
            self.sum_pred_rate = np.quantile(
                np.exp(self.pred['log_rate']),
                probs,
                axis=0
            )
        
    def summary_cint(self, probs: tuple=(0.025, 0.5, 0.975)):
        if not hasattr(self, 'pred_cint'):
            self.get_pred_cint()
            
        if not hasattr(self, 'sum_pred_mcint'):
            self.sum_pred_cint = {
                var: {
                    name: np.quantile(value, probs, axis=0)
                    for name, value in cat.items()
                }
                for var, cat in self.pred_cint.items()
            }
            
        return self.sum_pred_cint
            
    def summary_mcint(self, probs: tuple=(0.025, 0.5, 0.975)):
        if not hasattr(self, 'pred_cint'):
            self.get_pred_cint()
            
        if not hasattr(self, 'sum_pred_mcint'):
            self.sum_pred_mcint = {
                var: {
                    name: np.quantile(value.sum(axis=-1), probs, axis=0)
                    for name, value in cat.items()
                }
                for var, cat in self.pred_cint.items()
            }
        
        return self.sum_pred_mcint
 
    def evaluate_cint(self):
        return aggregate_metrics(self.cint_eval, self.summary_post_cint())
    
    def evaluate_mcint(self):
        return aggregate_metrics(self.mcint_eval, self.summary_post_mcint())
 
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
