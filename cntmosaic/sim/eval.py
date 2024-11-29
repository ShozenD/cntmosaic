from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import arviz as az
from plotnine import *
from mizani.formatters import scientific_format
    
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
