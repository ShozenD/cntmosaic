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
  
def plot_base_patterns(patterns: dict):
    """Plots the base contact patterns used to construct synthetic contact matrices.
    
    :param patterns: Dictionary containing contact patterns for different settings
    
    :return: Plotnine plot
    """
    # Prepare data for plotnine
    data = []
    for setting, matrix in patterns.items():
        df = pd.DataFrame(matrix)
        df = df.reset_index().melt(id_vars='index', var_name='part_age', value_name='y')
        df['part_age'] = df['part_age'].astype(int)
        df.rename(columns={'index': 'cnt_age'}, inplace=True)
        df['y'] = (df['y'] - np.min(df['y'])) / (np.max(df['y']) - np.min(df['y']))
        
        df['Setting'] = setting.capitalize()
        data.append(df)

    # Concatenate all data into a single DataFrame
    data = pd.concat(data, ignore_index=True)
    data['Setting'] = pd.Categorical(data['Setting'], categories=['Household', 'School', 'Work', 'Community'])

    # Create the plot
    plot = (
        ggplot(data, aes(x='part_age', y='cnt_age')) +
        geom_tile(aes(fill='y')) +
        scale_fill_cmap('Spectral_r') +
        scale_x_continuous(expand=[0, 0]) +
        scale_y_continuous(expand=[0, 0]) +
        facet_wrap('~Setting', nrow=1) +
        labs(
            x="Age of Contacting Individual",
            y="Age of Contact"
        ) +
        theme_bw() +
        theme(
            aspect_ratio=1,
            strip_background=element_blank(),
            strip_text=element_text(size=9),
            axis_text=element_text(size=8),
            axis_title=element_text(size=9),
            legend_position="none",
            figure_size=(10, 3)
        )
    )
    return plot