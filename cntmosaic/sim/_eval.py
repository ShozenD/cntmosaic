from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from jax import random
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import arviz as az
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
    def __init__(self, model, data_eval: tuple=None):
        self.model = model
        self.prng_key = random.PRNGKey(0)
        
        if data_eval is not None:
            self.cint_eval, self.mcint_eval = data_eval
        
    def set_data_eval(self, data_eval: tuple):
        """Set the data for evaluation
        
        Parameters
        ----------
        data_eval : tuple
            A tuple containing the dictionaries of true contact intensity and marginal contact intensity values
        """
        self.cint_eval, self.mcint_eval = data_eval
    
class ModelEvaluatorSVI(ModelEvaluator):
    """Class for diagnosing, evaluating, and summarising SVI results.
    
    Parameters
    ----------
    model : Model
        The model object
    
    data_eval : tuple, default=None
        A tuple containing the dictionaries of true contact intensity and marginal contact intensity values
    
    Attributes
    ----------
    pred : dict
        The posterior predictive samples
    
    pred_cint : dict
        The posterior predictive contact intensity
        
    sum_pred_rate : np.ndarray
        The summary of the posterior predictive contact rate
        
    sum_pred_cint : dict
        The summary of the posterior predictive contact intensity
    
    sum_pred_mcint : dict
        The summary of the posterior predictive marginal contact intensity
    """
    def __init__(self, model, data_eval: tuple=None):
        super().__init__(model, data_eval)
        
    def get_predictive(self):
        """Get posterior predictive samples from the SVI run."""
        self.pred = self.model.posterior_predictive_svi(self.prng_key, self.model.guide)
        
    def get_pred_cint(self):
        """Calculate posterior predictive contact intensity from SVI samples"""
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
        """Summarise the posterior predictive contact rate
        
        Parameters
        ----------
        probs : tuple
            The quantiles to compute
        
        Returns
        -------
        np.ndarray
            The summary of the posterior predictive contact rate
        """
        if not hasattr(self, 'pred'):
            self.pred = self.get_predictive()
            
        if not hasattr(self, 'sum_pred_rate'):
            self.sum_pred_rate = np.quantile(
                np.exp(self.pred['log_rate']),
                probs,
                axis=0
            )
        
        return self.sum_pred_rate
        
    def summary_cint(self, probs: tuple=(0.025, 0.5, 0.975)):
        """Summarise the posterior predictive contact intensity
        
        Parameters
        ----------
        probs : tuple
            The quantiles to compute
            
        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior predictive contact intensity
        """
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
        """Summarise the posterior predictive marginal contact intensity
        
        Parameters
        ----------
        probs : tuple
            The quantiles to compute
        
        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior predictive marginal contact intensity
        """
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
        """Evaluate the posterior predictive contact intensity.
        Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metrics summary
        """
        return aggregate_metrics(self.cint_eval, self.summary_cint())
    
    def evaluate_mcint(self):
        """Evaluate the posterior predictive marginal contact intensity.
        Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metrics summary
        """
        return aggregate_metrics(self.mcint_eval, self.summary_mcint())
 
class ModelEvaluatorMCMC(ModelEvaluator):
    """Class for diagnosing, evaluating, and summarising MCMC results.
    
    Parameters
    ----------
    model : Model
        The model object
        
    data_eval : tuple
        A tuple containing the dictionaries of true contact intensity and marginal contact intensity values
        
    Attributes
    ----------
    idata : az.InferenceData
        The ArviZ InferenceData object
    
    diag : pd.DataFrame
        The diagnostics summary
        
    loo : az.LOO
        The Leave-One-Out cross-validation object
        
    post : dict
        The posterior samples
    """
    def __init__(self, model, data_eval: tuple=None):
        super().__init__(model, data_eval)
        
    def diagnose(self):
        """Diagnose the MCMC run
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the diagnostics summary
        """
        if not hasattr(self, 'idata'):
            self.idata = az.from_numpyro(self.model.mcmc)
            self.diag = az.summary(self.idata, kind='diagnostics')
            self.loo = az.loo(self.model.mcmc)
        
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
        
    def get_posterior(self):
        """Get posterior samples from the MCMC run."""
        self.post = self.model.mcmc.get_samples()
        
    def get_post_cint(self):
        """Calculate posterior contact intensity from MCMC samples"""
        if not hasattr(self, 'post'):
            self.get_posterior()
            
        log_rate = self.post['log_rate']
        post_cint = {}
        for name, site in self.post.items():
            if 'log_delta' in name:
                var = name.split('/')[0]
                cat = self.model.data[var].cat.categories
                post_cint[var] = {
                    cat[i]: np.exp(log_rate[:,None,:,:] + site + self.model.log_P[None,None,:,:])[:,i,:,:]
                    for i in range(len(cat))
                }
        self.post_cint = post_cint
        
    def summary_rate(self, probs: tuple=(0.025, 0.5, 0.975)):
        """Summarise the posterior contact rate
        
        Parameters
        ----------
        probs : tuple
            The quantiles to compute
        
        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior contact rate
        """
        if not hasattr(self, 'prob'):
            self.pred = self.get_posterior()
            
        if not hasattr(self, 'sum_post_rate'):
            self.sum_post_rate = np.quantile(
                np.exp(self.post['log_rate']),
                probs,
                axis=0
            )
            
        return self.sum_post_rate
        
    def summary_cint(self, probs: tuple=(0.025, 0.5, 0.975)):
        """Summarise the posterior contact intensity
        
        Parameters
        ----------
        probs : tuple
            The quantiles to compute
        
        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior contact intensity
        """
        if not hasattr(self, 'post_cint'):
            self.get_post_cint()
            
        if not hasattr(self, 'sum_post_mcint'):
            self.sum_post_cint = {
                var: {
                    name: np.quantile(value, probs, axis=0)
                    for name, value in cat.items()
                }
                for var, cat in self.post_cint.items()
            }
            
        return self.sum_post_cint
            
    def summary_mcint(self, probs: tuple=(0.025, 0.5, 0.975)):
        """Summarise the posterior marginal contact intensity
        
        Parameters
        ----------
        probs : tuple
            The quantiles to compute
        
        Returns
        -------
        dict
            A dictionary containing the quantiles of the posterior marginal contact intensity
        """
        if not hasattr(self, 'post_cint'):
            self.get_post_cint()
            
        if not hasattr(self, 'sum_post_mcint'):
            self.sum_post_mcint = {
                var: {
                    name: np.quantile(value.sum(axis=-1), probs, axis=0)
                    for name, value in cat.items()
                }
                for var, cat in self.post_cint.items()
            }
        
        return self.sum_post_mcint
    
    def post_pred_check(self):
        pass
    
    def evaluate_cint(self):
        """Evaluate the posterior contact intensity.
        Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metrics summary
        """
        return aggregate_metrics(self.cint_eval, self.summary_cint())
    
    def evaluate_mcint(self):
        """Evaluate the posterior marginal contact intensity.
        Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the metrics summary
        """
        return aggregate_metrics(self.mcint_eval, self.summary_mcint())