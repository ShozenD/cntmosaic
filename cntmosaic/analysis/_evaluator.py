import pandas as pd
import numpy as np
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
	mape = mean_absolute_percentage_error(y_true, y_est) * 100
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
	
class ModelEvaluator:
	"""Class for evaluating model accuracy.
	
	Parameters
	----------
	summariser : ModelSummariser
			A model summariser instance
			
	data_eval : tuple
			A tuple containing the dictionaries of true contact intensity and marginal contact intensity values
	""" 
	def __init__(self, summariser, cint_matrices_true: dict):
		self.summariser = summariser
		self.cint_matrices_true = cint_matrices_true
		# Calculate true marginal contact intensity
		self.mcint_true = {}
		for v in cint_matrices_true.keys():
			self.mcint_true[v] = {
    		w: cint_matrices_true[v][w].sum(axis=1)
				for w in cint_matrices_true[v].keys()
      }
   
		# Evaluate accuracy of inferred contact quantities
		self.evaluate_cint()
		self.evaluate_mcint()
	
	def evaluate_cint(self):
		"""Evaluate the posterior contact intensity.
		Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
		
		Returns
		-------
		pd.DataFrame
				A DataFrame containing the metrics summary
		"""
		if not hasattr(self, 'eval_cint'):
			self.eval_cint = aggregate_metrics(self.cint_matrices_true, self.summariser.summarise_cint())
   
		return self.eval_cint
	
	def evaluate_mcint(self):
		"""Evaluate the posterior marginal contact intensity.
		Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
		
		Returns
		-------
		pd.DataFrame
				A DataFrame containing the metrics summary
		"""
		if not hasattr(self, 'eval_mcint'):
			self.eval_mcint = aggregate_metrics(self.mcint_true, self.summariser.summarise_mcint())
   
		return self.eval_mcint
		