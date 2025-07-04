import pandas as pd
import numpy as np
from sklearn.metrics import (
	root_mean_squared_error,
	mean_absolute_error,
	mean_absolute_percentage_error
)
from ..utils import pixilate, depixilate, AgeBins
from ..models import (
	BRCfine,
	BRCrefine,
	HiBRCfine,
	HiBRCrefine,
)
from ._summariser import (
  ModelSummariserSocialMix,
  ModelSummariserPrem,
  ModelSummariserSVI
)

def interval_score(y_true, y_low, y_high, alpha):
	"""
	Compute the interval score for given true values and interval bounds.
	"""
	return np.mean(
   	(y_high - y_low) + 2/alpha * (y_low - y_true) * np.maximum(0, y_low - y_true) +
				   2/alpha * (y_high - y_true) * np.maximum(0, y_high - y_true)
)

def compute_metrics(y_true, y_est, y_low, y_high):
	"""
	Compute RMSE, MAE, and coverage for given true values, estimates, and interval bounds.
	"""
	rmse = root_mean_squared_error(y_true, y_est)
	mae = mean_absolute_error(y_true, y_est)
	mape = mean_absolute_percentage_error(y_true, y_est) * 100
	int_score = interval_score(y_true, y_low, y_high, alpha=0.05)
	coverage = np.mean((y_true >= y_low) & (y_true <= y_high)) * 100
	return rmse, mae, mape, int_score, coverage

def process_variable_metrics(var, data_eval, data_est):
	"""
	Compute metrics for a single variable across its categories and overall.
	"""
	metrics = []
	for cat, values in data_est[var].items():
			rmse, mae, mape, int_score, coverage = compute_metrics(
					data_eval[var][cat], values[1], values[0], values[2]
			)
			metrics.append({
					'var': var,
					'cat': cat,
					'rmse': rmse,
					'mae': mae,
					'mape': mape,
					'interval_score': int_score,
					'coverage': coverage
			})

	# Compute overall metrics for the variable
	y_true = np.vstack([data_eval[var][cat] for cat in data_est[var].keys()])
	y_est = np.vstack([values[1] for values in data_est[var].values()])
	y_low = np.vstack([values[0] for values in data_est[var].values()])
	y_high = np.vstack([values[2] for values in data_est[var].values()])
  
	rmse, mae, mape, int_score, coverage = compute_metrics(y_true, y_est, y_low, y_high)
 
	metrics.append({
			'var': var,
			'cat': 'all',
			'rmse':	rmse,
			'mae': mae,
			'mape': mape,
			'interval_score': int_score,
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


	rmse, mae, mape, int_score, coverage = compute_metrics(y_true, y_est, y_low, y_high)
	all_metrics.append({
			'var': 'all',
			'cat': 'all',
			'rmse': rmse,
			'mae': mae,
			'mape': mape,
			'interval_score': int_score,
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
	def __init__(self,
              summariser: ModelSummariserSVI,
              cint_matrices_true: np.ndarray | dict):
   
		self.summariser = summariser
		self.cint_matrices_true = cint_matrices_true
  
		if isinstance(cint_matrices_true, dict):
			self.mcint_true = {}
			for v in cint_matrices_true.keys():
				self.mcint_true[v] = {
					w: cint_matrices_true[v][w].sum(axis=1)
					for w in cint_matrices_true[v].keys()
				}
		else:
			self.mcint_true = cint_matrices_true.sum(axis=1)
	
	def evaluate_cint(self):
		"""Evaluate the posterior contact intensity.
		Calculates the RMSE, MAE, MAPE, and 95% posterior coverage.
		
		Returns
		-------
		pd.DataFrame
				A DataFrame containing the metrics summary
		"""
		if not hasattr(self, 'eval_cint'):
			if type(self.summariser.model) in (BRCfine, BRCrefine):
				sum_cint = self.summariser.summarise_cint()
				rmse, mae, mape, int_score, coverage = compute_metrics(
					self.cint_matrices_true,
					sum_cint[1],
					sum_cint[0],
					sum_cint[2]
				)
				self.eval_cint = pd.DataFrame({
					'rmse': rmse,
					'mae': mae,
					'mape': mape,
					'interval_score': int_score,
					'coverage': coverage
				}, index=[0])
    
			elif type(self.summariser.model) in (HiBRCfine, HiBRCrefine):
				self.eval_cint = aggregate_metrics(self.cint_matrices_true,
                                       		 self.summariser.summarise_cint())
   
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
			if type(self.summariser.model) in (BRCfine, BRCrefine):
				sum_mcint = self.summariser.summarise_mcint()
				rmse, mae, mape, int_score, coverage = compute_metrics(
					self.mcint_true,
					sum_mcint[1],
					sum_mcint[0],
					sum_mcint[2]
				)
				self.eval_mcint = pd.DataFrame({
					'rmse': rmse,
					'mae': mae,
					'mape': mape,
					'interval_score': int_score,
					'coverage': coverage
				}, index=[0])
    
			elif type(self.summariser.model) in (HiBRCfine, HiBRCrefine):
				self.eval_mcint = aggregate_metrics(self.mcint_true,
                                        		self.summariser.summarise_mcint())
  
		return self.eval_mcint

class ModelEvaluatorSocialMix:
	def __init__(self,
							summariser: ModelSummariserSocialMix,
							cint_matrix_true: np.ndarray):
		self.summariser = summariser
		self.age_bins = summariser.effective_age_bins
		self.m_true = cint_matrix_true 		# True contact intensity matrix (1-year age)
		self.pix_m_true = pixilate(self.m_true, self.age_bins, summariser.age_dist)   # True contact intensity matrix (prespecified age bins)
		self.depix_m_true = depixilate(self.pix_m_true, self.age_bins, summariser.age_dist)
		self.m_hat = summariser.cint			# Estimated contact intensity matrix (prespecified age bins)
		self.evaluate_cint()
	
	def evaluate_cint(self):
		"""Evaluate the estimated contact intensity matrix"""
		return pd.DataFrame({
			'disc_err': [self.eval_cint_disc_err()],
			'est_err': [self.eval_cint_est_err()],
			'total_err': [self.eval_cint_total_err()],
			'interval_score': [self.eval_cint_int_score()],
			'coverage': [self.eval_cint_coverage()],
		})

	def eval_cint_disc_err(self):
		"""Compute the discretisation error"""
		self.cint_disc_err = np.mean(np.square(self.depix_m_true - self.m_true))
  
		return self.cint_disc_err

	def eval_cint_est_err(self):
		"""Compute the estimation error"""
		self.cint_est_err = np.mean(np.square(self.m_hat - self.pix_m_true))
		return self.cint_est_err

	def eval_cint_total_err(self):
		"""Compute the total error"""
		self.cint_total_err = self.eval_cint_disc_err() + self.eval_cint_est_err()
		# The values i equavalent to np.mean(np.square(self.depix_m_hat - self.m_true))
		return self.cint_total_err

	def eval_cint_int_score(self):
		"""Compute the negatively oriented interval score"""
		if not hasattr(self, 'interval_score'):
			u = self.summariser.depix_sum_cint[2]
			l = self.summariser.depix_sum_cint[0]
			self.cint_int_score = np.mean(
				(u - l) + 2/self.summariser.alpha * (l - self.m_true) * np.maximum(0, l - self.m_true) +
				2/self.summariser.alpha * (u - self.m_true) * np.maximum(0, u - self.m_true)
			)
  
		return self.cint_int_score

	def eval_cint_coverage(self):
		"""Compute the coverage"""
		if not hasattr(self, 'coverage'):
			u = self.summariser.depix_sum_cint[2]
			l = self.summariser.depix_sum_cint[0]
			self.cint_coverage = np.mean((self.m_true >= l) & (self.m_true <= u)) * 100
   
		return self.cint_coverage
		
class ModelEvaluatorPrem:
	def __init__(self,
               summariser: ModelSummariserPrem,
               cint_matrix_true: np.ndarray):
   
		self.summariser = summariser
		self.age_bins = summariser.age_bins
		self.age_dist = summariser.age_dist
		self.m_true = cint_matrix_true 		# True contact intensity matrix (1-year age)
  
		# Precompute
		self.pix_m_true = pixilate(self.m_true, self.age_bins, summariser.age_dist)   # True contact intensity matrix (prespecified age bins)
		self.depix_m_true = depixilate(self.pix_m_true, self.age_bins, summariser.age_dist)
  
		self.m_hat = summariser.summarise_cint()[1]
		self.depix_m_hat_sum = summariser.summarise_cint(depix=True)
  
		self.evaluate_cint()
	
	def evaluate_cint(self):
		"""Evaluate the estimated contact intensity matrix"""
		return pd.DataFrame({
			'disc_err': [self.eval_cint_disc_err()],
			'est_err': [self.eval_cint_est_err()],
			'total_err': [self.eval_cint_total_err()],
			'interval_score': [self.eval_cint_int_score()],
			'coverage': [self.eval_cint_coverage()],
		})

	def eval_cint_disc_err(self):
		"""Compute the discretisation error"""
		self.cint_disc_err = np.mean(np.square(self.depix_m_true - self.m_true))
  
		return self.cint_disc_err

	def eval_cint_est_err(self):
		"""Compute the estimation error"""
		self.cint_est_err = np.mean(np.square(self.m_hat - self.pix_m_true))
		return self.cint_est_err

	def eval_cint_total_err(self):
		"""Compute the total error"""
		self.cint_total_err = self.eval_cint_disc_err() + self.eval_cint_est_err()
		# The values i equavalent to np.mean(np.square(self.depix_m_hat - self.m_true))
		return self.cint_total_err

	def eval_cint_int_score(self):
		"""Compute the negatively oriented interval score"""
		if not hasattr(self, 'interval_score'):
			u = self.depix_m_hat_sum[2]
			l = self.depix_m_hat_sum[0]
			alpha = self.summariser.alpha
			self.cint_int_score = np.mean(
				(u - l) + 2/alpha * (l - self.m_true) * np.maximum(0, l - self.m_true) +
				2/alpha * (u - self.m_true) * np.maximum(0, u - self.m_true)
			)
  
		return self.cint_int_score

	def eval_cint_coverage(self):
		"""Compute the coverage"""
		if not hasattr(self, 'coverage'):
			u = self.depix_m_hat_sum[2]
			l = self.depix_m_hat_sum[0]
			self.cint_coverage = np.mean((self.m_true >= l) & (self.m_true <= u)) * 100
   
		return self.cint_coverage