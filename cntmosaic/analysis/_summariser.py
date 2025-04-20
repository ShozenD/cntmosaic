import numpy as np
import jax

class ModelSummariserSVI:
	def __init__(self, model):
		self.model = model
		self.prng_key = jax.random.PRNGKey(0)
		self.get_post_predictive()
		self.get_post_predictive_cint()
		
	def get_post_predictive(self):
		"""
		Get the posterior predictive distribution of the model.
		This is a wrapper around the model's posterior_predictive_svi method.
		It uses the model's guide to sample from the posterior predictive distribution.
		The guide is a variational approximation to the posterior distribution.
		"""
		self.post_pred = self.model.posterior_predictive_svi(self.prng_key, self.model.guide)
	
	def get_post_predictive_cint(self):
		log_rate = self.post_pred['log_rate']
		post_pred_cint = {}
		for name, site in self.post_pred.items():
			if 'log_delta' in name:
				var = name.split('/')[0]
				cat = self.model.data[var].cat.categories
				post_pred_cint[var] = {
					cat[i]: np.exp(log_rate[:, None, :, :] + site + self.model.log_P[None, None, :, :])[:, i, :, :]
					for i in range(len(cat))
				}
	
		self.post_pred_cint = post_pred_cint
		
	def summarise_rate(self, probs: tuple = (0.025, 0.5, 0.975)):
		"""
		Summarise the rate parameter of the model.
		This is a wrapper around the model's summarise_rate method.
		It uses the model's posterior predictive distribution to compute the summary statistics.
		"""
		if 'sum_rate' not in self.__dict__:
			self.sum_rate = np.quantile(np.exp(self.post_pred['log_rate']), probs, axis=0)
	 
		return self.sum_rate

	def summarise_cint(self, probs: tuple = (0.025, 0.5, 0.975)):
		"""
		Summarise the contact intensity matrix of the model.
		It uses the model's posterior predictive distribution to compute the summary statistics.
		"""
		if 'sum_cint' not in self.__dict__:
			self.sum_cint = {
				var: {
      			name: np.quantile(value, probs, axis=0)
         		for name, value in cat.items()
			 		}
				for var, cat in self.post_pred_cint.items()
			}
						
		return self.sum_cint

	def summarise_mcint(self, probs: tuple = (0.025, 0.5, 0.975)):
		"""
		Summarise the marginal contact intensity of the model.
		It uses the model's posterior predictive distribution to compute the summary statistics.
		"""
		if 'sum_mcint' not in self.__dict__:
			self.sum_mcint = {
				var: {
						name: np.quantile(value.sum(axis=2), probs, axis=0)
				 		for name, value in cat.items()
			 		}
				for var, cat in self.post_pred_cint.items()
			}
						
		return self.sum_mcint