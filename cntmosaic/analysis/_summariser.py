import numpy as np
from numpy.typing import NDArray

import jax
from ..models import (
  SocialMix,
  BRCfine,
  BRCrefine,
  HiBRCfine,
  HiBRCrefine,
  Prem
)
from ..utils import pixilate, depixilate, AgeBins

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
		if isinstance(self.model, (HiBRCfine, HiBRCrefine)): 
			# For HiBRC models, the contact intensity needs to be computed
			log_rate = self.post_pred['log_rate'].astype(np.float32)  # Convert early to save memory
			log_P = self.model.log_P.astype(np.float32)  # Convert early to save memory
			post_pred_cint = {}
			
			for name, site in self.post_pred.items():
				if 'log_delta' in name:
					var = name.split('/')[0]
					cat = self.model.ds.attrs['grp_vars'][var]
					site = site.astype(np.float32)  # Convert early to save memory
					
					# Initialize dict for this variable
					post_pred_cint[var] = {}
					
					# Process each category separately to avoid memory explosion
					for i, c in enumerate(cat):
						# Compute contact intensity for this category only
						# Add dimensions efficiently without creating large intermediate arrays
						log_rate_expanded = log_rate[:, np.newaxis, :, :]  # shape: (n_samples, 1, A, A)
						site_cat = site[:, i:i+1, :, :]  # shape: (n_samples, 1, A, A) - slice to keep dims
						log_P_expanded = log_P[np.newaxis, np.newaxis, :, :]  # shape: (1, 1, A, A)
						
						# Compute log sum and exp in one operation to minimize memory
						log_sum = log_rate_expanded + site_cat + log_P_expanded
						
						# Use np.exp with out parameter to avoid creating intermediate arrays
						cint = np.exp(log_sum, dtype=np.float32).squeeze(axis=1)  # Remove singleton dimension
						post_pred_cint[var][c] = cint
						
						# Clean up intermediate arrays to free memory immediately
						del log_rate_expanded, site_cat, log_sum, cint
			
			self.post_pred_cint = post_pred_cint
		elif isinstance(self.model, (BRCfine, BRCrefine)):
			pass
		
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
			if type(self.model) in (BRCfine, BRCrefine):
				# For BRC models, the contact intensity is stored in 'log_cint'
				self.sum_cint = np.quantile(self.post_pred['log_cint'], probs, axis=0)
				self.sum_cint = np.exp(self.sum_cint)
    
			elif type(self.model) in (HiBRCfine, HiBRCrefine):
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
			if type(self.model) in (BRCfine, BRCrefine):
				mcint = np.exp(self.post_pred['log_cint']).sum(axis=2)
				self.sum_mcint = np.quantile(mcint, probs, axis=0)
    
			elif type(self.model) in (HiBRCfine, HiBRCrefine):
				self.sum_mcint = {
					var: {
							name: np.quantile(value.sum(axis=2), probs, axis=0)
							for name, value in cat.items()
						}
					for var, cat in self.post_pred_cint.items()
				}
						
		return self.sum_mcint

class ModelSummariserSocialMix:
	def __init__(self, sm: SocialMix, alpha: float=0.05):
		assert alpha > 0 and alpha < 1, "alpha must be between 0 and 1."	
  
		self.sm = sm
		self.age_bins = sm.age_bins
		self.effective_age_bins = sm.effective_age_bins
		self.age_dist = sm.df_age_dist['P'].values
		self.cint = sm.cint
		self.rate = sm.rate
		self.alpha = alpha
		self.summarise_rate(probs=(alpha/2, 0.5, 1-alpha/2))
		self.summarise_cint(probs=(alpha/2, 0.5, 1-alpha/2))
		self.summarise_mcint(probs=(alpha/2, 0.5, 1-alpha/2))
		
	def summarise_rate(self, probs: tuple = (0.025, 0.5, 0.975)):
		"""
		Summarise the contact rate matrix.
		"""
		if not hasattr(self.sm, 'boots_rate'):
			raise ValueError("Bootstrapping has not been performed.")
		
		if not hasattr(self, 'sum_rate'):
			self.sum_rate = np.quantile(self.sm.boots_rate, probs, axis=0)
			self.depix_sum_rate = np.array([ # TODO: This is not correct
				depixilate(self.sum_rate[i,:,:], self.sm.effective_age_bins)
				for i in range(self.sum_rate.shape[0])
			])
			
		return self.sum_rate
			
	def summarise_cint(self, probs: tuple = (0.025, 0.5, 0.975)):
		"""
		Summarise the contact intensity matrix.
		"""
		if not hasattr(self.sm, 'boots_cint'):
			raise ValueError("Bootstrapping has not been performed.")
		
		if not hasattr(self, 'sum_cint'):
			self.sum_cint = np.quantile(self.sm.boots_cint, probs, axis=0)
			self.depix_sum_cint = np.array([
				depixilate(self.sum_cint[i,:,:], self.sm.effective_age_bins, self.age_dist)
				for i in range(self.sum_cint.shape[0])
			])
			
		return self.sum_cint
			
	def summarise_mcint(self, probs: tuple = (0.025, 0.5, 0.975)):
		"""
		Summarise the marginal contact intensity of the model.
		"""
		if not hasattr(self, 'sum_mcint'):
			mcint = self.sm.boots_cint.sum(axis=2)
			depix_boots_cint = np.array([
				depixilate(self.sm.boots_cint[i,:,:], self.sm.effective_age_bins, self.age_dist)
				for i in range(self.sm.boots_cint.shape[0])
			])
			depix_mcint = depix_boots_cint.sum(axis=2)
			self.sum_mcint = np.quantile(mcint, probs, axis=0)
			self.depix_sum_mcint = np.quantile(depix_mcint, probs, axis=0)

		return self.sum_mcint

class ModelSummariserMCMC:
    '''
    Basic Model implementation only
    '''
    def __init__(self, model):
        self.model = model
        self.prng_key = jax.random.PRNGKey(0)
    
    def get_posterior(self):
        """Get posterior samples from the MCMC run."""
        self.post = self.model.mcmc.get_samples()
        
    def get_post_cint(self):
        """Calculate posterior contact intensity from MCMC samples"""
        if not hasattr(self, 'post'):
            self.get_posterior()
        self.post_cint = {'general': np.exp(self.post['log_cint'])}
        return self.post_cint
        
    def summarise_rate(self, probs: tuple=(0.025, 0.5, 0.975)):
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
        if not hasattr(self, 'post'):
            self.post = self.get_posterior()
            
        if not hasattr(self, 'sum_post_rate'):
            self.sum_post_rate = np.quantile(
                np.exp(self.post['log_rate']),
                probs,
                axis=0
            )
            
        return self.sum_post_rate
        
    def summarise_cint(self, probs: tuple=(0.025, 0.5, 0.975)):
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
            
        if not hasattr(self, 'sum_post_cint'):
            self.sum_post_cint = {name: np.quantile(value, probs, axis=0) for name, value in self.post_cint.items()
            }
            
        return self.sum_post_cint
            
    def summarise_mcint(self, probs: tuple=(0.025, 0.5, 0.975)):
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
      
class ModelSummariserPrem:
	def __init__(self,
               model: Prem,
               age_bins: AgeBins=None,
               age_dist: NDArray | None = None,
               age_grp_dist: NDArray | None = None,
               alpha=0.05):
		"""Summarises the inference results for Prem et al. style models.
  
		Parameters
		----------
		model: Prem
			A object of class Prem. SVI or MCMC must have been run.
		age_bins: AgeBins, optional
			AgeBins object that defines the age bins used in the model.
			This data is used for pixilating and depixilating the contact intensity matrix.
		age_dist: NDArray, optional
			An array of population sizes for each fine age (1-year age).
			This data is used for depixilating the contact intensity matrix.
			Must be an NDarray of shape (num_fine_age,).
		age_grp_dist: NDArray, optional
			An array of population sizes for each age group.
			Must be of shape (num_age_grps,).
		alpha: float, default=0.05
			Significance level for credible intervals.
  	"""
   
		self.model = model
		self.age_bins = age_bins
		self.age_dist = age_dist
		self.age_grp_dist = age_grp_dist
		self.alpha = alpha
		self.prng_key = jax.random.PRNGKey(0)

		if hasattr(model, 'mcmc'):
			self.post = model.mcmc.get_samples()
		elif hasattr(model, 'svi'):
			self.post = model.posterior_predictive_svi(self.prng_key, model.guide)
		else:
			raise ValueError("Model must have either mcmc or svi attributes.")
		self.post_cint = np.exp(self.post['log_cint'])
  
		if self.age_grp_dist is None:
			if self.age_bins is not None and self.age_dist is not None:
				# Calculate age_grp_dist from age_dist and age_bins
				age_grp_dist = []
				age_edges = self.age_bins.left + [self.age_bins.max + 1]
				for i in range(len(age_edges)-1):
					start_age = age_edges[i]
					end_age = age_edges[i+1]
					age_grp_dist.append(self.age_dist[start_age:end_age].sum())

				self.age_grp_dist = np.array(age_grp_dist)
  
	def symmetrize_cint(self):
		"""Symmetrize the contact intensity matrix using the reciprocity adjustment."""
  
		if self.age_grp_dist is None:
			raise ValueError("age_grp_dist must be provided for symmetrization.")

		# Symmetrize the contact intensity matrix
		M = self.post_cint
		print()
		P = np.diag(self.age_grp_dist)[np.newaxis, ...]
		P_inv = np.diag(1 / self.age_grp_dist)[np.newaxis, ...]
		self.post_cint = 0.5 * (M + P_inv @ np.transpose(M, (0, 2, 1)) @ P)

	def summarise_cint(self,
                     depix: bool = False,
                     symmetrize: bool = False,
                     probs: tuple = None,
                     alpha: float = 0.05):
		"""
		Summarise the posterior contact intensity matrix.

		Parameters
		----------
		depix: bool, default=False
			Whether to depixilate the contact intensity matrix.
		symmetrize: bool, default=False
			Whether to apply the reciprocity adjustment to the contact intensity matrix.
		probs: tuple, optional
			The quantiles to compute. If None, uses (alpha/2, 0.5, 1-alpha/2).
		alpha: float, default=0.05
			Significance level for credible intervals.
   
		Returns
		-------
		NDArray
			The quantiles of the contact intensity matrix.
		"""
		if probs is None:
			probs = (alpha/2, 0.5, 1-alpha/2)
  
		if symmetrize:
			self.symmetrize_cint()
    
		self.sum_cint = np.quantile(self.post_cint, probs, axis=0)

		if depix:
			if self.age_bins is None:
				raise ValueError("age_bins must be provided for depixilation.")
			if self.age_dist is None:
				raise ValueError("age_dist must be provided for depixilation.")
  
			# Depixilate the summed contact intensity
			if not hasattr(self, 'depix_sum_cint'):
				self.depix_sum_cint = np.array([
					depixilate(self.sum_cint[i,:,:], self.age_bins, self.age_dist)
					for i in range(self.sum_cint.shape[0])
				])
    
			return self.depix_sum_cint
		else:
			return self.sum_cint
  
	def summarise_mcint(self,
                     	depixilate: bool = False,
                      symmetrize: bool = False,
                     	probs: tuple = None,
										 	alpha: float = 0.05):
		"""
		Summarise the marginal contact intensity of the model.
  
		Parameters
		----------
		depixilate: bool, default=False
			Whether to depixilate the contact intensity matrix before calculating marginal contact intensity.
		symmetrize: bool, default=False
			Whether to apply the reciprocity adjustment to the contact intensity matrix.
		probs: tuple, optional
			The quantiles to compute. If None, uses (alpha/2, 0.5, 1-alpha/2).
		alpha: float, default=0.05
			Significance level for credible intervals.
		
  	Returns
		-------
		NDArray
			The quantiles of the marginal contact intensity.
		"""
		if probs is None:
			probs = (alpha/2, 0.5, 1-alpha/2)
   
		if not hasattr(self, 'sum_mcint'):
			if symmetrize:
				self.symmetrize_cint()
			mcint = self.post_cint.sum(axis=1)
			self.sum_mcint = np.quantile(mcint, probs, axis=0)
		
		if depixilate and self.age_bins is not None and self.age_dist is not None:
			if not hasattr(self, 'depix_sum_mcint'):
				# Depixilate the summed marginal contact intensity
				depix_cint = np.array([
					depixilate(self.post_cint[i,:,:], self.age_bins, self.age_dist)
					for i in range(self.post_cint.shape[0])
				])
				depix_mcint = depix_cint.sum(axis=1)
				self.depix_sum_mcint = np.quantile(depix_mcint, probs, axis=0)
   
			return self.depix_sum_mcint
		else:
			return self.sum_mcint
