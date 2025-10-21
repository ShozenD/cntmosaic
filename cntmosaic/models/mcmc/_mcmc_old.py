import jax.numpy as jnp
from jax import jit, lax, random
from jax.scipy.stats import norm
from functools import partial
from typing import Tuple

# ===========================
# Utility functions for Polya-Gamma sampling
# References: 
# - Polson, Scott, and Windle (2013). "Bayesian Inference for Logistic Models Using Polya-Gamma Latent Variables"
# - BayesLogit R package: https://github.com/jwindle/BayesLogit
# ===========================

@jit
def invgauss_cdf(x, mu=0.0, lam=1.0):
	"""Cumulative distribution function for Inverse Gaussian distribution

  Parameters
  ----------
  x: float or NDArray
    Value(s) at which to evaluate the CDF.
  mu: float, default=0.0
    Mean parameter of the Inverse Gaussian distribution.
  lam: float, default=1.0
    Shape parameter of the Inverse Gaussian distribution.
  """
	Z = 1.0 / mu
	b = jnp.sqrt(lam / x) * (x * Z - 1)
	a = -1.0 * jnp.sqrt(lam / x) * (x * Z + 1)
	
	return norm.cdf(b) + jnp.exp(2 * lam * Z) * norm.cdf(a)

@jit
def q_and_p(Z, trunc=0.64):
	fz = jnp.pi**2 / 8 + Z**2 / 2
	p = (0.5 * jnp.pi) * jnp.exp( -1.0 * fz * trunc) / fz
	q = 2 * jnp.exp(-1.0 * Z) * invgauss_cdf(trunc, 1.0/Z, 1.0)

	return q, p, q/p

@jit
def truncated_exponential_pmf(x, trunc=0.64):
	"""PMF of truncated exponential distribution"""
	fz = jnp.pi**2 / 8 + x**2 / 2
	b = jnp.sqrt(1.0 / trunc) * (trunc * x - 1)
	a = -1.0 * jnp.sqrt(1.0 / trunc) * (trunc * x + 1)

	x0 = jnp.log(fz) + fz * trunc
	xb = x0 - x + norm.logcdf(b)
	xa = x0 + x + norm.logcdf(a)
 
	qdivp = 4 / jnp.pi * ( jnp.exp(xb) + jnp.exp(xa) )
 
	return 1.0 / (1.0 + qdivp)

@jit
def truncated_invgauss_inv_abs_z_1_sample(key, Z, trunc=0.64):
	"""
	Sampling from a truncated inverse Gaussian distribution where
	mean = 1/|Z| and truncated at trunc.

	Args:
		key: JAX random key
		Z: parameter Z (will be converted to absolute value)
		R: truncation parameter (default 0.64)
	
	Returns:
		Sampled value X from truncated inverse Gaussian
	"""
	Z = jnp.abs(Z)
	mu = 1.0 / Z
	
	def sample_when_mu_greater_than_trunc(key):
		"""Sampling when mu > trunc using rejection sampling"""
		
		def rejection_body(carry):
			key, _ = carry
			key, subkey1, subkey2, subkey3 = random.split(key, 4)
			
			# Generate two exponential random variables
			E = random.exponential(subkey1, shape=(2,))
			
			# Inner while loop: while E[0]^2 > 2 * E[1] / trunc
			def inner_body(inner_carry):
				inner_key, _ = inner_carry
				inner_key, inner_subkey = random.split(inner_key)
				E_new = random.exponential(inner_subkey, shape=(2,))
				return inner_key, E_new
			
			def inner_cond(inner_carry):
				_, E_curr = inner_carry
				return E_curr[0]**2 > 2 * E_curr[1] / trunc

			# Run inner while loop
			_, E_final = lax.while_loop(
				inner_cond,
				inner_body,
				(subkey2, E)
			)
			
			# Calculate X
			X = trunc / (1 + trunc * E_final[0])**2
			
			# Calculate alpha (acceptance probability)
			alpha = jnp.exp(-0.5 * Z**2 * X)
			
			# Generate uniform random variable for acceptance test
			u = random.uniform(subkey3)
			
			return key, (X, alpha, u)
		
		def rejection_cond(carry):
			_, (X, alpha, u) = carry
			return u > alpha  # Continue while uniform > alpha
		
		# Initial state for rejection sampling
		key, init_subkey = random.split(key)
		initial_X = trunc + 1.0
		initial_alpha = 0.0
		initial_u = 1.0
		
		# Run rejection sampling loop
		final_key, (final_X, _, _) = lax.while_loop(
			rejection_cond,
			rejection_body,
			(init_subkey, (initial_X, initial_alpha, initial_u))
		)
		
		return final_X
	
	def sample_when_mu_less_equal_trunc(key):
		"""Sampling when mu <= trunc using inverse Gaussian generation"""
		
		def inv_gauss_body(carry):
			key, _ = carry
			key, subkey1, subkey2 = random.split(key, 3)
			
			# Standard inverse Gaussian generation
			lambda_param = 1.0
			Y = random.normal(subkey1)**2
			
			# Calculate X using the standard formula
			sqrt_term = jnp.sqrt(4 * mu * lambda_param * Y + (mu * Y)**2)
			X = mu + 0.5 * mu**2 / lambda_param * Y - 0.5 * mu / lambda_param * sqrt_term
			
			# Acceptance/rejection step
			u = random.uniform(subkey2)
			X_final = jnp.where(u > mu / (mu + X), mu**2 / X, X)
			
			return key, X_final
		
		def inv_gauss_cond(carry):
			_, X = carry
			return X > trunc  # Continue while X > trunc

		# Initial state
		key, init_subkey = random.split(key)
		initial_X = trunc + 1.0  # Start with X > trunc to enter the loop

		# Run while loop until X <= trunc
		final_key, final_X = lax.while_loop(
			inv_gauss_cond,
			inv_gauss_body,
			(init_subkey, initial_X)
		)
		
		return final_X
	
	# Choose sampling method based on mu vs R
	result = lax.cond(
		mu > trunc,
		sample_when_mu_greater_than_trunc,
		sample_when_mu_less_equal_trunc,
		key
	)
	
	return result

@jit
def a_coef(n, x, trunc=0.64):
  """Calculate coefficient a_n(x) in the PG(1,0) density"""
  return jnp.where(
    x > trunc,
    jnp.pi * (n + 0.5) * jnp.exp(-1.0 * (n + 0.5)**2 * jnp.pi**2 * x / 2.0),
    (2.0 / jnp.pi / x)**1.5 * jnp.pi * (n + 0.5) * jnp.exp(-2.0 * (n + 0.5)**2 / x)
  )
  
@jit
def polyagamma_sample_1(key, Z=0.0, trunc=0.64):
  """
  JAX implementation of Devroye's algorithm for sampling from PG(1, Z).
  
  Args:
      key: JAX random key
      Z: parameter Z 
      trunc: truncation parameter (default 0.64)
  
  Returns:
      Sample from PG(1, Z) distribution
  """
  Z = jnp.abs(Z) * 0.5  # PG(1,z) = 1/4 J*(1,Z/2)
  
  # Calculate fz for exponential proposal
  fz = jnp.pi**2 / 8 + Z**2 / 2
  
  def outer_loop_body(carry):
      key, _ = carry
      key, subkey1, subkey2, subkey3 = random.split(key, 4)
      
      # Choose between truncated exponential and truncated inverse Gaussian
      mass_texpon_prob = truncated_exponential_pmf(Z, trunc)
      use_texpon = random.uniform(subkey1) < mass_texpon_prob
      
      # Sample X based on the chosen method
      def sample_texpon(subkey):
          exp_sample = random.exponential(subkey)
          return trunc + exp_sample / fz
      
      def sample_tigauss(subkey):
          return truncated_invgauss_inv_abs_z_1_sample(subkey, Z, trunc)
      
      X = lax.cond(
          use_texpon,
          sample_texpon,
          sample_tigauss,
          subkey2
      )
      
      # Start the alternating series
      S = a_coef(0, X, trunc)
      Y = random.uniform(subkey3) * S
      
      # Inner loop for alternating series
      def inner_loop_body(inner_carry):
          inner_key, n, S_curr, accepted = inner_carry
          
          # Calculate new coefficient
          a_n = a_coef(n, X, trunc)
          
          # Update S based on whether n is odd or even
          S_new = jnp.where(n % 2 == 1, S_curr - a_n, S_curr + a_n)
          
          # Check acceptance condition
          accept_condition = jnp.where(n % 2 == 1, Y <= S_new, Y > S_new)
          
          return inner_key, n + 1, S_new, accept_condition
      
      def inner_loop_cond(inner_carry):
          _, n, S_curr, accepted = inner_carry
          # Continue until we get an acceptance decision
          return ~accepted
      
      # Run inner alternating series loop
      _, final_n, final_S, final_accepted = lax.while_loop(
          inner_loop_cond,
          inner_loop_body,
          (key, 1, S, False)  # Start with n=1, initial S, not accepted
      )
      
      # Final acceptance check
      final_acceptance = Y <= final_S
      
      return key, (X, final_acceptance)
  
  def outer_loop_cond(carry):
      _, (X, accepted) = carry
      return ~accepted  # Continue until accepted
  
  # Initial state for outer loop
  key, init_subkey = random.split(key)
  initial_X = 0.0
  initial_accepted = False
  
  # Run outer rejection sampling loop
  final_key, (final_X, _) = lax.while_loop(
      outer_loop_cond,
      outer_loop_body,
      (init_subkey, (initial_X, initial_accepted))
  )
  
  # Return 0.25 * X as in the original R code
  return 0.25 * final_X

def polyagamma(
  key: random.PRNGKey,
  h: int=1,
  z: float=0.0,
  shape: tuple=None,
  trunc: float=0.64
) -> jnp.ndarray:
  """
  JAX implementation of general Polya-Gamma sampler PG(h, z).
  
  Parameters
  ----------
    key: jax.random.PRNGKey
      Random number generator key.
    h: int or array-like
      Shape parameter (can be scalar or array)
    z: float or array-like
      Location parameter (can be scalar or array)
    shape: tuple, optional
      Desired output shape. If None, inferred from broadcasting h and z
    trunc: float
      Truncation parameter (default 0.64)

  Returns
  -------
    jnp.ndarray
      Array of samples from PG(h, z) distribution
  """  
  # Convert inputs to arrays
  h_array = jnp.atleast_1d(jnp.asarray(h))
  z_array = jnp.atleast_1d(jnp.asarray(z))
  
  # Determine output shape by broadcasting h and z
  if shape is None:
    # Use numpy broadcasting rules to determine the output shape
    broadcast_shape = jnp.broadcast_shapes(h_array.shape, z_array.shape)
  else:
    broadcast_shape = shape
  
  # Broadcast both arrays to the final shape
  h_broadcasted = jnp.broadcast_to(h_array, broadcast_shape)
  z_broadcasted = jnp.broadcast_to(z_array, broadcast_shape)
  
  # Flatten for easier iteration
  h_flat = h_broadcasted.flatten()
  z_flat = z_broadcasted.flatten()
  size = h_flat.size

  # Generate all samples using scan for efficiency
  @jit
  def scan_fn(carry_key, i):
      carry_key, subkey = random.split(carry_key)
      
      h_i = h_flat[i]
      z_i = z_flat[i]
      
      # Sum h_i samples of PG(1, z_i) using while loop
      def sum_pg1_body(sum_carry):
          sum_key, j, current_sum = sum_carry
          sum_key, sum_subkey = random.split(sum_key)
          
          pg1_sample = polyagamma_sample_1(sum_subkey, z_i, trunc)
          new_sum = current_sum + pg1_sample
          
          return sum_key, j + 1, new_sum
      
      def sum_pg1_cond(sum_carry):
          _, j, _ = sum_carry
          return j < h_i
      
      # Sum h_i samples of PG(1, z_i)
      _, _, total_sample = lax.while_loop(
          sum_pg1_cond,
          sum_pg1_body,
          (subkey, 0, 0.0)
      )
      
      return carry_key, total_sample
  
  # Generate all samples
  key, init_subkey = random.split(key)
  _, samples = lax.scan(scan_fn, init_subkey, jnp.arange(size))
  
  # Reshape to the broadcast shape and handle scalar case
  if broadcast_shape == ():
    return jnp.squeeze(samples)
  elif broadcast_shape == (1,):
    return jnp.squeeze(samples)
  else:
    return samples.reshape(broadcast_shape)

# ===========================
# Slice sampler (specifically for Prem model's gamma0)
# ===========================
def make_slice_kernel(log_f, *, width: float = 0.3, m: int = 5, shrink: bool = False):
  """
  Build a JIT-compatible slice step for the hyper-parameter γ₀ > 0.

  Parameters
  ----------
  log_f   : callable(g0, gamma) -> log posterior density up to a constant
            (expects γ₀ > 0).  `gamma` is the current vector of γⱼ.
  width   : initial bracket width on the *log* scale.
  m       : maximum number of stepping-out expansions per side.
  shrink  : if True, perform Neal-style shrink-back before the final draw.

  Returns
  -------
  slice_kernel(carry, key) -> (new_carry, gamma0_new)
      carry = (β₀, gamma, γ₀).  Only γ₀ is updated.
  """
  width  = float(width)
  m      = int(m)
  shrink = bool(shrink)

  # log-target in θ = log γ₀ with Jacobian term
  def log_f_theta(theta, gamma):
      g0 = jnp.exp(theta)
      return log_f(g0, gamma) + theta

  @partial(jit, static_argnums=(2, 3, 4))
  def slice_kernel(
    key: random.PRNGKey,
    carry: Tuple[float, jnp.ndarray, float],
    width: float = width,
    m: int = m,
    shrink: bool = shrink
  ) -> Tuple[Tuple[float, jnp.ndarray, float], float]:

    gamma, gamma0 = carry
    theta_curr = jnp.log(gamma0)                   # unconstrained variable
    log_f_curr = log_f_theta(theta_curr, gamma)

    k_z, k_u, k_v, k_x, k_s = random.split(key, 5)

    # 1. draw slice height -------------------------------------------------
    z = log_f_curr - random.exponential(k_z)

    # 2. initial bracket on θ-scale ---------------------------------------
    u = random.uniform(k_u)
    L = theta_curr - width * u         # left end
    R = L + width                      # right end

    v = random.uniform(k_v)
    J = jnp.floor(m * v).astype(jnp.int32)
    K = (m - 1 - J).astype(jnp.int32)

    # --- stepping out to the left ----------------------------------------
    def cond_left(state):
        L, J = state
        return jnp.logical_and(J > 0, log_f_theta(L, gamma) > z)

    def body_left(state):
        L, J = state
        return (L - width, J - 1)

    L, _ = lax.while_loop(cond_left, body_left, (L, J))

    # --- stepping out to the right ---------------------------------------
    def cond_right(state):
        R, K = state
        return jnp.logical_and(K > 0, log_f_theta(R, gamma) > z)

    def body_right(state):
        R, K = state
        return (R + width, K - 1)

    R, _ = lax.while_loop(cond_right, body_right, (R, K))

    # 3. optional shrink-back loop ----------------------------------------
    if shrink:
      def cond_shrink(state):
          θ_new, L, R, _ = state
          return log_f_theta(θ_new, gamma) < z

      def body_shrink(state):
          _, L, R, k = state
          k, sub = random.split(k)
          θ_prop = L + random.uniform(sub) * (R - L)
          # shrink the side that does NOT contain θ_curr
          L = jnp.where(θ_prop < theta_curr, θ_prop, L)
          R = jnp.where(θ_prop < theta_curr, R,      θ_prop)
          return (θ_prop, L, R, k)

      init = ((L + R) / 2.0, L, R, k_s)
      _, L, R, _ = lax.while_loop(cond_shrink, body_shrink, init)

    # 4. final draw from the accepted slice --------------------------------
    theta_new  = random.uniform(k_x, minval=L, maxval=R)
    gamma0_new = jnp.exp(theta_new)                # back-transform
        
    return gamma0_new

  return slice_kernel
