# Implements multivariate normal utilities in JAX.
import numpy as np
from numpy import random
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, solve_triangular
from scipy.special import factorial
import scipy.sparse as sp
import sksparse as sksp

def diff_matrix(
	num_nodes: int,
	order: int,
	sparse: bool = False,
	sparse_format: str = 'csc'
) -> NDArray | sp.spmatrix:
	
	D = np.zeros((num_nodes - order, num_nodes), dtype=np.int32)
	i_vals = np.arange(order + 1)
	coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (-1) ** (order - i_vals)
 
	for i in range(num_nodes - order):
		D[i, i:i+order+1] = coeff.astype(np.int32)
	
	if sparse:
		if sparse_format == 'csc':
			D = sp.csc_matrix(D)
		elif sparse_format == 'csr':
			D = sp.csr_matrix(D)
		elif sparse_format == 'coo':
			D = sp.coo_matrix(D)
		else:
			raise ValueError(f"Unsupported sparse_format: {sparse_format}")
	
	return D

def laplacian_matrix(
  num_nodes: int,
  order: int,
  sparse: bool = False,
	sparse_format: str = 'csc'
) -> NDArray | sp.spmatrix:
  
	D = diff_matrix(num_nodes,
                 order,
                 sparse=sparse,
                 sparse_format=sparse_format)

	return D.T @ D

def mvn_logpdf_prec_chol(
	x: NDArray,
	mean: NDArray,
	L: NDArray,
) -> NDArray:
	"""
	Compute the log probability density function of a multivariate normal distribution
	using precision matrix parameterization.

	This function computes the log-pdf of a multivariate normal distribution where
	the precision matrix A is given via its Cholesky decomposition L such that
	A = L @ L.T. The covariance matrix is V = A^{-1}.

	Parameters
	----------
	x : NDArray
		Input vector of shape (d,) where d is the dimensionality.
	mean : NDArray  
		Mean vector of shape (d,).
	L : NDArray
		Lower triangular Cholesky factor of the precision matrix A of shape (d, d).
		The precision matrix is reconstructed as A = L @ L.T.

	Returns
	-------
	NDArray
		Log probability density value as a scalar array.

	Notes
	-----
	The computation uses the precision matrix formulation to avoid matrix inversion.
	The quadratic form (x-μ)ᵀ A (x-μ) is computed efficiently using the Cholesky
	factor via solve_triangular, and the log determinant is computed as twice the
	sum of log diagonal elements of L.

	The log-pdf formula implemented is:
	log p(x) = 0.5 * log|A| - 0.5 * (x-μ)ᵀ A (x-μ) - 0.5 * d * log(2π)
	
	where A is the precision matrix and d is the dimensionality.
	"""  
	diff = x - mean
	y = solve_triangular(L.T, diff, lower=False)   # y = L^T diff
	quad = np.dot(y, y)
	logdetA = 2.0 * np.sum(np.log(np.diag(L)))
	d = x.shape[0]
	return 0.5 * logdetA - 0.5 * quad - 0.5 * d * np.log(2.0 * np.pi)

def sample_mvn_prec_chol(
	mean: NDArray,
	L: NDArray
) -> NDArray:
	"""
	Sample from a multivariate normal distribution using precision matrix parameterization.

	This function samples from a multivariate normal distribution N(mean, A^{-1}) where
	A is the precision matrix (inverse covariance matrix) represented by its Cholesky
	decomposition A = L @ L.T.

	Parameters
	----------
	mean : NDArray
		Mean vector of the multivariate normal distribution. Shape (..., n).
	L : NDArray
		Lower triangular Cholesky factor of the precision matrix A, where
		A = L @ L.T. Shape (..., n, n).

	Returns
	-------
	NDArray
		Sample from the multivariate normal distribution N(mean, A^{-1}).
		Same shape as mean.

	Notes
	-----
	The algorithm works by:
	1. Sampling z ~ N(0, I) where I is the identity matrix
	2. Solving the triangular system L.T @ x = z to get x ~ N(0, A^{-1})
	3. Returning mean + x

	This approach is numerically stable and efficient for precision matrix
	parameterizations commonly used in Bayesian inference.
	"""
	z = random.normal(size=mean.shape)
	x = solve_triangular(L, z, lower=True)
	
	return mean + x

def mvn_cond_params(
	x: NDArray,
	mean: NDArray,
	prec: NDArray,
	ix_target: NDArray,
	ix_cond: NDArray
):
	"""
	Compute the conditional mean and precision given the full mean vector,
	precision matrix, and indices of the target elements.
	
	Parameters
	----------
	x : NDArray
		Full vector of shape (d,) where d is the dimensionality.
	mean : NDArray
		Mean vector of the full multivariate normal distribution.
	prec : NDArray
		Precision matrix of the full multivariate normal distribution.
	ix_target : NDArray
		Indices of the elements of interest for which to compute the conditional distribution.
	ix_cond : NDArray
		Indices of the lements for which to condition on.
		
	Returns
	-------
	cond_mean : NDArray
		Conditional mean vector for the elements at the specified indices.
	cond_prec : NDArray
		Conditional precision matrix for the elements at the specified indices.
	"""
	
	# The values of x to be conditioned on
	x_cond = x[ix_cond]
	
	# Partition mean
	m_target = mean[ix_target]
	m_cond = mean[ix_cond]

	# Partition precision
	P11 = prec[np.ix_(ix_target, ix_target)]
	P12 = prec[np.ix_(ix_target, ix_cond)]
	
	# Conditional precision (unchanged)
	cond_prec = P11
	
	# Conditional mean
	cond_mean = m_target - np.linalg.solve(P11, P12 @ (x_cond - m_cond))
	
	return cond_mean, cond_prec

def sample_mvn_cond(
	x: NDArray,
	mean: NDArray,
	prec: NDArray,
	ix_target: NDArray,
	ix_cond: NDArray
) -> NDArray:
	"""
	Sample from the conditional distribution of a multivariate normal distribution.
	
	Parameters
	----------
	x : NDArray
		Full vector of shape (d,) where d is the dimensionality.
	mean : NDArray
		Mean vector of the full multivariate normal distribution.
	prec : NDArray
		Precision matrix of the full multivariate normal distribution.
	ix_target : NDArray
		Indices of the elements of interest for which to compute the conditional distribution.
	ix_cond : NDArray
		Indices of the elements for which to condition on.
		
	Returns
	-------
	NDArray
		Sample from the conditional distribution for the elements at the specified indices.
	"""
	
	m, P = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
	L = np.linalg.cholesky(P)
	
	return sample_mvn_prec_chol(m, L)

def logpdf_igmrf(
	beta: NDArray,
	Q0: NDArray,
	tau: float,
	r_eff: float = None,
	logdet_Q_const: float = None
):
	Q = Q0 * tau
	
	if r_eff is None:
		eigval_Q0, _ = np.linalg.eigh(Q0)
		pos = eigval_Q0 > 0
		
		r_eff = pos.sum()
		logdet_Q_const = np.sum(
			np.log(np.where(pos, eigval_Q0, 1.0))
		)
	
	b = beta
	quad = b @ (Q @ b)
	logdet_pseudo = r_eff * np.log(tau) + logdet_Q_const
	
	return 0.5 * logdet_pseudo - 0.5 * quad