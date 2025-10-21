import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import lambertw

def E_pg(
  b: ArrayLike,
  c: ArrayLike,
  eps_pg: float = 1e-6
):
  """
  Compute the expected value of a Polya-Gamma distribution.

  This function calculates E[ω] for a Polya-Gamma distribution PG(b, c) using
  the formula E[PG(b, c)] = b/(2c) * tanh(c/2). For numerical stability when
  c is close to zero, it uses the limit value b/4.

  Parameters
  ----------
  b : array_like
    Shape parameter of the Polya-Gamma distribution. Must be positive.
  c : array_like
    Scale parameter of the Polya-Gamma distribution.
  eps_pg : float, optional
    Threshold for numerical stability when c is close to zero. 
    Default is 1e-6.

  Returns
  -------
  np.ndarray
    Expected value E[ω] of the Polya-Gamma distribution PG(b, c).

  Notes
  -----
  The Polya-Gamma distribution PG(b, c) has expected value:
  - E[PG(b, c)] = b/(2c) * tanh(c/2) for c ≠ 0
  - E[PG(b, 0)] = b/4 (limiting case as c → 0)

  The function handles the numerical instability when |c| < eps_pg by
  using the limiting value b/4 instead of the general formula.

  Examples
  --------
  >>> import jax.numpy as np
  >>> E_pg(1.0, 2.0)
  Array(0.4621172, dtype=float32)
  >>> E_pg(2.0, 0.0)  # Uses limiting case
  Array(0.5, dtype=float32)
  """
  """Compute E[omega] for Polya-Gamma distribution PG(b, c)"""
   # E[PG(b, c)] = b/(2c) * tanh(c/2); safe at c~0 ⇒ b/4
  abs_c = np.abs(c)
  val = (b / (2.0 * np.where(abs_c < eps_pg, 1.0, c))) * np.tanh(0.5 * c)
  return np.where(abs_c < eps_pg, b * 0.25, val)

def tune_r(lam: NDArray, max_dist: float, max_r: float=500):
  """
  Tunes the number of success parameter r in the negative binomial distribution
  based on a prespecified maximum distance from a Poisson distribution.
  
  Parameters
  ----------
  lam: array_like
    The rate parameter of the Poisson distribution
  max_dist: float
    The allowed maximum distance between the negative binomial distribution and the Poisson distribution
  max_r: float
    The maximum r value allowed.
  """
  
  logc = lam + np.log1p(max_dist**2)
  arg = -np.exp(-logc / lam) * (logc / lam)
  W0 = lambertw(arg)
  r = (lam * logc) / (logc + lam * W0)
    
  return np.minimum(r, max_r)