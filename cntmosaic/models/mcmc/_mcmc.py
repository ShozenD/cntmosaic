import numpy as np
from numpy import random
from typing import Tuple

# ===========================
# Slice sampler (specifically for Prem model's gamma0)
# ===========================
def slice_kernel(
  gamma: np.ndarray,
  gamma0: float,
  log_f: callable,
  width: float = 0.3,
  m: int = 5,
  shrink: bool = False
):
  """
  Build a slice sampler for the hyper-parameter γ₀ > 0.

  Parameters
  ----------
  gamma   : current vector of γⱼ.
  gamma0  : current value of γ₀ > 0.
  log_f   : callable(g0, gamma) -> log posterior density up to a constant
            (expects γ₀ > 0).  `gamma` is the current vector of γⱼ.
  width   : initial bracket width on the *log* scale.
  m       : maximum number of stepping-out expansions per side.
  shrink  : if True, perform Neal-style shrink-back before the final draw.

  Returns
  -------
  gamma0_new : new value of γ₀ > 0.
  """
  width  = float(width)
  m      = int(m)
  shrink = bool(shrink)

  # log-target in θ = log γ₀ with Jacobian term
  def log_f_theta(theta, gamma):
    g0 = np.exp(theta)
    return log_f(g0, gamma) + theta

  theta_curr = np.log(gamma0)                   # unconstrained variable
  log_f_curr = log_f_theta(theta_curr, gamma)

  # 1. draw slice height -------------------------------------------------
  z = log_f_curr - random.exponential()

  # 2. initial bracket on θ-scale ---------------------------------------
  u = random.uniform()
  L = theta_curr - width * u         # left end
  R = L + width                      # right end

  v = random.uniform()
  J = np.floor(m * v).astype(np.int32)
  K = (m - 1 - J).astype(np.int32)

  # --- stepping out to the left ----------------------------------------
  while J > 0 and log_f_theta(L, gamma) > z:
    L = L - width
    J = J - 1

  # --- stepping out to the right ---------------------------------------
  while K > 0 and log_f_theta(R, gamma) > z:
    R = R + width
    K = K - 1

  # 3. optional shrink-back loop ----------------------------------------
  if shrink:
    θ_new = (L + R) / 2.0
    while log_f_theta(θ_new, gamma) < z:
      θ_prop = L + random.uniform() * (R - L)
      # shrink the side that does NOT contain θ_curr
      if θ_prop < theta_curr:
          L = θ_prop
      else:
          R = θ_prop
      θ_new = θ_prop

  # 4. final draw from the accepted slice --------------------------------
  theta_new = random.uniform(L, R)
  gamma0_new = np.exp(theta_new)                # back-transform
        
  return gamma0_new