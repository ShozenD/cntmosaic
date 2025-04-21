import numpy as np
from scipy.ndimage import gaussian_filter

class ContactMatrixGenerator:
    def __init__(self, patterns: dict, age_dist: np.ndarray):
        self.patterns = patterns
        self.age_dist = age_dist

    def generate(self, max_margin: float = 30, sigma: float = 2.0, seed: int = None) -> np.ndarray:
        if seed is not None:
          np.random.seed(seed)
          
        mixing_weights = np.abs(np.random.normal(0, 3, 4))
        cint_pattern = (
          self.patterns['community'] * mixing_weights[0] +
          self.patterns['school'] * mixing_weights[1] +
          self.patterns['work'] * mixing_weights[2] +
          self.patterns['household'] * mixing_weights[3]
        )

        mcint_max = cint_pattern.sum(axis=0).max() # Find the maximum marginal contact intensity
        cint_pattern = cint_pattern / mcint_max    # Ensures that maximum marginal cint is 1

        M = cint_pattern * max_margin # Ensures that maximum marginal cint is max_margin
        
        P = np.diag(self.age_dist)
        M = 1/2 * (M + np.linalg.inv(P) @ M.T @ P)
        M = gaussian_filter(M, sigma=sigma)
        
        return M