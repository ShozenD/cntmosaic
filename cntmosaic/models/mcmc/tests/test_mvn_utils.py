import pytest
import numpy as np
from numpy import random
from .._mvn_utils import (
    mvn_logpdf_prec_chol,
    sample_mvn_prec_chol,
    mvn_cond_params,
    sample_mvn_cond
)

def test_basic_2d_case():
  """Test basic 2D case with simple known values."""
  # Create a simple 2D example
  x = np.array([1.0, 2.0])
  mean = np.array([0.0, 0.0])
  prec = np.array([[2.0, 0.5], 
                   [0.5, 1.0]])
  # Condition on x[1], compute conditional for x[0]
  ix_target = np.array([0])
  ix_cond = np.array([1])
    
  cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
    
  # Expected values computed analytically
  # P11 = [[2.0]], P12 = [[0.5]], x_rest = [2.0], mean_rest = [0.0]
  # cond_mean = 0.0 - inv([[2.0]]) @ [[0.5]] @ ([2.0] - [0.0])
  #           = 0.0 - 0.5 * 2.0 / 2.0 = -0.5
  expected_mean = np.array([-0.5])
  expected_prec = np.array([[2.0]])
  
  print(cond_mean)
  print(cond_prec)
    
  np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)
  np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)

def test_3d_case_single_index():
  """Test 3D case conditioning on a single variable."""
  x = np.array([1.0, 2.0, 3.0])
  mean = np.array([0.5, 1.0, 1.5])
  prec = np.array([[3.0, 1.0, 0.5],
                   [1.0, 2.0, 0.0],
                   [0.5, 0.0, 1.5]])
  ix_target = np.array([1])  # Compute conditional for x[1]
  ix_cond = np.array([0, 2])  # Condition on x[0] and x[2]
  
  cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
  
  # Check dimensions
  assert cond_mean.shape == (1,)
  assert cond_prec.shape == (1, 1)
  
  # Verify precision matrix is unchanged for the target variable
  expected_prec = prec[1:2, 1:2]  # P11 = [[2.0]]
  np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)
  
  # Check conditional mean computation
  # x_rest = [x[0], x[2]] = [1.0, 3.0]
  # mean_rest = [mean[0], mean[2]] = [0.5, 1.5]
  # P12 = [[1.0, 0.0]]
  # cond_mean = 1.0 - inv([[2.0]]) @ [[1.0, 0.0]] @ ([1.0, 3.0] - [0.5, 1.5])
  #           = 1.0 - (1/2.0) * 1.0 * (1.0 - 0.5) = 1.0 - 0.25 = 0.75
  expected_mean = np.array([0.75])
  np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)
  
def test_3d_case_multiple_indices():
  """Test 3D case conditioning on multiple variables."""
  x = np.array([1.0, 2.0, 3.0])
  mean = np.array([0.0, 0.0, 0.0])
  prec = np.array([[2.0, 0.5, 0.0],
                   [0.5, 3.0, 1.0],
                   [0.0, 1.0, 1.5]])
  ix_target = np.array([0, 2])  # Compute conditional for x[0] and x[2]
  ix_cond = np.array([1])  # Condition on x[1]
  
  cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
  
  # Check dimensions
  assert cond_mean.shape == (2,)
  assert cond_prec.shape == (2, 2)
  
  # Expected precision matrix (P11)
  expected_prec = np.array([[2.0, 0.0],
                            [0.0, 1.5]])
  np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)
  
  # For conditional mean with mean=0 and conditioning on x[1]=2.0:
  # P12 = [[0.5], [1.0]]
  # cond_mean = 0 - inv(P11) @ P12 @ (x[1] - 0)
  expected_mean = -np.linalg.solve(expected_prec, np.array([0.5, 1.0])) * 2.0
  np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)


class TestSampleMvnCond:
    """Test cases for the sample_mvn_cond function."""
    
    def test_basic_2d_case(self):
        """Test basic 2D case with simple known values."""
        x = np.array([1.0, 2.0])
        mean = np.array([0.0, 0.0])
        prec = np.array([[2.0, 0.5], 
                         [0.5, 1.0]])
        ix_target = np.array([0])
        ix_cond = np.array([1])
        
        # Generate multiple samples to test statistical properties
        n_samples = 1000
        samples = []
        for i in range(n_samples):
            sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Check output shape
        assert samples.shape == (n_samples, 1)
        
        # Compute expected conditional parameters
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        expected_var = 1.0 / cond_prec[0, 0]  # Conditional variance
        
        # Check sample statistics
        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        
        # Statistical tests (with reasonable tolerance for Monte Carlo)
        np.testing.assert_allclose(sample_mean, cond_mean[0], atol=0.1)
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.2)
    
    def test_3d_case_single_target(self):
        """Test 3D case with single target variable."""
        x = np.array([1.0, 2.0, 3.0])
        mean = np.array([0.5, 1.0, 1.5])
        prec = np.array([[3.0, 1.0, 0.5],
                         [1.0, 2.0, 0.0],
                         [0.5, 0.0, 1.5]])
        ix_target = np.array([1])
        ix_cond = np.array([0, 2])
        
        # Generate samples
        n_samples = 500
        samples = []
        for i in range(n_samples):
            sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Check shape
        assert samples.shape == (n_samples, 1)
        
        # Check that samples are reasonable (finite and not constant)
        assert np.all(np.isfinite(samples))
        assert np.var(samples) > 1e-6  # Should have some variance
    
    def test_3d_case_multiple_targets(self):
        """Test 3D case with multiple target variables."""
        x = np.array([1.0, 2.0, 3.0])
        mean = np.array([0.0, 0.0, 0.0])
        prec = np.array([[2.0, 0.5, 0.0],
                         [0.5, 3.0, 1.0],
                         [0.0, 1.0, 1.5]])
        ix_target = np.array([0, 2])
        ix_cond = np.array([1])
        
        # Single sample to check shape
        sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
        
        # Check shape
        assert sample.shape == (2,)
        
        # Generate multiple samples for statistical tests
        n_samples = 800
        samples = []
        for i in range(n_samples):
            sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
            samples.append(sample)
        
        samples = np.array(samples)
        assert samples.shape == (n_samples, 2)
        
        # Basic checks
        assert np.all(np.isfinite(samples))
        
        # Check that covariance structure is reasonable
        sample_cov = np.cov(samples.T)
        assert sample_cov.shape == (2, 2)
        assert np.all(np.diag(sample_cov) > 0)  # Positive variances
    
    def test_deterministic_conditioning(self):
        """Test that conditioning works deterministically."""
        # Create a case where we know the exact conditional distribution
        x = np.array([1.0, 0.0])  # Condition on x[1] = 0
        mean = np.array([2.0, 0.0])
        # Use diagonal precision matrix for independence
        prec = np.array([[1.0, 0.0],
                         [0.0, 4.0]])
        ix_target = np.array([0])
        ix_cond = np.array([1])
        
        # With independent variables, conditional should equal marginal
        sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
        
        # Check that the conditional parameters are correct
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        
        # For independent variables, conditional mean should equal marginal mean
        np.testing.assert_allclose(cond_mean, np.array([2.0]), rtol=1e-10)
        np.testing.assert_allclose(cond_prec, np.array([[1.0]]), rtol=1e-10)
    
    def test_identity_precision(self):
        """Test with identity precision matrix."""
        x = np.array([1.0, 2.0, 3.0])
        mean = np.array([0.5, 1.0, 1.5])
        prec = np.eye(3)  # Identity matrix
        ix_target = np.array([0, 2])
        ix_cond = np.array([1])
        
        # Generate a sample
        sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
        
        # Check shape
        assert sample.shape == (2,)
        
        # With identity precision, conditional should equal marginal
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        expected_mean = mean[ix_target]
        expected_prec = prec[np.ix_(ix_target, ix_target)]
        
        np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)
    
    def test_edge_case_single_variable(self):
        """Test edge case with single variable system."""
        x = np.array([1.5])
        mean = np.array([0.0])
        prec = np.array([[2.0]])
        ix_target = np.array([0])
        ix_cond = np.array([], dtype=np.int32)  # No conditioning variables
        
        # This should give marginal distribution
        sample = sample_mvn_cond(x, mean, prec, ix_target, ix_cond)
        
        assert sample.shape == (1,)
        assert np.isfinite(sample[0])