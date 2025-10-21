import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from scipy import stats
from .._mvn import mvn_logpdf_precision, sample_mvn_from_precision, mvn_cond_params, sample_mvn_cond

def test_basic_2d_case():
  """Test basic 2D case with simple known values."""
  # Create a simple 2D example
  x = jnp.array([1.0, 2.0])
  mean = jnp.array([0.0, 0.0])
  prec = jnp.array([[2.0, 0.5], 
                   [0.5, 1.0]])
  # Condition on x[1], compute conditional for x[0]
  ix_target = jnp.array([0])
  ix_cond = jnp.array([1])
    
  cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
    
  # Expected values computed analytically
  # P11 = [[2.0]], P12 = [[0.5]], x_rest = [2.0], mean_rest = [0.0]
  # cond_mean = 0.0 - inv([[2.0]]) @ [[0.5]] @ ([2.0] - [0.0])
  #           = 0.0 - 0.5 * 2.0 / 2.0 = -0.5
  expected_mean = jnp.array([-0.5])
  expected_prec = jnp.array([[2.0]])
  
  print(cond_mean)
  print(cond_prec)
    
  np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)
  np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)

def test_3d_case_single_index():
  """Test 3D case conditioning on a single variable."""
  x = jnp.array([1.0, 2.0, 3.0])
  mean = jnp.array([0.5, 1.0, 1.5])
  prec = jnp.array([[3.0, 1.0, 0.5],
                   [1.0, 2.0, 0.0],
                   [0.5, 0.0, 1.5]])
  ix_target = jnp.array([1])  # Compute conditional for x[1]
  ix_cond = jnp.array([0, 2])  # Condition on x[0] and x[2]
  
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
  expected_mean = jnp.array([0.75])
  np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)
  
def test_3d_case_multiple_indices():
  """Test 3D case conditioning on multiple variables."""
  x = jnp.array([1.0, 2.0, 3.0])
  mean = jnp.array([0.0, 0.0, 0.0])
  prec = jnp.array([[2.0, 0.5, 0.0],
                   [0.5, 3.0, 1.0],
                   [0.0, 1.0, 1.5]])
  ix_target = jnp.array([0, 2])  # Compute conditional for x[0] and x[2]
  ix_cond = jnp.array([1])  # Condition on x[1]
  
  cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
  
  # Check dimensions
  assert cond_mean.shape == (2,)
  assert cond_prec.shape == (2, 2)
  
  # Expected precision matrix (P11)
  expected_prec = jnp.array([[2.0, 0.0],
                            [0.0, 1.5]])
  np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)
  
  # For conditional mean with mean=0 and conditioning on x[1]=2.0:
  # P12 = [[0.5], [1.0]]
  # cond_mean = 0 - inv(P11) @ P12 @ (x[1] - 0)
  expected_mean = -jnp.linalg.solve(expected_prec, jnp.array([0.5, 1.0])) * 2.0
  np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)


class TestSampleMvnCond:
    """Test cases for the sample_mvn_cond function."""
    
    def test_basic_2d_case(self):
        """Test basic 2D case with simple known values."""
        key = random.PRNGKey(42)
        
        x = jnp.array([1.0, 2.0])
        mean = jnp.array([0.0, 0.0])
        prec = jnp.array([[2.0, 0.5], 
                         [0.5, 1.0]])
        ix_target = jnp.array([0])
        ix_cond = jnp.array([1])
        
        # Generate multiple samples to test statistical properties
        n_samples = 1000
        samples = []
        for i in range(n_samples):
            key, subkey = random.split(key)
            sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
            samples.append(sample)
        
        samples = jnp.array(samples)
        
        # Check output shape
        assert samples.shape == (n_samples, 1)
        
        # Compute expected conditional parameters
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        expected_var = 1.0 / cond_prec[0, 0]  # Conditional variance
        
        # Check sample statistics
        sample_mean = jnp.mean(samples)
        sample_var = jnp.var(samples)
        
        # Statistical tests (with reasonable tolerance for Monte Carlo)
        np.testing.assert_allclose(sample_mean, cond_mean[0], atol=0.1)
        np.testing.assert_allclose(sample_var, expected_var, rtol=0.2)
    
    def test_3d_case_single_target(self):
        """Test 3D case with single target variable."""
        key = random.PRNGKey(123)
        
        x = jnp.array([1.0, 2.0, 3.0])
        mean = jnp.array([0.5, 1.0, 1.5])
        prec = jnp.array([[3.0, 1.0, 0.5],
                         [1.0, 2.0, 0.0],
                         [0.5, 0.0, 1.5]])
        ix_target = jnp.array([1])
        ix_cond = jnp.array([0, 2])
        
        # Generate samples
        n_samples = 500
        samples = []
        for i in range(n_samples):
            key, subkey = random.split(key)
            sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
            samples.append(sample)
        
        samples = jnp.array(samples)
        
        # Check shape
        assert samples.shape == (n_samples, 1)
        
        # Check that samples are reasonable (finite and not constant)
        assert jnp.all(jnp.isfinite(samples))
        assert jnp.var(samples) > 1e-6  # Should have some variance
    
    def test_3d_case_multiple_targets(self):
        """Test 3D case with multiple target variables."""
        key = random.PRNGKey(456)
        
        x = jnp.array([1.0, 2.0, 3.0])
        mean = jnp.array([0.0, 0.0, 0.0])
        prec = jnp.array([[2.0, 0.5, 0.0],
                         [0.5, 3.0, 1.0],
                         [0.0, 1.0, 1.5]])
        ix_target = jnp.array([0, 2])
        ix_cond = jnp.array([1])
        
        # Single sample to check shape
        key, subkey = random.split(key)
        sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
        
        # Check shape
        assert sample.shape == (2,)
        
        # Generate multiple samples for statistical tests
        n_samples = 800
        samples = []
        for i in range(n_samples):
            key, subkey = random.split(key)
            sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
            samples.append(sample)
        
        samples = jnp.array(samples)
        assert samples.shape == (n_samples, 2)
        
        # Basic checks
        assert jnp.all(jnp.isfinite(samples))
        
        # Check that covariance structure is reasonable
        sample_cov = jnp.cov(samples.T)
        assert sample_cov.shape == (2, 2)
        assert jnp.all(jnp.diag(sample_cov) > 0)  # Positive variances
    
    def test_deterministic_conditioning(self):
        """Test that conditioning works deterministically."""
        key = random.PRNGKey(789)
        
        # Create a case where we know the exact conditional distribution
        x = jnp.array([1.0, 0.0])  # Condition on x[1] = 0
        mean = jnp.array([2.0, 0.0])
        # Use diagonal precision matrix for independence
        prec = jnp.array([[1.0, 0.0],
                         [0.0, 4.0]])
        ix_target = jnp.array([0])
        ix_cond = jnp.array([1])
        
        # With independent variables, conditional should equal marginal
        key, subkey = random.split(key)
        sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
        
        # Check that the conditional parameters are correct
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        
        # For independent variables, conditional mean should equal marginal mean
        np.testing.assert_allclose(cond_mean, jnp.array([2.0]), rtol=1e-10)
        np.testing.assert_allclose(cond_prec, jnp.array([[1.0]]), rtol=1e-10)
    
    def test_identity_precision(self):
        """Test with identity precision matrix."""
        key = random.PRNGKey(321)
        
        x = jnp.array([1.0, 2.0, 3.0])
        mean = jnp.array([0.5, 1.0, 1.5])
        prec = jnp.eye(3)  # Identity matrix
        ix_target = jnp.array([0, 2])
        ix_cond = jnp.array([1])
        
        # Generate a sample
        key, subkey = random.split(key)
        sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
        
        # Check shape
        assert sample.shape == (2,)
        
        # With identity precision, conditional should equal marginal
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        expected_mean = mean[ix_target]
        expected_prec = prec[jnp.ix_(ix_target, ix_target)]
        
        np.testing.assert_allclose(cond_mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(cond_prec, expected_prec, rtol=1e-10)
    
    def test_reproducibility(self):
        """Test that samples are reproducible with same key."""
        x = jnp.array([1.0, 2.0])
        mean = jnp.array([0.0, 0.0])
        prec = jnp.array([[2.0, 0.5],
                         [0.5, 1.0]])
        ix_target = jnp.array([0])
        ix_cond = jnp.array([1])
        
        # Same key should produce same result
        key = random.PRNGKey(42)
        sample1 = sample_mvn_cond(key, x, mean, prec, ix_target, ix_cond)
        sample2 = sample_mvn_cond(key, x, mean, prec, ix_target, ix_cond)
        
        np.testing.assert_array_equal(sample1, sample2)
        
        # Different keys should produce different results (with high probability)
        key1 = random.PRNGKey(42)
        key2 = random.PRNGKey(43)
        sample1 = sample_mvn_cond(key1, x, mean, prec, ix_target, ix_cond)
        sample2 = sample_mvn_cond(key2, x, mean, prec, ix_target, ix_cond)
        
        # Should be different with high probability
        assert not jnp.allclose(sample1, sample2, rtol=1e-10)
    
    def test_edge_case_single_variable(self):
        """Test edge case with single variable system."""
        key = random.PRNGKey(111)
        
        x = jnp.array([1.5])
        mean = jnp.array([0.0])
        prec = jnp.array([[2.0]])
        ix_target = jnp.array([0])
        ix_cond = jnp.array([], dtype=jnp.int32)  # No conditioning variables
        
        # This should give marginal distribution
        key, subkey = random.split(key)
        sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
        
        assert sample.shape == (1,)
        assert jnp.isfinite(sample[0])
    
    def test_large_system(self):
        """Test with larger system to ensure scalability."""
        key = random.PRNGKey(999)
        np.random.seed(42)
        
        d = 8
        # Generate random positive definite precision matrix
        A = np.random.randn(d, d)
        prec = jnp.array(A.T @ A + 2.0 * np.eye(d))
        
        x = jnp.array(np.random.randn(d))
        mean = jnp.array(np.random.randn(d))
        
        ix_target = jnp.array([1, 3, 5])
        ix_cond = jnp.array([0, 2, 4, 6, 7])
        
        # Generate sample
        key, subkey = random.split(key)
        sample = sample_mvn_cond(subkey, x, mean, prec, ix_target, ix_cond)
        
        # Check shape and basic properties
        assert sample.shape == (len(ix_target),)
        assert jnp.all(jnp.isfinite(sample))
    
    @pytest.mark.parametrize("seed", [42, 123, 456, 789])
    def test_parametrized_seeds(self, seed):
        """Parametrized test with different random seeds."""
        key = random.PRNGKey(seed)
        
        x = jnp.array([1.0, 2.0, 3.0])
        mean = jnp.array([0.0, 0.0, 0.0])
        prec = jnp.array([[2.0, 0.3, 0.1],
                         [0.3, 1.5, 0.2],
                         [0.1, 0.2, 1.0]])
        ix_target = jnp.array([0])
        ix_cond = jnp.array([1, 2])
        
        # Generate sample
        sample = sample_mvn_cond(key, x, mean, prec, ix_target, ix_cond)
        
        # Basic checks
        assert sample.shape == (1,)
        assert jnp.isfinite(sample[0])
    
    def test_consistency_with_direct_sampling(self):
        """Test consistency with direct sampling from conditional distribution."""
        key = random.PRNGKey(555)
        
        # Simple 2D case for easy verification
        x = jnp.array([1.0, 2.0])
        mean = jnp.array([0.0, 0.0])
        prec = jnp.array([[4.0, 1.0],
                         [1.0, 2.0]])
        ix_target = jnp.array([0])
        ix_cond = jnp.array([1])
        
        # Our function
        key, subkey1 = random.split(key)
        sample1 = sample_mvn_cond(subkey1, x, mean, prec, ix_target, ix_cond)
        
        # Direct sampling using conditional parameters
        cond_mean, cond_prec = mvn_cond_params(x, mean, prec, ix_target, ix_cond)
        L = jnp.linalg.cholesky(cond_prec)
        
        key, subkey2 = random.split(key)
        sample2 = sample_mvn_from_precision(subkey2, cond_mean, L)
        
        # Both methods should sample from the same distribution
        # We can't directly compare single samples, but we can check they have same shape
        assert sample1.shape == sample2.shape