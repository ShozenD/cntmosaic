import pickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import multinomial, poisson, multivariate_normal
from typing import Callable

class DataGenerator():
  def __init__(self,
               n: int,
               A: int,
               pop: NDArray,
               seed: int = 0,
               C: int = 1/200):
    self.n = n
    self.A = A
    self.pop = pop
    self.seed = seed
    self.C = C

  def get_rate(self):
    """Generate a contact pattern matrix using a mixture of 3 bivariate Gaussians"""
    X = np.arange(0, self.A, 1)
    Y = np.arange(0, self.A, 1)
    grid = np.array([[i, j] for i in X for j in Y])
    p_main = multivariate_normal.pdf(grid, mean=[15, 15], cov=[[90, 90*0.90], [90*0.90, 90]])
    p_sub1 = multivariate_normal.pdf(grid, mean=[20, 5], cov=[[90, 90*0.80], [90*0.80, 90]])
    p_sub2 = multivariate_normal.pdf(grid, mean=[5, 20], cov=[[90, 90*0.80], [90*0.80, 90]])

    p = (p_main + p_sub1 + p_sub2) / 3
    p = p / p.sum() # Re-normalise

    rate = (self.C * p).reshape(self.A, self.A)

    return rate
  
class DataGeneratorBasic(DataGenerator):
  def __init__(self,
               n: int,
               A: int,
               pop: NDArray,
               seed: int = 0,
               C: int = 1/200):
    super().__init__(n=n, A=A, pop=pop, seed=seed, C=C)
    self.sample_size = multinomial.rvs(n=self.n, p=self.pop/self.pop.sum())
    self.rate = self.get_rate()

  def generate(self):
    # Index for all pairs of age groups
    aidx = np.array([[i, j] for i in range(self.A) for j in range(self.A)])

    # Contact intensity
    self.cint = self.rate * self.pop[None,:]

    # Generate contacts
    mu = (self.cint * self.sample_size[:,None])[aidx[:,0], aidx[:,1]]
    self.y = poisson.rvs(mu=mu)

    # Create dataframe
    age_part = aidx[:,0]
    age_cnt = aidx[:,1]
    self.data = pd.DataFrame({
      'age_part': age_part,
      'age_cnt': age_cnt,
      'n': self.sample_size[age_part],
      'p': self.pop[age_cnt],
      'rate': self.rate[aidx[:,0], aidx[:,1]],
      'cint': self.cint[aidx[:,0], aidx[:,1]],
      'y': self.y
    })

    return self.data

class DataGeneratorStratified(DataGeneratorBasic):
  def __init__(self,
               n: int,
               A: int,
               pop: NDArray,
               seed: int = 0,
               C: int = 1/200):
    super().__init__(n=n, A=A, pop=pop, seed=seed, C=C)

  def set_subgroup_rates(self, func: Callable[..., NDArray], **kwargs) -> NDArray:
    """
    Set the contact rate for each subgroup using a specified function which applies some form
    of transformation to the base contact rate matrix.

    :param x: Base contact rate matrix
    :type x: NDArray
    :param func: Function to apply to the base contact rate matrix. The function should take the base contact rate matrix as input and return a matrix with dimensions (n_subgroups, n_age_groups, n_age_groups)
    :type func: Callable[[NDArray], NDArray]
    :param kwargs: Additional arguments to pass to the function
    :type kwargs: Dict
    """
    self.rate_subgroups = func(self.rate, **kwargs)

  def set_subgroup_sample_sizes(self, func: Callable[..., NDArray], **kwargs) -> NDArray:
    """
    Set the sample size of each subgroup using a specified function which splits the
    base sample size into several subgroups.

    :param x: Base sample size by age group
    :type x: NDArray
    :param func: Function to split the sample size into subgroups. The function should take the base sample size as input and return a matrix with dimensions (n_subgroups, n_age_groups)
    :type func: Callable[[NDArray], NDArray]
    :param kwargs: Additional arguments to pass to the function
    :type kwargs: Dict
    """
    self.sample_size_subgroups = func(self.sample_size, **kwargs)

  def set_subgroup_populations(self, func: Callable[..., NDArray], **kwargs) -> NDArray:
    """
    Set the population size of each subgroup using a specified function which splits the
    base population into several subgroups. 

    :param x: Base population size by age group
    :type x: NDArray
    :param func: Function to split the population into subgroups. The function should take the base population as input and return a matrix with dimensions (n_subgroups, n_age_groups)
    :type func: Callable[[NDArray], NDArray]
    :param kwargs: Additional arguments to pass to the function
    :type kwargs: Dict
    """
    self.pop_subgroups = func(self.pop, **kwargs)

  def generate(self):
    K, A, B = self.rate_subgroups.shape

    idx = np.array([[k, i, j] for k in range(K) for i in range(A) for j in range(B)])
    self.cint_subgroups = self.rate_subgroups * self.pop[None,None,:]
    self.mu = self.cint_subgroups * self.sample_size_subgroups[:,:,None]
    self.y = poisson.rvs(mu=self.mu[idx[:,0], idx[:,1], idx[:,2]])

    self.data = pd.DataFrame({
      'subgroup': idx[:,0],
      'age_part': idx[:,1],
      'age_cnt': idx[:,2],
      'n': self.sample_size_subgroups[idx[:,0], idx[:,1]],
      'p': self.pop[idx[:,2]],
      'rate': self.rate_subgroups[idx[:,0], idx[:,1], idx[:,2]],
      'cint': self.cint_subgroups[idx[:,0], idx[:,1], idx[:,2]],
      'y': self.y
    })

    return self.data
  
  def to_pickle(self, path: str):
    data = {
      'data': self.data,
      'pop': self.pop_subgroups 
    }
    with open(path, 'wb') as f:
      pickle.dump(data, f)