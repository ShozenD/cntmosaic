import numpy as np
from numpy.typing import NDArray

import pandas as pd

class ParticipantGenerator:
  def __init__(
    self,
    sample_age_dist: NDArray | list[NDArray] | dict[NDArray]= None,
    sample_age_prop: NDArray | list[NDArray] | dict[NDArray] = None
  ):
    """
    Generate participant data based on age distributions or age proportions.
    
    This class creates synthetic participant datasets by sampling from specified age 
    distributions. It supports single populations as well as multiple subgroups, making 
    it useful for generating stratified participant samples for contact studies or 
    demographic simulations.
    
    Parameters
    ----------
    sample_age_dist : NDArray, list of NDArray, or dict of NDArray, optional
      The age distribution(s) to sample from. Can be:
      - NDArray: A single age distribution for a homogeneous population
      - list of NDArray: Multiple age distributions, one per subgroup (indexed 0, 1, ...)
      - dict of NDArray: Multiple age distributions with custom subgroup labels as keys
      
      The distributions will be automatically normalized to proportions.
      
    sample_age_prop : NDArray, list of NDArray, or dict of NDArray, optional
      The age proportion(s) to sample from (already normalized). Follows the same 
      structure as sample_age_dist. Use this parameter if you already have normalized 
      proportions and want to avoid redundant normalization.
    
    Notes
    -----
    Either `sample_age_dist` or `sample_age_prop` must be provided, but not both.
    If `sample_age_dist` is provided, it will be normalized internally to proportions.
    
    Examples
    --------
    >>> import numpy as np
    >>> from cntmosaic.sim import ParticipantGenerator
    
    **Example 1: Single population**
    
    Generate participants from a single age distribution:
    
    >>> age_dist = np.array([100, 200, 300, 400, 500])  # Counts by age group
    >>> pg = ParticipantGenerator(age_dist)
    >>> df_participants = pg.generate(n=1000, seed=42)
    >>> print(df_participants.head())
       id  age_part
    0   1         3
    1   2         4
    2   3         2
    3   4         3
    4   5         2
    
    **Example 2: Multiple subgroups with list**
    
    Generate participants from multiple subgroups (e.g., different regions):
    
    >>> age_dist_urban = np.array([150, 250, 350, 250, 100])
    >>> age_dist_rural = np.array([100, 150, 200, 300, 250])
    >>> pg = ParticipantGenerator([age_dist_urban, age_dist_rural])
    >>> df_participants = pg.generate(n=500, seed=42)
    >>> print(df_participants.head())
       id  age_part  subgroup
    0   1         2         0
    1   2         3         0
    2   3         1         0
    3   4         2         0
    4   5         2         0
    >>> print(df_participants['subgroup'].value_counts())
    subgroup
    0    500
    1    500
    
    **Example 3: Multiple subgroups with custom labels**
    
    Generate participants with named subgroups:
    
    >>> age_dists = {
    ...     'healthcare': np.array([50, 200, 300, 250, 100]),
    ...     'education': np.array([100, 250, 300, 200, 50])
    ... }
    >>> pg = ParticipantGenerator(age_dists)
    >>> df_participants = pg.generate(n=300, seed=42)
    >>> print(df_participants['subgroup'].value_counts())
    subgroup
    healthcare    300
    education     300
    
    **Example 4: Using pre-normalized proportions**
    
    If you already have normalized proportions:
    
    >>> age_prop = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    >>> pg = ParticipantGenerator(sample_age_prop=age_prop)
    >>> df_participants = pg.generate(n=1000, seed=42)
    """
    if sample_age_dist is not None:
      self.sample_age_dist = sample_age_dist
      
      if isinstance(sample_age_dist, list):
        self.sample_age_prop = [dist / dist.sum() for dist in sample_age_dist]
      elif isinstance(sample_age_dist, dict):
        self.sample_age_prop = {key: dist / dist.sum() for key, dist in sample_age_dist.items()}
      else:
        self.sample_age_prop = sample_age_dist / sample_age_dist.sum()
    elif sample_age_prop is not None:
      self.sample_age_prop = sample_age_prop
    else:
      raise ValueError("Either sample_age_dist or sample_age_prop must be provided.")

  @staticmethod
  def _generate(n: int, age_prop: NDArray) -> pd.DataFrame:
    """
    Generate a DataFrame of participant ages based on the provided age distribution.

    Parameters
    ----------
    n: int
      The number of participants to generate.
    age_prop: NDArray
      The age distribution to sample from.

    Returns
    -------
    pd.DataFrame
      A DataFrame containing the generated participant ages.
    """
    age = np.random.choice(np.arange(len(age_prop)), size=n, p=age_prop)

    return pd.DataFrame({'age_part': age})
    
  def generate(self, n: int, seed: int = 0) -> pd.DataFrame:
    """
    Generate participant data.

    Parameters
    ----------
    n: int
      The total number of participants in the study.
    seed: int
      The random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
      A DataFrame containing the generated participant data.
      Columns include 'id', 'age_part', and optionally 'subgroup'.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    dfs = []
    if isinstance(n, int) and isinstance(self.sample_age_prop, list):
      for i in range(len(self.sample_age_prop)):
        df = self._generate(n, self.sample_age_prop[i])
        df['subgroup'] = i
        dfs.append(df)
    elif isinstance(n, int) and isinstance(self.sample_age_prop, dict):
      for key, prop in self.sample_age_prop.items():
          df = self._generate(n, prop)
          df['subgroup'] = key
          dfs.append(df)
    elif isinstance(n, int) and isinstance(self.sample_age_prop, np.ndarray):
      dfs = [self._generate(n, self.sample_age_prop)]
    else:
      raise ValueError("n must be of int type and sample_age_prop must be a NDArray, a list of NDArray arrays, or a dict of NDArray arrays.")
       
    df_part = pd.concat(dfs)
   
    # Assign a unique ID to each participant
    df_part['id'] = np.arange(df_part.shape[0]) + 1
      
    # Bring ID column to the front
    cols = df_part.columns.tolist()
    cols.insert(0, cols.pop(cols.index('id')))
    df_part = df_part[cols]

    return df_part