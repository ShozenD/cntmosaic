import numpy as np
import pandas as pd

class ParticipantGenerator:
  def __init__(self, age_dist: np.ndarray | list[np.ndarray]):
    self.age_dist = age_dist
        
  def _generate(self, n: int, age_dist: np.ndarray) -> pd.DataFrame:
    total = age_dist.sum()
    prop = age_dist / total
    age = np.random.choice(np.arange(len(age_dist)), size=n, p=prop)
      
    return pd.DataFrame({'age_part': age})
    
  def generate(self, n: int, seed: int = 0) -> pd.DataFrame:
    # Set random seed for reproducibility
    np.random.seed(seed)
      
    if isinstance(n, int) and isinstance(self.age_dist, list):
      dfs = []
      for i in range(len(self.age_dist)):
        df = self._generate(n, self.age_dist[i])
        df['subgroup'] = i
        dfs.append(df)
        
    elif isinstance(n, list) and isinstance(self.age_dist, np.ndarray):
      dfs = []
      for i in range(len(n)):
          df = self._generate(n[i], self.age_dist)
          df['subgroup'] = i
          dfs.append(df)
          
    elif isinstance(n, list) and isinstance(self.age_dist, list):
      assert len(n) == len(self.age_dist), "n and age_dist must have the same length"
      dfs = []
      for i in range(len(n)):
          df = self._generate(n[i], self.age_dist[i])
          df['subgroup'] = i
          dfs.append(df)
          
    elif isinstance(n, int) and isinstance(self.age_dist, np.ndarray):
      dfs = [self._generate(n, self.age_dist)]
      
    else:
      raise ValueError("n must be an int or a list of ints, and age_dist must be a numpy array or a list of numpy arrays")
       
    df_part = pd.concat(dfs)
   
    # Assign a unique ID to each participant
    df_part['id'] = np.arange(df_part.shape[0]) + 1
      
    # Bring ID column to the front
    cols = df_part.columns.tolist()
    cols.insert(0, cols.pop(cols.index('id')))
    df_part = df_part[cols]
  
    self.data = df_part
      
    return self.data