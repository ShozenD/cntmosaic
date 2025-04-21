import numpy as np
import pandas as pd

class ParticipantGenerator:
    def __init__(self, n: int | list, age_dist: np.ndarray | list):
        self.age_dist = age_dist
        self.n = n
        
    def _generate(self, n: int, age_dist: np.ndarray) -> pd.DataFrame:
        total = age_dist.sum()
        prop = age_dist / total
        age = np.random.choice(np.arange(len(age_dist)), size=n, p=prop)
        
        return pd.DataFrame({'age_part': age})
    
    def generate(self, seed: int = 0) -> pd.DataFrame:
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        if isinstance(self.n, int):
          df_part = self._generate(self.n, self.age_dist)
        else:
          dfs = []
          for i in range(len(self.n)):
              df = self._generate(self.n[i], self.age_dist[i])
              df['subgroup'] = i
              dfs.append(df)
          
          df_part = pd.concat(dfs)
     
        # Assign a unique ID to each participant
        df_part['id'] = np.arange(df_part.shape[0]) + 1
        
        # Bring ID column to the front
        cols = df_part.columns.tolist()
        cols.insert(0, cols.pop(cols.index('id')))
        df_part = df_part[cols]
    
        self.data = df_part

        return self.data