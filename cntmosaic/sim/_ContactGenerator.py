import numpy as np
import pandas as pd

class ContactGenerator:
  def __init__(self, df_part: pd.DataFrame, cint_matrix: np.ndarray | list):
    self.df_part = df_part
    self.cint_matrix = cint_matrix # Note: If cint_matrix is a list, it must match the number of subgroups in df_part
    
  def _generate(self, df_part: pd.DataFrame, cint_matrix: np.ndarray, label: str = None):
    age = df_part['age_part'].astype(int).values
    lambda_ = cint_matrix[age, :]
    samples = np.random.poisson(lambda_, size=(len(age), lambda_.shape[1]))
    positions = [np.where(row > 0)[0] for row in samples]
    
    data = []
    for j, pos in enumerate(positions):
      for p in pos:
        if label is not None:
          data.append([df_part['id'].values[j], p, label, samples[j, p]])
        else:
          data.append([df_part['id'].values[j], p, samples[j, p]])
        
    if label is not None:
      return pd.DataFrame(data, columns=['id', 'age_cnt', 'subgroup', 'y'])
    else:
      return pd.DataFrame(data, columns=['id', 'age_cnt', 'y'])
    
  def generate(self, seed: int = 0):
    np.random.seed(seed)

    if isinstance(self.cint_matrix, list):
      subgroups = self.df_part['subgroup'].unique()
      dfs = []
      for i, subgroup in enumerate(subgroups):
        df_part_sub = self.df_part[self.df_part['subgroup'] == subgroup]
        dfs.append(
          self._generate(df_part_sub, self.cint_matrix[i], label=subgroup) # TODO: Enable different sampling distributions
        )
      self.data = pd.concat(dfs, ignore_index=True)
    else:
      self.data = self._generate(self.df_part, self.cint_matrix)  
    
    return self.data