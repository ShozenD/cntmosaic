import numpy as np
import pandas as pd

class ContactGenerator:
  allowed_models = ['poisson', 'negbin']
  
  def __init__(self,
               df_part: pd.DataFrame,
               cint_matrix: np.ndarray | list,
               model: str='poisson',
               odisp: float | None = None,
               rnd_eff_shape = 5,
               rnd_eff_rate = 5):
    assert model in self.allowed_models, f"Model '{model}' is not supported. Allowed models: {self.allowed_models}"
    
    self.df_part = df_part
    self.cint_matrix = cint_matrix # Note: If cint_matrix is a list, it must match the number of subgroups in df_part
    self.model = model
    self.odisp = odisp
    self.rnd_eff_shape = rnd_eff_shape
    self.rnd_eff_rate = rnd_eff_rate
    
  def _generate(self, df_part: pd.DataFrame, cint_matrix: np.ndarray, label: str = None):
    age = df_part['age_part'].astype(int).values
    lambda_ = cint_matrix[age, :]
    lambda_ *= np.random.gamma( # Add individual effects
      shape=self.rnd_eff_shape,
      scale=1/self.rnd_eff_rate,
      size=cint_matrix[age, :].shape
    )

    if self.model == 'poisson':
      samples = np.random.poisson(lambda_, size=(len(age), lambda_.shape[1]))
    elif self.model == 'negbin':
      if self.odisp is None:
        raise ValueError("Overdispersion parameter 'odisp' must be provided for negative binomial model.")
      n_success = 1 / self.odisp
      p_success = n_success / (n_success + lambda_)
      samples = np.random.negative_binomial(n_success,
                                            p_success,
                                            size=(len(age), lambda_.shape[1]))
      
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