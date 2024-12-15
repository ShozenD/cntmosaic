from importlib import resources
import pickle

def load_pickle_data(data_file_name):
  """Loads `data_file_name` from the package's data directory."""
  
  data_path = resources.files('cntmosaic.datasets.data') / data_file_name
  with data_path.open('rb') as pickle_file:
    return pickle.load(pickle_file)
  
def load_polymod_germany():
  """Loads the German Polymod dataset.
  
  This function loads a cleaned version of the German POLYMOD dataset.
  
  Returns
  -------
  dict
      A dictionary with the following
      - 'contacts': a pandas DataFrame with the contact data
      - 'participants': a pandas DataFrame with the participant data
      - 'population': a pandas DataFrame with the population data
  """
  
  return load_pickle_data('polymod_germany.pkl')

def load_covimod():
  """Loads the COVIMOD dataset.
  
  This function loads the Covimod dataset.
  
  Returns
  -------
  dict
      A dictionary with the following
      - 'contacts': a pandas DataFrame with the contact data
      - 'participants': a pandas DataFrame with the participant data
      - 'population': a pandas DataFrame with the population data
  """
  
  return load_pickle_data('covimod.pkl')