import itertools
import re
import numpy as np
import pandas as pd

import re
import pandas as pd

def as_interval_type(s):
    """Parse interval string to Interval. Return None if input is None, not a string, or improperly formatted."""
    # Convert non-string inputs to string (except None which directly returns None)
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)

    # Remove all spaces
    s = s.replace(' ', '')

    # Validate the format of the string to ensure it is suitable for interval parsing
    if not re.match(r'^[\[\(]\d+,\d+[\]\)]$', s):
        return None  # Return None if the format does not match expected pattern

    # Extract numbers using regex and convert them to integers
    left, right = map(int, re.findall(r'\d+', s))

    # Determine the closure of the interval using the first and last character
    t = ('both' if s[0] == '[' and s[-1] == ']' else
         'left' if s[0] == '[' and s[-1] == ')' else
         'right' if s[0] == '(' and s[-1] == ']' else
         'neither')  # for the case '(a,b)'

    return pd.Interval(left, right, closed=t)



def expand_age_interval(df: pd.DataFrame,
                        interval_col: str,
                        name: str = 'age_expanded') -> pd.DataFrame:
    
    if df[interval_col].dtype.name == 'object':
        df[interval_col] = df[interval_col].apply(as_interval_type)
    
    expanded_rows = []
    dtype_dict = df.dtypes  # Store the original dtypes
    
    for _, row in df.iterrows(): # Note: Iterrows is slow, but more readable
        age_grp = row[interval_col]
        age_range = range(int(age_grp.left), int(age_grp.right))
        for age in age_range:
            new_row = row.copy()
            new_row[name] = age
            expanded_rows.append(new_row)
    
    # Create a new DataFrame from expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Set categorical dtypes explicitly
    for col, dtype in dtype_dict.items():
        if dtype.name.startswith('category'):
            expanded_df[col] = pd.Categorical(expanded_df[col], categories=df[col].cat.categories, ordered=df[col].cat.ordered)

    return expanded_df

def check_required_columns(data: pd.DataFrame):
	if 'y' not in data.columns:
		raise ValueError("data must contain a column contact count column 'y'")
	if 'N' not in data.columns:
		raise ValueError("data must contain a column sample size column 'N'")
	if 'age_part' not in data.columns:
		raise ValueError("data must contain a column 'age_part'")
	if ('age_cnt' not in data.columns) and ('age_grp_cnt' not in data.columns):
		raise ValueError("data must contain a column 'age_cnt' or 'age_grp_cnt'")

def expand_grid(data_dict):
    """Create a dataframe from a dictionary of lists. Analogous to R's expand.grid."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def make_full_grid(data: pd.DataFrame,
                   age_vars: list[str],
                   grp_vars: list[str] | None=None) -> pd.DataFrame:
    """Create a full grid of all possible combinations of age and grouping variables.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data containing necessary columns.
    age_vars : list[str]
        List of age variables.
    grp_vars : list[str] or None, optional
        List of non-age grouping variables. default is None.
        
    Returns
    -------
    pd.DataFrame
        Full grid of age and grouping variables.
    """
    
    if grp_vars is None:
        grp_vars_all = age_vars
    else:
        grp_vars_all = age_vars + grp_vars
        
    data_dict = {k: data[k].unique() for k in grp_vars_all}
    
    if 'age_cnt' == age_vars[1]:
        
        min_age = np.min([data_dict['age_part'].min(), data_dict['age_cnt'].min()])
        max_age = np.max([data_dict['age_part'].max(), data_dict['age_cnt'].max()])
        
        data_dict['age_part'] = np.arange(min_age, max_age + 1)
        data_dict['age_cnt'] = np.arange(min_age, max_age + 1)
        
    elif 'age_grp_cnt' == age_vars[1]:
        data_dict['age_grp_cnt'] = data['age_grp_cnt'].cat.categories

    return expand_grid(data_dict)