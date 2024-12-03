import itertools
import pandas as pd
import numpy as np

def expand_grid(data_dict):
    """Create a dataframe from a dictionary of lists. Analogous to R's expand.grid."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def convert_to_categorical(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to categorical type."""
    for col in cols:
        data[col] = pd.Categorical(data[col])
    return data

def document_categories(data: pd.DataFrame) -> dict:
    """Document the categories of categorical columns."""
    return {col: data[col].cat.categories for col in data.select_dtypes(include='category').columns}

def make_train_data(data: pd.DataFrame,
                    id_var: str,
                    grp_vars: list[str]=[]) -> pd.DataFrame:
    """
    Prepare training data for Bayesian Rate Consistency model.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing necessary columns.
    id_var : str
        Name of the column containing unique individual identifiers.
    grp_vars : list[str], optional
        Additional grouping variables for stratification, by default [].

    Returns
    -------
    pd.DataFrame
        Processed training data.
        
    Examples
    --------
    Basic usage with no grouping variables:
    
    >>> df_train = make_train_data(data, 'id')
    
    Usage with grouping variables
    
    >>> df_train = make_train_data(data, 'id', ['sex', 'ses'])
    """
    
    # Isolate the necessary columns
    data = data.copy()
    selected_vars = [id_var, 'age_part', 'age_cnt'] + grp_vars + ['y']
    data = data[selected_vars]
    
    # Convert non-numeric columns to categorical
    non_numeric_cols = data.select_dtypes(include='object').columns
    data = convert_to_categorical(data, non_numeric_cols)
    
    # Document the categories
    dict_cats = document_categories(data)
    
    # ===== Calculate N and y =====
    grp_vars_part = ['age_part'] + grp_vars
    df_N = data.groupby(grp_vars_part, observed=False).agg(N = (id_var, 'nunique')).reset_index()
    
    grp_vars_cnt = ['age_part', 'age_cnt'] + grp_vars
    df_y = data.groupby(grp_vars_cnt, observed=False).agg(y = ('y', 'sum')).reset_index()
    
    # ===== Create the grid =====
    data_dict = {k: data[k].unique() for k in grp_vars_part}
    data_dict['age_cnt'] = np.arange(np.array(data_dict['age_part']).max() + 1)
    df_grid = expand_grid(data_dict)
    
    # ===== Merge the dataframes =====
    df = pd.merge(df_grid, df_y, on=grp_vars_cnt, how='left')
    df = pd.merge(df, df_N, on=grp_vars_part, how='left')
    
    # ===== Finalise the data =====
    df.dropna(subset=['N'], inplace=True)
    df = df[df['N'] != 0]
    df['y'] = df['y'].fillna(0)
    df['y'] = df['y'].astype(int)
    df['N'] = df['N'].astype(int)
    df['age_part'] = df['age_part'].astype(int)
    df['age_cnt'] = df['age_cnt'].astype(int)
    
    # Bring ['age_part', 'age_cnt'] to the front
    age_part = df.pop('age_part')
    df.insert(0, 'age_part', age_part)
    age_cnt = df.pop('age_cnt')
    df.insert(1, 'age_cnt', age_cnt)
    
    # Sort by ['age_part', 'age_cnt']
    df.sort_values(['age_part', 'age_cnt'], inplace=True)
    
    # Convert to categorical
    for col in dict_cats.keys():
        df[col] = pd.Categorical(df[col], categories=dict_cats[col])
    
    return df
  