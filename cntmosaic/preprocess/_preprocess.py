import warnings
import pandas as pd
import numpy as np

from ._utils import make_full_grid

def convert_to_categorical(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Convert columns to categorical type."""
    for col in cols:
        data[col] = pd.Categorical(data[col])
    return data

def document_categories(data: pd.DataFrame) -> dict:
    """Document the categories of categorical columns."""
    return {col: data[col].cat.categories for col in data.select_dtypes(include='category').columns}

def impute_age_min_max(data: pd.DataFrame,
                       age_col: str,
                       age_min_col: str,
                       age_max_col: str,
                       dropna: bool=False,
                       remove_min_max_col: bool=True) -> pd.DataFrame:
    """
    Impute missing values in an age column based on the average of corresponding 
    minimum and maximum age columns and optionally clean up the DataFrame.

    This function fills missing values in the specified 'age_col' by computing the 
    average of the 'age_min_col' and 'age_max_col'. It can also remove rows where 
    'age_col' remains NaN after imputation if specified. Additionally, it offers the 
    option to remove the minimum and maximum age columns from the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the age data.
    age_col : str
        The name of the column in `data` where the age (or imputed age) is stored.
    age_min_col : str
        The name of the column containing the minimum age values.
    age_max_col : str
        The name of the column containing the maximum age values.
    dropna : bool, optional
        If True, rows where 'age_col' is NaN after imputation are dropped. Default is False.
    remove_min_max_col : bool, optional
        If True, the columns specified by 'age_min_col' and 'age_max_col' are removed 
        from the DataFrame after processing. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the imputed age column. Depending on the options, it may have 
        fewer rows and columns.

    Raises
    ------
    Warning
        Warns about the number of dropped rows if any rows are removed due to NaN values in 'age_col'.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'age': [None, 25, None, 30],
    ...     'age_min': [20, 25, 40, 30],
    ...     'age_max': [30, 25, 50, 30]
    ... })
    >>> imputed_df = impute_age_min_max(df, 'age', 'age_min', 'age_max', dropna=True, remove_min_max_col=True)
    >>> print(imputed_df)
       age
    1   25
    3   30

    Note
    ----
    After imputation, if 'dropna' is True and 'age_col' still contains NaN, such rows will be dropped,
    potentially reducing the size of the returned DataFrame.
    """
    data = data.copy()
    data[age_col] = np.where(data[age_col].isna(), 
                             (data[age_min_col] + data[age_max_col]) // 2,
                             data[age_col])
    
    if dropna:
        n0 = data.shape[0]
        data = data.dropna(subset=[age_col])
        n1 = data.shape[0]
        Warning(f'Dropped {n0 - n1} rows with missing values in {age_col}')
    
    if remove_min_max_col:
        data = data.drop(columns=[age_min_col, age_max_col], axis=1)
    
    return data


def make_train_data(data: pd.DataFrame,
                    id_var: str,
                    grp_vars: str | list[str] | None=None) -> pd.DataFrame:
    """
    Prepare training data for Bayesian Rate Consistency model.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing necessary columns.
    id_var : str
        Name of the column containing unique individual identifiers.
    grp_vars : str or list[str], optional
        Additional grouping variables for stratification, by default None.

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
    
    data = data.copy()
    
    # Drop rows with missing values
    n_all = data.shape[0]
    data = data.dropna(axis=0)
    n_dropped = n_all - data.shape[0]
    if n_dropped > 0:
        warnings.warn(f'Dropped {n_dropped} rows with missing values', RuntimeWarning)
    
    # Add a column 'y' if it does not exist
    if 'y' not in data.columns:
        warnings.warn('No column "y" found. Assuming each row represents a single contact.', RuntimeWarning)
        data['y'] = 1
            
    if 'age_cnt' in data.columns:
        age_vars = ['age_part', 'age_cnt']
    elif 'age_grp_cnt' in data.columns:
        age_vars = ['age_part', 'age_grp_cnt']
    else:
        raise ValueError('data must contain a column "age_cnt" or "age_grp_cnt"')
    
    if grp_vars is None:
        selected_vars = [id_var] + age_vars + ['y']
        grp_vars_part = grp_vars_cnt = ['age_part']
    else:
        grp_vars = [grp_vars] if isinstance(grp_vars, str) else grp_vars
        selected_vars = [id_var] + age_vars + grp_vars + ['y']
        grp_vars_part = ['age_part'] + grp_vars
        grp_vars_cnt = age_vars + grp_vars
            
    data = data[selected_vars]
    
    # Convert non-numeric columns to categorical
    non_numeric_cols = data[grp_vars_cnt].select_dtypes(include='object').columns
    data = convert_to_categorical(data, non_numeric_cols)
    
    # Document the categories, intervals
    dict_cats = document_categories(data)
    
    # ===== Calculate N and y =====
    df_N = data.groupby(grp_vars_part, observed=False).agg(N = (id_var, 'nunique')).reset_index()
    df_y = data.groupby(grp_vars_cnt, observed=False).agg(y = ('y', 'sum')).reset_index()
    
    # ===== Create the grid =====
    df_grid = make_full_grid(data, age_vars, grp_vars)
    
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
    if 'age_cnt' in df.columns:
        df['age_cnt'] = df['age_cnt'].astype(int)
    
    # Bring ['age_part', 'age_cnt'] to the front
    age_part = df.pop('age_part')
    df.insert(0, 'age_part', age_part)
    
    age_cnt = df.pop(age_vars[1])
    df.insert(1, age_vars[1], age_cnt)
    df.sort_values(age_vars, inplace=True)
    
    # Convert to categorical
    for col in dict_cats.keys():
        df[col] = pd.Categorical(df[col], categories=dict_cats[col])
    
    return df

def make_grp_cnt_offsets(df_cnt: pd.DataFrame,
                        df_grp: pd.DataFrame,
                        grp_vars: str | list[str] | None=None,
                        max: int | None=None) -> pd.DataFrame:
    """Create offsets for group contacts and contacts with missing information.
    
    Suppose grp_vars = ['sex_part']. Given the total number of contacts with complete information :math:`Y^g_a = sum_{b} Y^{g}_{a,b}`
    and the total number of contacts with missing information :math:`Z^g_a`, the offset for group contacts is calculated as
    
    ..math::
		S^g_a = \frac{Y^g_a}{Y^g_a + Z^g_a}
	
	If S is missing or 0, it is replaced with 1 to avoid missing value and log(0) = -inf errors.
	
    Parameters
    ----------
    df_cnt : pd.DataFrame
		Data frame containing the contact counts of contacts with complete information.
  		Usually, this is the output of the make_train_data function.
	df_grp : pd.DataFrame
		Data frame containing information for each individual and the number of group & contacts with missing information
		in a column named `z`.
	grp_vars : str | list | None, optional
		Column(s) to group by. default is None.
	max : int | None, optional
		Maximum number of group & missing contacts allowed. This is useful for avoiding offsets that are too large.
  
    Returns
	-------
	pd.DataFrame
		A data frame containing the offsets.
  
	Examples
	--------
	>>> df_cnt = make_train_data(df, 'id', ['sex_part'])
	>>> df_grp = df_part
	>>> df_grp['z'] = df_grp['class_size'] + df_grp['work_contacts_nr'] # Group contacts for school and work
	>>> offsets = make_group_cnt_offsets(df_cnt, df_grp, 'sex_part', max=60)
    """
    
    if 'y' not in df_cnt.columns:
        warnings.warn('No column "y" found. Assuming each row represents a single contact.', RuntimeWarning)
        df_cnt['y'] = 1
    
    if 'z' not in df_grp.columns:
        raise RuntimeError('No column "z" found. Please provide the number of group contacts in a column named "z"')
    
    if grp_vars is None:
        grp_vars = ['age_part']
    else:
        grp_vars = [grp_vars] if isinstance(grp_vars, str) else grp_vars
        grp_vars = ['age_part'] + grp_vars
    
    df_cnt_part = df_cnt.groupby(grp_vars, observed=True).agg({'y': 'sum'}).reset_index()
    df_grp_sum = df_grp.groupby(grp_vars, observed=True).agg({'z': 'sum'}).reset_index()
    
    if max is not None: # Cap the sum of group contacts
        df_grp_sum['z'] = df_grp_sum['z'].apply(lambda x: min(x, max))
    
    df_offsets = pd.merge(df_cnt_part, df_grp_sum, on=grp_vars, how='left')
    df_offsets['S'] = df_offsets['y'] / (df_offsets['y'] + df_offsets['z'])
    df_offsets['S'] = df_offsets['S'].fillna(0)
    df_offsets['S'] = np.where(df_offsets['S'] == 0, 1, df_offsets['S']) # Avoid log(0) = -inf errors
    
    cols = grp_vars + ['S']
    return df_offsets[cols]