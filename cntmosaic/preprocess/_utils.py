import pandas as pd

def expand_age_grp_cnt(df: pd.DataFrame):
    expanded_rows = []
    dtype_dict = df.dtypes  # Store the original dtypes
    
    for _, row in df.iterrows(): # Note: Iterrows is slow, but more readable
        age_grp = row['age_grp_cnt']
        age_range = range(int(age_grp.left), int(age_grp.right))
        for age in age_range:
            new_row = row.copy()
            new_row['age_cnt'] = age
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