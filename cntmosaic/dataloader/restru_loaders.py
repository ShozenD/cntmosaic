import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional
from dataclasses import dataclass
import itertools
from sparse import COO
import warnings



@dataclass(frozen=True)
class CoordToColumns:
    age_part: str
    age_cnt: Optional[str] = None
    age_grp_cnt: Optional[str] = None
    id_var: str = 'id'
    y: Optional[str] = 'y'
    grp_vars: Optional[list[str] | str] = None
    

    def age_vars(self):
        if self.age_cnt:
            return [self.age_part, self.age_cnt]
        elif self.age_grp_cnt:
            return [self.age_part, self.age_grp_cnt]
        else:
            raise ValueError('One of age_cnt or age_grp_cnt must be provided')

    def __post_init__(self):
        if isinstance(self.grp_vars, str):
            object.__setattr__(self, 'grp_vars', [self.grp_vars])
        elif self.grp_vars is None:
            object.__setattr__(self, 'grp_vars', [])


class GeneralLoader:
    def load(self, sparse=False) -> xr.Dataset:
        """Convert selected columns from DataFrame to Xarray Dataset."""
        cols_needed = [self.col_map.y,
                       self.col_map.id_var] + self.col_map.age_vars()

        if self.col_map.grp_vars:
            cols_needed.extend(self.col_map.grp_vars)

        data = self.raw_df[cols_needed].dropna().copy()

        # Convert object columns to categorical
        for col in data.select_dtypes(include='object').columns:
            data[col] = pd.Categorical(data[col])

        # Store category info for future use
        self.categories = {
            col: data[col].cat.categories
            for col in data.select_dtypes(include='category').columns
        }

        cols_needed.remove(self.col_map.y)
        dim_vals = {var: data[var].values for var in cols_needed}
        if self.col_map.age_cnt:
            min_age = np.min([data[self.col_map.age_part].min(), data[self.col_map.age_cnt].min()])
            max_age = np.max([data[self.col_map.age_part].max(), data[self.col_map.age_cnt].max()])

            dim_vals[self.col_map.age_part] = np.arange(min_age, max_age + 1, dtype=int)
            dim_vals[self.col_map.age_cnt] = np.arange(min_age, max_age + 1, dtype=int)
        
        panel = data.groupby(cols_needed).sum().reset_index()
        expanded_dims = self.col_map.age_vars()
        dim_vals[self.col_map.age_cnt] = np.arange(min_age, max_age + 1, dtype=int)
        
        panel = data.groupby(cols_needed).sum().reset_index()
        expanded_dims = self.col_map.age_vars()
        static_dims = [v for v in cols_needed if v not in expanded_dims]

        # Cartesian product only for expanded_dims
        exp_vals = {k: dim_vals[k] for k in expanded_dims}
        full_index = list(itertools.product(*exp_vals.values()))
        df_full_exp = pd.DataFrame(full_index, columns=expanded_dims)

        # Combine with unique static dims (e.g., id, sex)
        df_static = panel[static_dims].drop_duplicates().reset_index(drop=True)

        # Merge to get the full grid more efficiently
        df_full = df_full_exp.merge(df_static, how="cross")

        df_full = df_full.merge(panel, on=cols_needed, how='left')
        df_full[self.col_map.y] = df_full[self.col_map.y].astype('Int64')
        ds_out = df_full.set_index(cols_needed).to_xarray()
        '''
        for var in cols_needed:
            ds_out.coords[var] = ds_out[var]
        '''    
        if sparse:
            dense_y = ds_out[self.col_map.y].values
            sparse_y = COO.from_numpy(dense_y)
            ds_out[self.col_map.y] = xr.DataArray(
                sparse_y,
                dims=ds_out[self.col_map.y].dims,
                coords=ds_out[self.col_map.y].coords,
                name=self.col_map.y
            )

        self.ds = ds_out

    def stratify(self):
        pass


class RawLoader(GeneralLoader):
    '''
    Input:
        part: a dataframe containing information on participants only
        cnt: a dataframe containing information on contacts only
        col_map: a CoordToColumns object for mapping columns
    Assumes a common id_var in both participants and contacts
    Assumes no other duplicate columns, issue warnings if duplicated

    '''
    def __init__(self, part: pd.DataFrame, cnt: pd.DataFrame, col_map: CoordToColumns):
    
        common_cols = set(part.columns).intersection(set(cnt.columns))
        if not(col_map.id_var.split('.')[-1] in common_cols):
            raise KeyError('id_var needs to be present in both dataframes')
        part_cols = [col_map.age_part.split('.')[-1], col_map.id_var.split('.')[-1]]
        cnt_cols = [col_map.age_cnt.split('.')[-1] or col_map.age_grp_cnt.split('.')[-1]]
        cnt_cols.append(col_map.id_var.split('.')[-1])
        grp_vars = []
        for var in col_map.grp_vars:
            v = var.split('.')[-1]
            grp_vars.append(v)
            if v in common_cols:
                warnings.warn(f"Duplicated column name {var} found, merged as separate columns with suffixes", UserWarning)
                part_cols.append(v)
                cnt_cols.append(v)
            elif v in part.columns():
                part_cols.append(v)
            elif v in cnt.columns():
                cnt_cols.append(v)
            else:
                raise KeyError(f'Column {v} not found in either dataframes')            
        y = col_map.y.split('.')[-1]
        if y in cnt.columns:
            cnt_cols.append(y)
        elif y in part.columns:
            part_cols.append(y)
        else:
            cnt_cols.append(y)
            cnt['y'] = 1
        self.col_map = CoordToColumns(y=col_map.y, 
                                        age_part=part_cols[0], 
                                        age_cnt=col_map.age_cnt, 
                                        age_grp_cnt=col_map.age_grp_cnt, 
                                        id_var=part_cols[1],
                                        grp_vars=grp_vars)
        raw_df = pd.merge(part[part_cols], cnt[cnt_cols], on=self.col_map.id_var,suffixes=('_part', '_cnt'))
        cols = list(raw_df.columns)
        cols.remove(self.col_map.y)
        self.raw_df = raw_df.groupby(cols).sum()[self.col_map.y].reset_index()

class MergedLoader(GeneralLoader):
    '''
    Input:
        df: a pandas dataframe containing necessary columns
        col_map: a CoordToColumns object for mapping columns
    Assumes info on participants and contacts have been merged
    '''
    def __init__(self, df: pd.DataFrame, col_map: CoordToColumns):
        if not col_map.y in df.columns:
            df['y'] = 1
        self.raw_df = df.copy()
        self.col_map = col_map