import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional
from dataclasses import dataclass
import itertools
from sparse import COO


@dataclass(frozen=True)
class CoordToColumns:
    y: str
    age_part: str
    age_cnt: Optional[str] = None
    age_grp_cnt: Optional[str] = None
    id_var: str = 'id'
    grp_vars: Optional[list[str] | str] = None

    def age_vars(self):
        if self.age_cnt:
            return ['age_part', 'age_cnt']
        elif self.age_grp_cnt:
            return ['age_part', 'age_grp_cnt']
        else:
            raise ValueError('One of age_cnt or age_grp_cnt must be provided')

    def __post_init__(self):
        if isinstance(self.grp_vars, str):
            object.__setattr__(self, 'grp_vars', [self.grp_vars])
        elif self.grp_vars is None:
            object.__setattr__(self, 'grp_vars', [])


class PandasLoader:
    def __init__(self, df: pd.DataFrame, col_map: CoordToColumns):
        self.df = df.copy()
        self.col_map = col_map

    def load(self, sparse=False) -> xr.Dataset:
        """Convert selected columns from DataFrame to Xarray Dataset."""
        cols_needed = [self.col_map.y,
                       self.col_map.id_var] + self.col_map.age_vars()

        if self.col_map.grp_vars:
            cols_needed.extend(self.col_map.grp_vars)

        data = self.df[cols_needed].dropna().copy()

        # Convert object columns to categorical
        for col in data.select_dtypes(include='object').columns:
            data[col] = pd.Categorical(data[col])

        # Store category info for future use
        self.categories = {
            col: data[col].cat.categories
            for col in data.select_dtypes(include='category').columns
        }

        self.raw_df = data
        cols_needed.remove(self.col_map.y)
        dim_vals = {var: data[var].values for var in cols_needed}
        if self.col_map.age_cnt:
            min_age = np.min([data[self.col_map.age_part].min(), data[self.col_map.age_cnt].min()])
            max_age = np.max([data[self.col_map.age_part].max(), data[self.col_map.age_cnt].max()])

            dim_vals[self.col_map.age_part] = np.arange(min_age, max_age + 1, dtype=int)
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
        df_full[self.col_map.y] = df_full[self.col_map.y].fillna(0).astype(int)
        ds_out = df_full.set_index(cols_needed).to_xarray()

        if sparse:
            dense_y = ds_out[self.col_map.y].values
            sparse_y = COO.from_numpy(dense_y)
            ds_out[self.col_map.y] = xr.DataArray(
                sparse_y,
                dims=ds_out[self.col_map.y].dims,
                coords=ds_out[self.col_map.y].coords,
                name=self.col_map.y
            )

        for var in cols_needed:
            ds_out.coords[var] = ds_out[var]
        self.ds = ds_out

    def stratify(self):
        pass