import pandas as pd
import xarray as xr
import numpy as np
import jax.numpy as jnp
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
    population_age: Optional[str] = None
    population_size: Optional[str] = None
    '''
     A configuration class to map coordinate variable names to corresponding
    column names in a dataset.

    Attributes
    ----------
    age_part : str
        The column name representing the fine(integer) participants age.
    age_cnt : str, optional
        The column name representing the fine(integer) contacts age.
    age_grp_cnt : str, optional
        The column name representing the coarse(interval) contacts age.
    id_var : str, default='id'
        The column name used as a unique identifier.
    y : str, optional, default='y'
        The column name for the number of contact counts
    grp_vars : list of str or str, optional
        The column name(s) representing grouping variables (e.g., sex, region).
    population_age : str, optional
        The column name for age in the population dataset.
    population_size : str, optional
        The column name for population size in the population dataset.

    Raises
    ------
    ValueError
        If neither `age_cnt` nor `age_grp_cnt` is provided, or if one of
        `population_age` or `population_size` is provided without the other.
    '''

    @property
    def cnt(self):
        return self.age_cnt or self.age_grp_cnt
    
    @property
    def part(self):
        return self.age_part

    def __post_init__(self):
        if isinstance(self.grp_vars, str):
            object.__setattr__(self, 'grp_vars', [self.grp_vars])
        elif self.grp_vars is None:
            object.__setattr__(self, 'grp_vars', [])
        if (self.population_age is None) != (self.population_size is None):
            raise ValueError("Both 'population_age' and 'population_size' must be set together or both left as None.")
        if not (self.age_grp_cnt or self.age_cnt):
            raise ValueError('One of age_cnt or age_grp_cnt must be provided')
    

@dataclass
class HyperParams:
    """
    A class for managing and displaying model hyperparameters, particularly
    prior distributions and their associated parameters.

    Attributes
    ----------
    prior : dict
        A dictionary mapping parameter names to distribution objects (e.g., from `scipy.stats` or `torch.distributions`).

    Others: to be appended by model class and dataloader class
    """
    def __init__(self):
        self.prior = {}
    
    def __repr__(self):
        """
        Returns a string representation similar to __str__, but in a more
        code-like format for debugging.

        Returns
        -------
        str
            A detailed string including all attributes and prior parameters.
        """
        lines = [f"{self.__class__.__name__}("]
        for k, v in self.__dict__.items():
            if k != 'prior':
                lines.append(f"  {k} = {repr(v)},")
        lines.append("  prior = {")
        for k, v in self.prior.items():
            lines.append(f"    {repr(k)}: {repr(v)},")
            d = self.get_params(v)
            for k1, v1 in d.items():
                lines.append(f"      # {k1}: {repr(v1)}")
        lines.append("  }")
        lines.append(")")
        return "\n".join(lines)

    def __str__(self):
        """
        Returns a formatted string representation of the hyperparameters,
        including all attributes except 'prior', and then details of each
        prior distribution and its parameters.

        Returns
        -------
        str
            A multi-line string listing all non-prior attributes and
            each prior's parameters.
        """
        lines = []
        for k, v in self.__dict__.items():
            if k != 'prior':
                lines.append(f"{k}: {v}")
        lines.append("prior:")
        for k, v in self.prior.items():
            lines.append(f'{k}:{v}')
            d = self.get_params(v)
            for k1, v1 in d.items():
                lines.append(f'{k1}:{v1}')
        return '\n'.join(lines)
    
    @staticmethod     
    def get_params(distr):
        """
        Extracts a predefined set of commonly used distribution parameters from a distribution object.

        Parameters
        ----------
        distr : object
            A distribution object that has standard attributes (e.g., loc, scale, mean, variance).

        Returns
        -------
        dict
            A dictionary of selected distribution parameters and their values.
        """
        common_distribution_params = [
            # Central tendency & spread
            "loc",
            "scale",
            "mean",
            "variance",
            
            # Rate & shape
            "rate",
            "concentration",
            "concentration0",
            "concentration1",
            "scale_tril",
            "precision_matrix",
            "covariance_matrix",
            
            # Discrete/multivariate
            "total_count",
            "probs",
            "logits",
            "low",
            "high",
            "df",  # degrees of freedom (StudentT)
            
            # Meta/shape
            "batch_shape",
            "event_shape",
            "support"
        ]
        return {
                attr: getattr(distr, attr)
                for attr in dir(distr)
                if (not attr.startswith("_") and not callable(getattr(distr, attr) )) and (attr in common_distribution_params)
            }


class GeneralLoader:
    """
    Base class for transforming contact and participant data into an Xarray Dataset.

    This class handles age filtering, categorical encoding, reshaping to a complete 
    grid (panel data), and creation of a structured Xarray dataset. It also includes 
    precomputation logic for statistical modeling and analysis.

    Attributes:
        raw_df (pd.DataFrame): The raw merged dataset of participants and contacts.
        col_map (CoordToColumns): Object mapping standard coordinate names to dataframe columns.
        ds (xr.Dataset): The processed Xarray dataset.
        precomputes (HyperParams): Precomputed values for model use (e.g., age distribution, log-population).

    Methods:
        load(sparse=False): Transforms data into Xarray Dataset with optional sparse encoding.
        set_age_bounds(min_age=0, max_age=None): Sets the bounds for age dimensions.
        precompute(): Generates precomputed statistics and transformations for downstream modeling.
        stratify(): Placeholder for stratified analysis logic (to be implemented in subclasses).
    """

    def load(self, sparse=False) -> xr.Dataset:
        '''
        Converts the cleaned contact-participant DataFrame into an Xarray Dataset.

        Performs the following operations:
        - Filters rows based on age constraints.
        - Converts object columns to categorical and stores category metadata.
        - Constructs a complete panel via Cartesian product over age dimensions.
        - Aggregates and reshapes data into a structured Xarray Dataset.
        - Optionally converts dense arrays into sparse COO format.
        - Triggers precomputations needed for modeling.

        Parameters:
            sparse (bool, optional): If True, converts the main data variable to a sparse
                COO representation to save memory. Defaults to False.

        Returns:
            xr.Dataset: An Xarray Dataset with structured coordinates, ready for analysis or modeling.

        Raises:
            NotImplementedError: If age columns required for alignment are missing or unsupported.
        '''

        cols_needed = [self.col_map.y,
                       self.col_map.id_var,
                       self.col_map.cnt,
                       self.col_map.part] + self.col_map.grp_vars

        data = self.raw_df[cols_needed].dropna().copy()
        if not hasattr(self, 'min_age'):
            self.set_age_bounds()
        else:
            # filter by age
            data = data[data[self.col_map.age_part] <= self.max_age]
            data = data[data[self.col_map.age_part] >= self.max_age]
            if self.col_map.age_grp_cnt:
                data = data[data[self.col_map.age_grp_cnt].apply(lambda x: x.right <= self.max_age)]
                data = data[data[self.col_map.age_grp_cnt].apply(lambda x: x.left >= self.max_age)]
            else:
                data = data[data[self.col_map.age_cnt] <= self.max_age]
                data = data[data[self.col_map.age_cnt] >= self.max_age]

        # Convert object columns to categorical
        for col in data.select_dtypes(include='object').columns:
            data[col] = pd.Categorical(data[col])

        # Store category info for future use
        self.categories = {
            col: data[col].cat.categories
            for col in data.select_dtypes(include='category').columns
        }

        cols_needed.remove(self.col_map.y)
        dim_vals = {}

        if self.col_map.age_cnt and self.col_map.age_part:
            dim_vals[self.col_map.age_part] = np.arange(self.min_age, self.max_age + 1, dtype=int)
            dim_vals[self.col_map.age_cnt] = np.arange(self.min_age, self.max_age + 1, dtype=int)
        elif self.col_map.age_grp_cnt and self.col_map.age_part:
            dim_vals[self.col_map.age_part] = np.arange(self.min_age, self.max_age + 1, dtype=int)
            dim_vals[self.col_map.age_grp_cnt] = np.unique(data[self.col_map.age_grp_cnt])
        else:
            raise NotImplementedError('Fine age for participant is required in current version')

        panel = data.groupby(cols_needed, observed=False).sum().reset_index()
        expanded_dims = [self.col_map.cnt, self.col_map.part]
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
        self.precomputes = self.precompute()

    # consider class abstractmethod, current implementation is for fine age

    def set_age_bounds(self, min_age=0, max_age=None):
        data = self.raw_df
        if max_age:
            self.min_age = min_age
            self.max_age = max_age
        elif self.col_map.age_cnt and self.col_map.age_part:
            self.min_age = np.min([data[self.col_map.age_part].min(), data[self.col_map.age_cnt].min()])
            self.max_age = np.max([data[self.col_map.age_part].max(), data[self.col_map.age_cnt].max()])
        elif self.col_map.age_grp_cnt and self.col_map.age_part:
            self.min_age = min(data[self.col_map.age_part].min(), data[self.col_map.age_grp_cnt].min().left)
            self.max_age = max(data[self.col_map.age_part].max(), data[self.col_map.age_grp_cnt].max().right-1)
        else:
            raise NotImplementedError
    
    def precompute(self):
        """
        Compute and return model-ready summary statistics and age distributions.

        This method prepares key precomputed quantities necessary for statistical modeling
        of contact data. It calculates:
        
        - Age distribution (`age_dist`) of the population from provided demographic data or 
        assumes a uniform distribution if none is provided.
        - Total number of discrete age values (`A`) between the participant age bounds.
        - Total contact counts (`y`) across individuals and groups.
        - Number of unique individuals (`N`) per participant age and optional grouping variables.
        - Log-transformed `N` and `age_dist` for modeling purposes.
        - Participant (`aid`) and contact (`bid`) age indices for matrix alignment.

        Returns:
            HyperParams: A container with attributes:
                - `prior`: Meta-information about precomputation.
                - `A`: Number of discrete ages.
                - `age_dist`: Normalized age distribution array.
                - `aid`: Participant age indices.
                - `bid`: Contact age indices.
                - `y`: Flattened contact count vector.
                - `log_N`: Logarithm of unique individuals per age/group.
                - `log_P`: Logarithm of the population age distribution (row vector).

        Raises:
            ValueError: 
                - If both `population_age` and `population_size` are not set when `pop` is provided.
                - If the age range in the population data does not match the participant age dimension.
        """
        precomp = HyperParams()
        precomp.prior = {'precompute': 'complete'}
        precomp.A = int(self.max_age - self.min_age + 1)
        # population
        if self.pop is None:
            precomp.age_dist = 1 / precomp.A * np.ones(precomp.A)
        elif self.col_map.population_size is None:
            raise ValueError("Both 'population_age' and 'population_size' must be set")
        else:
            # this is assuming discrete integer age dimension for both population and participants
            pop = self.pop
            if pop[self.col_map.population_age].nunique() != precomp.A:
                raise ValueError(
                    f'Population age dimension({pop[self.col_map.population_age].nunique()}) does not match age dimension for participants and contacts({precomp.A})')
            pop = pop.groupby(self.col_map.population_age)[self.col_map.population_size].sum()
            precomp.age_dist = (pop/pop.sum()).values

        # y and N
        df = self.ds.sum(dim=[self.col_map.id_var]+self.col_map.grp_vars).to_dataframe().reset_index()
        
        cols_needed = [self.col_map.y,
                       self.col_map.id_var,
                       self.col_map.cnt,
                       self.col_map.part] + self.col_map.grp_vars
        N = self.raw_df[cols_needed].dropna().copy().groupby([self.col_map.part] + self.col_map.grp_vars, observed=False).agg(N=(self.col_map.id_var, 'nunique')).reset_index()

        m = pd.merge(df, N, on=self.col_map.part, how='left')
        m['N'] = m.N.fillna(1)
        # Others
        precomp.aid = m[self.col_map.part].values
        precomp.bid = m[self.col_map.cnt].values
        precomp.y = m[self.col_map.y].values
        precomp.log_N = jnp.log(m['N'].values)
        precomp.log_P = jnp.log(precomp.age_dist)[jnp.newaxis,:]

        return precomp

    
    def stratify(self):
        pass


class RawLoader(GeneralLoader):
    """
    Loader for datasets where participant and contact information are stored separately.

    Merges participant and contact data using a common identifier. Handles potential 
    column name conflicts, validates age and group variables, and prepares the dataset 
    for loading via the parent class.

    Parameters:
        part (pd.DataFrame): DataFrame containing participant-level data (e.g., age, id).
        cnt (pd.DataFrame): DataFrame containing contact-level data (e.g., age of contacts, count).
        col_map (CoordToColumns): Object mapping standard coordinate names to dataframe columns.
        pop (pd.DataFrame, optional): Optional population data used for age distribution computations.

    Raises:
        KeyError: If required columns are not present in input DataFrames or the id variable is missing.
        UserWarning: If group variable names overlap between `part` and `cnt`.
    """
    def __init__(self, part: pd.DataFrame, cnt: pd.DataFrame, col_map: CoordToColumns, pop=None):
        self.pop = pop
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
    """
    Loader for datasets where participant and contact data are already merged.

    Accepts a single DataFrame that includes all necessary variables (e.g., participant age, contact age, counts, etc.).
    This is useful when pre-merged or cleaned data is available.

    Parameters:
        df (pd.DataFrame): Combined DataFrame containing both participant and contact data.
        col_map (CoordToColumns): Object mapping standard coordinate names to dataframe columns.
        pop (pd.DataFrame, optional): Optional population data used for age distribution computations.

    Notes:
        If the `y` column (e.g., contact counts) is not present, it will be initialized to 1.
    """
    def __init__(self, df: pd.DataFrame, col_map: CoordToColumns, pop=None):
        if col_map.y not in df.columns:
            df['y'] = 1
        self.raw_df = df.copy()
        self.col_map = col_map
        self.pop = pop
