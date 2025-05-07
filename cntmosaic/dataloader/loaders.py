import pandas as pd
import warnings
from cntmosaic.preprocess._utils import make_full_grid
from cntmosaic.preprocess._preprocess import convert_to_categorical, document_categories

class GeneralLoader:
    def __init__(self, data):
        self.data = data

    def generate_input(self, column_map):
        '''
        Argument:
        column_map, dict
        required key: 
            y, contact counts
            id_var, unique identifier
            age_part, age for the participants
            age_cnt/age_grp_cnt, age for the contacts
        '''
        assert('y' in column_map or 'y' in self.data.columns)
        assert('id_var' in column_map or 'id_var' in self.data.columns)
        assert('age_part' in column_map or 'age_part' in self.data.columns)
        data = self.data.rename(
            columns={v: k for k, v in column_map.items() if k != 'grp_vars'})

        if 'age_cnt' in data.columns:
            age_vars = ['age_part', 'age_cnt']
        elif 'age_grp_cnt' in data.columns:
            age_vars = ['age_part', 'age_grp_cnt']
        else:
            raise ValueError(
                'data must contain a column "age_cnt" or "age_grp_cnt"')
        
        if 'grp_vars' not in column_map:
            selected_vars = ['id_var'] + age_vars + ['y']
            grp_vars_part = ['age_part']
            grp_vars_cnt = age_vars
            grp_vars = None
        else:
            grp_vars = column_map['grp_vars'] 
            if isinstance(grp_vars, str):
                grp_vars = [grp_vars]
            else:
                if not isinstance(grp_vars, list):
                    raise NotImplementedError
            selected_vars = ['id_var'] + age_vars + grp_vars + ['y']
            grp_vars_part = ['age_part'] + grp_vars
            grp_vars_cnt = age_vars + grp_vars

        data = data[selected_vars]

        # Drop rows with missing values
        n_all = data.shape[0]
        data = data.dropna(axis=0)
        n_dropped = n_all - data.shape[0]
        if n_dropped > 0:
            warnings.warn(f'Dropped {n_dropped} rows with missing values', RuntimeWarning)

        # Convert non-numeric columns to categorical
        non_numeric_cols = data[grp_vars_cnt].select_dtypes(include='object').columns
        data = convert_to_categorical(data, non_numeric_cols)

        # Document the categories, intervals
        dict_cats = document_categories(data)

        # ===== Calculate N and y =====
        df_N = data.groupby(grp_vars_part, observed=False).agg(
            N=('id_var', 'nunique')).reset_index()
        df_y = data.groupby(grp_vars_cnt, observed=False).agg(
            y=('y', 'sum')).reset_index()

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

        return df.reset_index(drop=True)

class DataFrameLoader(GeneralLoader):
    def __init__(self, participants, contacts, population=None):
        '''
        store info about participants, contacts in separate df
        population and other factor to follow
        '''

        if isinstance(participants, pd.DataFrame):
            self.part = participants
        elif isinstance(participants, list) and all(isinstance(df, pd.DataFrame) for df in participants):
            print('merging required, implement later')
        else:
            raise TypeError("Expected a DataFrame or a list of DataFrames")
            
        if isinstance(contacts, pd.DataFrame):
            self.cont = contacts
        elif isinstance(contacts, list) and all(isinstance(df, pd.DataFrame) for df in contacts):
            print('merging required, implement later')
        else:
            raise TypeError("Expected a DataFrame or a list of DataFrames")
        
        common_cols = list(set(self.part.columns) & set(self.cont.columns))
        self.data = pd.merge(self.part, self.cont, on=common_cols)