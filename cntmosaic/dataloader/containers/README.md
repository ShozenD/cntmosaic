# Data Containers

This directory contains the source code for data containers used in the data loading process. These containers help to validate, preprocess, and structure the data in order to simplify downstream preprocessing within `DataLoader`.

## `ParticiapantData` container
The `ParticipantData` container is responsible for handling participant-level data. In the most basic form, the class is instantiated as follows:
```python
ParticipantData(
    df_part=df_part, # pandas DataFrame containing participant data
    id_col="id",     # column name for participant IDs
    age_col="age",   # column name for participant ages (1-year resolution)
)
```
In this case, the `df_part` DataFrame should have the following structure:
| id  | age | ... |
|-----|-----|-----|
| 1   | 34  | ... |
| 2   | 45  | ... |
| ... | ... | ... |

Where `id` is the unique identifier for each participant and `age` is their age in 1 year age resolution. If there is a stratification varialbe (e.g. sex) then the container can be instantiated as follows:

```python
ParticipantData(
    df_part=df_part,       # pandas DataFrame containing participant data
    id_col="id",           # column name for participant IDs
    age_col="age",         # column name for participant ages (1-year resolution)
    strat_var_cols="sex"     # column name for stratification variable(s
)
```

and the `df_part` DataFrame should have the following structure
| id  | age | sex | ... |
|-----|-----|-----|-----|
| 1   | 34  | M   | ... |
| 2   | 45  | F   | ... |
| ... | ... | ... | ... |

