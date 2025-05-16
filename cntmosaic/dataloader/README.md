# Data Loaders

## BaseLoader
The `BaseLoader` class is an abstract base class for data loaders. It provides a common workflow for validating, loading, and processing the data.
The class is designed to be extended by other data loader classes that implement specific loading mechanisms.

### Data validation
The `_validate` method is responsible for checking the integrity of the data. It ensures that all the columns specified in the `CoordToColumns` instance is present in the data. It also checks the data types of the columns to ensure they are compatible with the expected types. If any validation fails, a descriptive error message is raised to help the user identify the source of the problem.

1. Checks if all the necessary columns are present in the data
2. Checks if all object columns are of type `pd.CategoricalDtype`. If they are not, convert them to `pd.CategoricalDtype` and raise a warning.
3. If `age_grp_cnt` is specified, checks if it is of type `pd.IntervalDtype`. If it is not, raise a TypeError.
4. Checks if the minimum and maximum values of `age_part` and `age_cnt` (or `age_grp_cnt`) are consistent with the population data. Most models 
won't work with out proper population size esimates hence the minimum and maximum age values must always match those in the population size data.