import pytest
import pandas as pd

from .._preprocess import make_train_data

def test_basic_functionality():
	# Create a simple DataFrame
	data = pd.DataFrame({
		'id': [1, 2, 3],
		'part_age': [0, 1, 2],
		'cnt_age': [0, 1, 2],
		'part_sex': ['M', 'F', 'M'],
		'y': [1, 1, 1]
	})
	
	df_train = make_train_data(data, 'id', 'part_sex')
	assert df_train.shape == (9, 5), "Incorrect dimensions"
		
def test_no_grouping_vars():
	# Create a simple DataFrame
	data = pd.DataFrame({
		'id': [1, 2, 3],
		'part_age': [0, 1, 2],
		'cnt_age': [0, 1, 2],
		'y': [1, 1, 1]
	})
	
	df_train = make_train_data(data, 'id')
	
	assert df_train.shape == (9, 4), "Incorrect dimensions"
		
def test_multiple_grouping_vars():
	# Create a simple DataFrame
	data = pd.DataFrame({
		'id': [1, 2, 3],
		'part_age': [0, 1, 2],
		'cnt_age': [0, 1, 2],
		'part_sex': ['M', 'F', 'M'],
		'ses': ['low', 'low', 'high'],
		'y': [1, 1, 1]
	})
	
	df_train = make_train_data(data, 'id', ['part_sex', 'ses'])
	
	assert df_train.shape == (9, 6), "Incorrect dimensions"

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_na_handling():
	# Create a simple DataFrame
	data = pd.DataFrame({
		'id': [1, 2, 3],
		'part_age': [0, 1, 2],
		'cnt_age': [0, 1, None],
		'y': [1, 1, 1]
	})
	
	with pytest.warns(RuntimeWarning, match=r'Dropped \d+ rows with missing values'):
		df_train = make_train_data(data, 'id')

@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_no_y():
	# Create a simple DataFrame
	data = pd.DataFrame({
		'id': [1, 2, 3],
		'part_age': [0, 1, 2],
		'cnt_age': [0, 1, None]
	})
	
	with pytest.warns(RuntimeWarning, match='No column "y" found. Assuming each row represents a single contact.'):
		df_train = make_train_data(data, 'id')