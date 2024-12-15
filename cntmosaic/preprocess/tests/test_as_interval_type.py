import pandas as pd
import pytest
from .._utils import as_interval_type  # Replace 'your_module' with the actual name of your Python module

def test_interval_closed_both():
    assert as_interval_type('[1,2]') == pd.Interval(1, 2, closed='both')

def test_interval_closed_left():
    assert as_interval_type('[1,2)') == pd.Interval(1, 2, closed='left')

def test_interval_closed_right():
    assert as_interval_type('(1,2]') == pd.Interval(1, 2, closed='right')

def test_interval_neither_closed():
    assert as_interval_type('(1,2)') == pd.Interval(1, 2, closed='neither')

def test_interval_with_spaces():
    assert as_interval_type(' [ 1 , 2 ] ') == pd.Interval(1, 2, closed='both')

def test_invalid_interval_missing_comma():
    assert as_interval_type('[1 2]') is None

def test_invalid_interval_letters():
    assert as_interval_type('[a,b]') is None

def test_invalid_format_extra_characters():
    assert as_interval_type('[1,2] extra') is None

def test_invalid_empty_string():
    assert as_interval_type('') is None

def test_invalid_none_input():
    assert as_interval_type(None) is None

def test_invalid_numeric_input():
    assert as_interval_type(123) is None
