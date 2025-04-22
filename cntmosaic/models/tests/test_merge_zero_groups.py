import pytest
import pandas as pd
from .._SocialMix import merge_zero_groups

# Language: python

def test_no_zero_counts():
	intervals = [
		pd.Interval(0, 10, closed='left'),
		pd.Interval(10, 20, closed='left')
	]
	counts = [5, 7]
	result = merge_zero_groups(intervals, counts)
	# Expecting the original intervals as no merge should occur.
	assert result == intervals

def test_example_from_docstring():
	intervals = [
		pd.Interval(0, 5, closed='left'),
		pd.Interval(5, 10, closed='left'),
		pd.Interval(10, 15, closed='left'),
		pd.Interval(15, 20, closed='left')
	]
	counts = [10, 0, 15, 0]
	expected = [
		pd.Interval(0, 10, closed='left'),
		pd.Interval(10, 20, closed='left')
	]
	result = merge_zero_groups(intervals, counts)
	assert result == expected

def test_all_zero_counts():
	intervals = [
		pd.Interval(0, 5, closed='left'),
		pd.Interval(5, 10, closed='left'),
		pd.Interval(10, 15, closed='left')
	]
	counts = [0, 0, 0]
	with pytest.raises(AssertionError, match="All counts are zero. There are no participants in any group."):
		merge_zero_groups(intervals, counts)
	

def test_first_zero_merges_with_next_nonzero():
	intervals = [
		pd.Interval(0, 5, closed='left'),
		pd.Interval(5, 10, closed='left')
	]
	counts = [0, 12]
	expected = [
		pd.Interval(0, 10, closed='left')
	]
	result = merge_zero_groups(intervals, counts)
	assert result == expected

def test_middle_zero_merges_with_previous():
	intervals = [
		pd.Interval(0, 10, closed='left'),
		pd.Interval(10, 20, closed='left'),
		pd.Interval(20, 30, closed='left')
	]
	counts = [10, 0, 5]
	expected = [
		pd.Interval(0, 20, closed='left'),
		pd.Interval(20, 30, closed='left')
	]
	result = merge_zero_groups(intervals, counts)
	assert result == expected
 
def test_edge_case_all_zero_except_one():
  intervals = [
    pd.Interval(0, 5, closed='left'),
    pd.Interval(5, 10, closed='left'),
    pd.Interval(10, 15, closed='left'),
    pd.Interval(15, 20, closed='left'),
    pd.Interval(20, 25, closed='left')
  ]
  counts = [0, 0, 10, 0, 0]
  expected = [
    pd.Interval(0, 25, closed='left')
  ]
  result = merge_zero_groups(intervals, counts)
  assert result == expected