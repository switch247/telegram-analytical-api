"""
Test data preprocessing utilities
"""
import pytest
import pandas as pd
import numpy as np
from src.utils import convert_to_datetime, handle_missing_values, remove_duplicates


def test_convert_to_datetime():
    """Test datetime conversion"""
    df = pd.DataFrame({'date': ['2023-01-01', 'invalid']})
    df = convert_to_datetime(df, 'date')
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert pd.isna(df['date'][1])


def test_handle_missing_values_drop():
    """Test dropping missing values"""
    df = pd.DataFrame({'col1': [1, np.nan, 3]})
    df = handle_missing_values(df, strategy='drop')
    assert len(df) == 2


def test_handle_missing_values_fill_mean():
    """Test filling missing values with mean"""
    df = pd.DataFrame({'col1': [1.0, np.nan, 3.0]})
    df = handle_missing_values(df, strategy='fill_mean', columns=['col1'])
    assert df['col1'][1] == 2.0


def test_remove_duplicates():
    """Test removing duplicate rows"""
    df = pd.DataFrame({'col1': [1, 1, 2], 'col2': [3, 3, 4]})
    df = remove_duplicates(df)
    assert len(df) == 2
