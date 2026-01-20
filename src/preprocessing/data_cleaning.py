"""
Data Preprocessing Module

Functions for cleaning and preprocessing financial data.
"""

import pandas as pd
import numpy as np
from typing import Union, List

def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Convert a column to datetime objects, handling mixed formats.
    
    Args:
        df: DataFrame
        column: Name of the column to convert
        
    Returns:
        DataFrame with converted column
    """
    df = df.copy()
    # Use mixed format inference and coerce errors to NaT
    df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)
    
    # If we have NaT, we might want to drop them or handle them, 
    # but for now we just return the dataframe with NaT
    
    # Ensure we just keep the date part for alignment if needed, 
    # but usually we want to keep the full datetime until alignment.
    # However, the notebook expects .dt.date access later or alignment by date.
    # The alignment module handles .dt.date conversion.
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', 
                          columns: List[str] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame
        strategy: 'drop' to remove rows, 'fill' to fill with method
        columns: Specific columns to check (default None = all)
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    if strategy == 'drop':
        if columns:
            df = df.dropna(subset=columns)
        else:
            df = df.dropna()
    elif strategy == 'ffill':
        df = df.fillna(method='ffill')
    elif strategy == 'bfill':
        df = df.fillna(method='bfill')
    elif strategy == 'fill_mean':
        if columns:
            df[columns] = df[columns].fillna(df[columns].mean())
        else:
            df = df.fillna(df.mean())
    
    return df

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows.
    
    Args:
        df: DataFrame
        subset: Columns to consider for identifying duplicates
        
    Returns:
        DataFrame without duplicates
    """
    return df.drop_duplicates(subset=subset)
