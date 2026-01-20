"""
Stock Metrics Module

Calculate daily returns, volatility, and create target variables for ML models.
"""

import pandas as pd
import numpy as np


def calculate_daily_returns(df: pd.DataFrame, price_col: str = 'Close') -> pd.Series:
    """
    Calculate percentage change in daily closing prices.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for closing price
        
    Returns:
        Series of daily returns (percentage change)
    """
    prices = pd.to_numeric(df[price_col], errors='coerce')
    returns = prices.pct_change()
    return returns


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        returns: Series of daily returns
        window: Rolling window size (default 20 days)
        
    Returns:
        Series of rolling volatility
    """
    volatility = returns.rolling(window=window).std()
    return volatility


def create_target_variable(returns: pd.Series, threshold: float = 0.0) -> pd.Series:
    """
    Create binary classification target: price up (1) or down (0).
    
    Args:
        returns: Series of daily returns
        threshold: Minimum return to classify as 'up' (default 0.0)
        
    Returns:
        Series of binary labels (1=up, 0=down)
    """
    target = (returns > threshold).astype(int)
    return target


def identify_significant_moves(returns: pd.Series, threshold: float = 0.02) -> pd.Series:
    """
    Flag days with significant price movements (>threshold).
    
    Args:
        returns: Series of daily returns
        threshold: Absolute return threshold (default 2%)
        
    Returns:
        Series of boolean flags
    """
    significant = returns.abs() > threshold
    return significant


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns over time.
    
    Args:
        returns: Series of daily returns
        
    Returns:
        Series of cumulative returns
    """
    cumulative = (1 + returns).cumprod() - 1
    return cumulative


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Sharpe ratio
    """
    # Annualize returns and volatility (assuming 252 trading days)
    excess_returns = returns - (risk_free_rate / 252)
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
    return sharpe
