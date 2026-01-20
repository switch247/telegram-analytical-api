"""
Correlation Analysis Module

Statistical correlation helpers that operate on any two numeric series.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple


def calculate_pearson_correlation(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """Calculate Pearson correlation coefficient with p-value for any two series."""

    valid_idx = ~(x.isna() | y.isna())
    x_clean = x[valid_idx]
    y_clean = y[valid_idx]
    
    if len(x_clean) < 3:
        return {'correlation': 0.0, 'p_value': 1.0, 'significant': False, 'n_samples': 0}
    
    corr, p_value = pearsonr(x_clean, y_clean)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < alpha,
        'n_samples': len(x_clean)
    }


def calculate_spearman_correlation(x: pd.Series, y: pd.Series, alpha: float = 0.05) -> Dict[str, float]:
    """Calculate Spearman rank correlation (non-parametric) for any two series."""

    valid_idx = ~(x.isna() | y.isna())
    x_clean = x[valid_idx]
    y_clean = y[valid_idx]
    
    if len(x_clean) < 3:
        return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}
    
    corr, p_value = spearmanr(x_clean, y_clean)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < alpha
    }


def calculate_lagged_correlation(x: pd.Series, y: pd.Series, max_lag: int = 5, alpha: float = 0.05) -> pd.DataFrame:
    """Test correlation with time lags between two series (x leading y)."""
    results = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            lagged_y = y
        else:
            lagged_y = y.shift(-lag)
        
        corr_result = calculate_pearson_correlation(x, lagged_y, alpha=alpha)
        corr_result['lag'] = lag
        results.append(corr_result)
    
    df = pd.DataFrame(results)
    return df[['lag', 'correlation', 'p_value', 'significant', 'n_samples']]


def test_correlation_significance(correlation: float, n: int, alpha: float = 0.05) -> bool:
    """Test if correlation is statistically significant for any two-series test."""
    if n < 3:
        return False
    
    # Calculate t-statistic
    t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    
    # Critical value for two-tailed test (approximate)
    from scipy.stats import t
    critical_value = t.ppf(1 - alpha/2, n - 2)
    
    return abs(t_stat) > critical_value


def calculate_rolling_correlation(x: pd.Series, y: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling correlation over time for any two aligned series."""
    rolling_corr = x.rolling(window).corr(y)
    return rolling_corr
