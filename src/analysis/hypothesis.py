"""Reusable hypothesis testing utilities for segment-level insurance risk analysis.

Each helper keeps sampling thresholds and aggregation logic in one place so notebooks
and scripts stay concise. Designed for heavily skewed insurance loss data.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats


def segment_kpis(data: pd.DataFrame, group_col: str, min_obs: int = 500) -> pd.DataFrame:
    """Aggregate risk/profit metrics by a segment, filtering sparse groups."""

    subset = data.dropna(subset=[group_col])
    counts = subset[group_col].value_counts()
    keepers = counts[counts >= min_obs].index
    subset = subset[subset[group_col].isin(keepers)]

    agg = subset.groupby(group_col).agg(
        policies=("PolicyID", "nunique"),
        exposure_months=("TransactionMonth", "count"),
        premium_sum=("TotalPremium", "sum"),
        claims_sum=("TotalClaims", "sum"),
        claimants=("claim_flag", "sum"),
        claim_rate=("claim_flag", "mean"),
        margin_mean=("margin", "mean"),
    )

    agg["severity"] = np.where(agg["claimants"] > 0, agg["claims_sum"] / agg["claimants"], np.nan)
    agg["loss_ratio"] = np.where(agg["premium_sum"] > 0, agg["claims_sum"] / agg["premium_sum"], np.nan)
    return agg.sort_values("loss_ratio", ascending=False)


def chi_square_claims(data: pd.DataFrame, group_col: str, min_obs: int = 500) -> Dict[str, float]:
    """Chi-squared test to check if claim frequency differs across groups."""

    subset = data.dropna(subset=[group_col, "claim_flag"])
    counts = subset[group_col].value_counts()
    keepers = counts[counts >= min_obs].index
    subset = subset[subset[group_col].isin(keepers)]

    contingency = pd.crosstab(subset[group_col], subset["claim_flag"])
    chi2, p, dof, _ = stats.chi2_contingency(contingency)
    return {"stat": chi2, "p_value": p, "dof": dof, "groups": len(contingency)}


def kruskal_by_group(data: pd.DataFrame, group_col: str, value_col: str, min_obs: int = 500) -> Dict[str, float]:
    """Non-parametric test (Kruskal–Wallis) for skewed numeric outcomes across groups."""

    subset = data.dropna(subset=[group_col, value_col])
    counts = subset[group_col].value_counts()
    keepers = counts[counts >= min_obs].index
    subset = subset[subset[group_col].isin(keepers)]

    samples: List[np.ndarray] = [
        group[value_col].values for _, group in subset.groupby(group_col) if len(group) >= min_obs
    ]
    if len(samples) < 2:
        return {"stat": np.nan, "p_value": np.nan, "groups": len(samples)}

    stat, p = stats.kruskal(*samples)
    return {"stat": stat, "p_value": p, "groups": len(samples)}


def format_test_result(name: str, result: Dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Return a tidy one-row frame summarizing the test decision."""

    decision = "Reject H0" if result["p_value"] < alpha else "Fail to reject H0"
    return pd.DataFrame(
        {
            "test": [name],
            "statistic": [result["stat"]],
            "p_value": [result["p_value"]],
            "groups_tested": [result.get("groups")],
            "alpha": [alpha],
            "decision": [decision],
        }
    )