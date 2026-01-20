"""Reusable EDA utilities for missingness, cardinality, correlations and helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Percent missing, dtype, and non-null counts."""
    missing_pct = df.isna().mean().mul(100).round(2)
    return (
        pd.DataFrame({"missing_pct": missing_pct, "dtype": df.dtypes.astype(str)})
        .assign(non_null=lambda d: df.shape[0] - (d["missing_pct"] * df.shape[0] / 100))
        .sort_values("missing_pct", ascending=False)
    )


def top_frequencies(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """Return top-n value counts with share."""
    counts = df[column].value_counts(dropna=False).head(n)
    return (
        counts.to_frame(name="count")
        .assign(share=lambda d: d["count"] / len(df))
        .reset_index()
        .rename(columns={"index": column})
    )


def flag_outliers_iqr(series: pd.Series, iqr_multiplier: float = 1.5) -> pd.Series:
    """Boolean mask for IQR-based outliers."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    return (series < lower) | (series > upper)


def constant_columns(df: pd.DataFrame) -> List[str]:
    """Columns with a single unique value (including NaN)."""
    nunique = df.nunique(dropna=False)
    return nunique[nunique <= 1].index.tolist()


def cardinality_report(df: pd.DataFrame, cat_columns: List[str], max_rows: int = 12) -> pd.DataFrame:
    """Unique counts and share for categorical columns."""
    rows = []
    total = len(df)
    for col in cat_columns:
        if col in df.columns:
            uniq = df[col].nunique(dropna=False)
            rows.append({"column": col, "unique": uniq, "unique_pct": uniq / total if total else 0})
    return pd.DataFrame(rows).sort_values("unique_pct", ascending=False).head(max_rows)


def class_balance(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    """Return class balance for the first available target column."""
    present = [c for c in candidates if c in df.columns]
    if not present:
        return pd.DataFrame()
    col = present[0]
    counts = df[col].value_counts(dropna=False)
    return (
        counts.to_frame(name="count")
        .assign(share=lambda d: d["count"] / len(df))
        .reset_index()
        .rename(columns={"index": col})
    )


def compute_target_correlations(
    df: pd.DataFrame,
    target: str,
    exclude: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute Pearson correlation of each numeric feature to binary `target`.

    Returns a DataFrame with columns ['feature', 'corr', 'abs_corr'] sorted by |corr| desc.
    """
    if target not in df.columns:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])  # empty

    exclude = set(exclude or []) | {target}
    num_cols = df.select_dtypes(include=[np.number]).columns
    cand = [c for c in num_cols if c not in exclude]
    if not cand:
        return pd.DataFrame(columns=["feature", "corr", "abs_corr"])  # empty

    corr_series = df[cand + [target]].corr(numeric_only=True)[target].drop(labels=[target])
    out = (
        corr_series.to_frame("corr")
        .assign(abs_corr=lambda d: d["corr"].abs())
        .sort_values("abs_corr", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    return out


def correlation_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Return correlation matrix for given columns (numeric-only)."""
    use_cols = [c for c in columns if c in df.columns]
    if not use_cols:
        return pd.DataFrame()
    return df[use_cols].corr(numeric_only=True)


def add_log1p_column(df: pd.DataFrame, column: str, new_name: Optional[str] = None) -> pd.DataFrame:
    """Return a copy with log1p(column) as new column.

    If new_name is None, defaults to f"{column}_log1p".
    """
    if column not in df.columns:
        return df.copy()
    new_name = new_name or f"{column}_log1p"
    series = pd.to_numeric(df[column], errors="coerce")
    return df.assign(**{new_name: np.log1p(series)})


def duplicates_count(df: pd.DataFrame) -> int:
    """Count fully duplicated rows in a DataFrame."""
    return int(df.duplicated().sum())


def dtypes_frame(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Return a DataFrame of dtypes; optionally head(max_rows)."""
    out = df.dtypes.astype(str).to_frame("dtype").reset_index().rename(columns={"index": "column"})
    return out if max_rows is None else out.head(max_rows)


def sample_df(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Safe sampling helper capped by length."""
    n_eff = min(n, len(df))
    return df.sample(n=n_eff, random_state=random_state) if len(df) > 0 else df


def describe_by_class(
    df: pd.DataFrame,
    class_col: str,
    columns: Sequence[str],
    percentiles: Sequence[float] | None = (0.5, 0.9, 0.99),
) -> pd.DataFrame:
    """Return descriptive stats for `columns` grouped by `class_col` with optional percentiles."""
    use_cols = [c for c in columns if c in df.columns]
    if class_col not in df.columns or not use_cols:
        return pd.DataFrame()
    desc = (
        df[use_cols + [class_col]]
        .groupby(class_col)[use_cols]
        .describe(percentiles=list(percentiles) if percentiles else None)
    )
    return desc


__all__ = [
    "missingness_summary",
    "top_frequencies",
    "flag_outliers_iqr",
    "constant_columns",
    "cardinality_report",
    "class_balance",
    "compute_target_correlations",
    "correlation_matrix",
    "add_log1p_column",
    "duplicates_count",
    "dtypes_frame",
    "sample_df",
    "describe_by_class",
]
