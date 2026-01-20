("""Data-agnostic evaluation and reporting utilities.

Provides lightweight helpers to:
- compute coverage (non-null fraction) for any column
- summarize counts of values per group (supports list-like values)
- save small metric reports to disk as JSON or CSV

Backwards-compatible wrappers remain for sentiment/theme specific use.
""")
from typing import Dict, Any, Iterable, Optional
import pandas as pd
import json
import os


def evaluate_coverage(df: pd.DataFrame, column: str) -> float:
	"""Return fraction (0-1) of rows with non-null values in `column`.

	- Returns 0.0 if the column does not exist or the DataFrame is empty.
	"""
	if column not in df.columns:
		return 0.0
	total = len(df)
	if total == 0:
		return 0.0
	non_null = df[column].notna().sum()
	return non_null / total


def evaluate_sentiment_coverage(df: pd.DataFrame, score_col: str = 'sentiment_score') -> float:
	"""Backward-compatible wrapper that delegates to `evaluate_coverage`."""
	return evaluate_coverage(df, score_col)


def summarize_counts(
	df: pd.DataFrame,
	group_col: str,
	value_col: str,
	list_types: Iterable[type] = (list, tuple, set),
	dropna: bool = True,
) -> pd.DataFrame:
	"""Summarize counts of `value_col` per `group_col` for any dataset.

	- Handles scalar values and list-like values by exploding them.
	- Returns a DataFrame with columns: `group_col`, `value_col`, `count`.
	- If columns are missing or no rows, returns an empty DataFrame with expected columns.
	"""
	if group_col not in df.columns or value_col not in df.columns:
		return pd.DataFrame(columns=[group_col, value_col, 'count'])

	# Select relevant columns and optionally drop NA in value_col
	data = df[[group_col, value_col]].copy()
	if dropna:
		data = data[data[value_col].notna()]

	rows = []
	for grp_val, grp in data.groupby(group_col):
		for v in grp[value_col]:
			if isinstance(v, tuple(list_types)):
				for item in v:
					rows.append((grp_val, item))
			else:
				rows.append((grp_val, v))

	if not rows:
		return pd.DataFrame(columns=[group_col, value_col, 'count'])

	summary = pd.DataFrame(rows, columns=[group_col, value_col])
	summary = summary.groupby([group_col, value_col]).size().reset_index(name='count')
	return summary


def summarize_theme_counts(df: pd.DataFrame, bank_col: str = 'bank_name', theme_col: str = 'theme') -> pd.DataFrame:
	"""Backward-compatible wrapper that delegates to `summarize_counts`."""
	return summarize_counts(df, group_col=bank_col, value_col=theme_col)


def save_metrics(metrics: Dict[str, Any], path: str, *, ensure_dir: bool = True, indent: Optional[int] = 2):
	"""Save metrics dict to JSON file at `path`.

	- `ensure_dir`: create parent directory if it doesn't exist.
	- `indent`: control JSON indentation (use None for compact output).
	"""
	if ensure_dir:
		parent = os.path.dirname(path)
		if parent:
			os.makedirs(parent, exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(metrics, f, indent=indent)


def save_csv(df: pd.DataFrame, path: str, *, index: bool = False, ensure_dir: bool = True, **to_csv_kwargs):
	"""Save any DataFrame to CSV.

	- `index`: include DataFrame index.
	- `ensure_dir`: create parent directory if it doesn't exist.
	- `to_csv_kwargs`: forwarded to `DataFrame.to_csv`.
	"""
	if ensure_dir:
		parent = os.path.dirname(path)
		if parent:
			os.makedirs(parent, exist_ok=True)
	df.to_csv(path, index=index, **to_csv_kwargs)


def save_sentiment_theme_csv(df: pd.DataFrame, path: str):
	"""Backward-compatible wrapper that delegates to `save_csv`."""
	save_csv(df, path, index=False, ensure_dir=True)

