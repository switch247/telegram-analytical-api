"""Reusable statistical modeling utilities for notebooks and scripts.

Helpers cover feature selection, optional sampling, lightweight model builders,
metric-based model selection, SHAP summarisation, and driver/action mapping.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

NUMERIC_FEATURE_CANDIDATES: Sequence[str] = (
    "candidate1",
)

CATEGORICAL_FEATURE_CANDIDATES: Sequence[str] = (
    "test",
)

DEFAULT_DRIVER_ACTIONS: Mapping[str, str] = {
    "test": "test value",
}


def select_tabular_features(
    df: pd.DataFrame,
    numeric_candidates: Iterable[str] | None = None,
    categorical_candidates: Iterable[str] | None = None,
) -> Tuple[list[str], list[str]]:
    """Filter candidate feature lists to columns present in the dataframe."""

    numeric_candidates = list(numeric_candidates) if numeric_candidates is not None else list(NUMERIC_FEATURE_CANDIDATES)
    categorical_candidates = (
        list(categorical_candidates) if categorical_candidates is not None else list(CATEGORICAL_FEATURE_CANDIDATES)
    )
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    categorical_cols = [c for c in categorical_candidates if c in df.columns]
    return numeric_cols, categorical_cols


def maybe_sample_df(df: pd.DataFrame, max_rows: int = 8000, random_state: int = 42) -> pd.DataFrame:
    """Optionally downsample large frames to keep prototyping fast."""

    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state)


def build_light_regression_models(preprocessor, random_state: int = 42) -> dict[str, Pipeline]:
    """Lightweight regression model set for quick iterations."""

    return {
        "linear": Pipeline([("prep", preprocessor), ("model", LinearRegression())]),
        "rf": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        max_depth=12,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gbr": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def build_light_classification_models(preprocessor, random_state: int = 42) -> dict[str, Pipeline]:
    """Lightweight classification model set for quick iterations."""

    return {
        "logreg": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=200,
                        class_weight="balanced",
                        n_jobs=-1,
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "rf": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=12,
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gbr": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def choose_best_model(scores_df: pd.DataFrame, prefer_cols: Sequence[str], direction: str = "max") -> tuple[str, str | None]:
    """Pick best model name given preference ordering and direction."""

    scores_df = scores_df.copy()
    scores_df = scores_df.set_index("model") if "model" in scores_df.columns else scores_df

    for col in prefer_cols:
        if col not in scores_df.columns:
            continue
        metric_series = scores_df[col].dropna()
        if metric_series.empty:
            continue
        if direction == "max":
            return metric_series.idxmax(), col
        return metric_series.idxmin(), col

    first_index = scores_df.index[0] if len(scores_df.index) else ""
    return first_index, None


def _to_array(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def compute_shap_and_top(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    top_n: int = 10,
) -> tuple["shap.Explanation", pd.DataFrame]:
    """Compute SHAP values and return top features by mean absolute SHAP."""

    import shap

    preprocessor = model.named_steps.get("prep", None)
    estimator = model.named_steps.get("model", model)

    X_train_enc = _to_array(preprocessor.transform(X_train) if preprocessor else X_train)
    X_eval_enc = _to_array(preprocessor.transform(X_eval) if preprocessor else X_eval)

    feature_names = (
        preprocessor.get_feature_names_out() if preprocessor and hasattr(preprocessor, "get_feature_names_out") else np.array([f"f{i}" for i in range(X_train_enc.shape[1])])
    )

    background = shap.utils.sample(X_train_enc, min(200, len(X_train_enc)), random_state=42)
    explainer = shap.Explainer(estimator, background, feature_names=feature_names, algorithm="auto")
    shap_values = explainer(X_eval_enc)

    values = shap_values.values
    if values.ndim == 3:  # classification outputs
        values = values[:, 1, :]

    mean_abs = np.abs(values).mean(axis=0)
    top_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )
    return shap_values, top_df


def attach_actions(top_df: pd.DataFrame, driver_actions: Mapping[str, str] | None = None, fallback: str | None = None) -> pd.DataFrame:
    """Map features to business actions using provided or default mapping."""

    driver_actions = driver_actions or DEFAULT_DRIVER_ACTIONS
    fallback_text = fallback or "Use in underwriting scorecard; monitor drift and recalibrate quarterly."
    return top_df.assign(action=top_df["feature"].map(driver_actions).fillna(fallback_text))


def plot_metrics_bar(scores_df: pd.DataFrame, plotter, title_prefix: str) -> None:
    """Plot metric bars if the dataframe has numeric columns."""

    if scores_df is None or scores_df.empty:
        print(f"[Plot] No scores available for {title_prefix}")
        return
    numeric_cols = scores_df.select_dtypes(include=["float", "int"]).columns.tolist()
    if "model" not in scores_df.columns or not numeric_cols:
        print(f"[Plot] Missing model column or numeric metrics for {title_prefix}")
        return
    long_df = scores_df[["model"] + numeric_cols].melt(id_vars="model", value_name="value", var_name="metric").dropna(subset=["value"])
    if long_df.empty:
        print(f"[Plot] All metrics are NaN for {title_prefix}")
        return
    plotter.plot_bar(long_df, x="model", y="value", title=f"{title_prefix} metrics", xlabel="model", ylabel="score")
