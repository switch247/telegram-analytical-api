"""Reusable tabular modeling helpers for Task 4 (severity + claim probability).

These utilities keep preprocessing, model selection, and evaluation in one place
so notebooks stay lean and repeatable.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Optional dependency
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


def build_preprocessor(numeric_cols: Iterable[str], categorical_cols: Iterable[str]) -> ColumnTransformer:
    """Create a ColumnTransformer with imputation, scaling, and one-hot encoding."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_cols)),
            ("cat", categorical_pipeline, list(categorical_cols)),
        ]
    )
    return preprocessor


def split_features_target(
    df: pd.DataFrame,
    target: str,
    drop_cols: Iterable[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test sets with the target column removed from features.

    Parameters
    - target: target column name
    - drop_cols: columns to drop from features (in addition to target)
    - stratify: when True, uses y for stratification (useful for imbalanced classes)
    """

    drop_cols = set(drop_cols or [])
    features = df.drop(columns=list(drop_cols | {target}))
    y = df[target]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train, X_test, y_train, y_test


def build_regression_models(preprocessor: ColumnTransformer, random_state: int = 42) -> Dict[str, Pipeline]:
    """Return a dictionary of regression pipelines that share the same preprocessor."""

    models: Dict[str, Pipeline] = {
        "linear_regression": Pipeline(steps=[("prep", preprocessor), ("model", LinearRegression())]),
        "random_forest_regressor": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=None,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting_regressor": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingRegressor(random_state=random_state),
                ),
            ]
        ),
    }

    if _HAS_XGB:
        models["xgb_regressor"] = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        random_state=random_state,
                        n_estimators=200,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def build_classification_models(preprocessor: ColumnTransformer, random_state: int = 42) -> Dict[str, Pipeline]:
    """Return a dictionary of classification pipelines that share the same preprocessor."""

    models: Dict[str, Pipeline] = {
        "log_reg": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    LogisticRegression(max_iter=1000, n_jobs=-1),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(random_state=random_state),
                ),
            ]
        ),
        "random_forest_clf": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=None,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting_clf": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(random_state=random_state),
                ),
            ]
        ),
    }

    if _HAS_XGB:
        models["xgb_classifier"] = Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        random_state=random_state,
                        n_estimators=200,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def evaluate_regression(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return MAE, RMSE, and R² for each regression model."""

    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        # Some sklearn versions lack squared=False; compute RMSE manually
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)
        rows.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
    return pd.DataFrame(rows).sort_values("rmse")


def evaluate_classification(models: Dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return accuracy, precision, recall, F1, and ROC-AUC for each classifier."""

    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            proba = model.decision_function(X_test)
        else:
            proba = None

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba) if proba is not None else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("f1", ascending=False)