"""Plotting utility with auto-save and inline display."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from src.config.settings import settings


class Plotter:
    """Lightweight plotting helpers for notebooks and scripts.

    - Always displays plots inline.
    - Automatically saves PNG files to the configured figures directory using a slugified title.
    - Caller only needs to pass the title; no manual paths.
    """

    def __init__(self, figures_dir: Optional[Path] = None):
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        self.figures_dir = Path(figures_dir) if figures_dir else settings.figures_dir
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify(title: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip()).strip("_").lower()
        return slug or "figure"

    def _finalize(self, title: str | None, xlabel: str | None, ylabel: str | None):
        if title:
            plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.tight_layout()

        if title:
            filename = f"{self._slugify(title)}.png"
            out_path = self.figures_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300)

        plt.show()
        plt.close()

    def plot_histogram(self, df, column, title=None, xlabel=None, ylabel="Count", bins=20, log_scale=False):
        """Plot and save a histogram with KDE; coerces to numeric and drops NaNs."""
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            print(f"No numeric data available to plot for column '{column}'.")
            return

        plt.figure()
        sns.histplot(series, bins=bins, kde=True)
        if log_scale:
            plt.xscale("symlog")
        self._finalize(title or f"Distribution of {column}", xlabel or column, ylabel)

    def plot_bar(self, df, x, y, title=None, xlabel=None, ylabel=None):
        """Plot and save a bar chart."""
        plt.figure()
        sns.barplot(data=df, x=x, y=y)
        plt.xticks(rotation=45)
        self._finalize(title or f"{y} by {x}", xlabel or x, ylabel or y)

    def plot_pie(self, data, labels, title=None, autopct='%1.1f%%', startangle=140, legend=False):
        """Plot and save a pie chart."""
        plt.figure(figsize=(10, 10))
        if legend:
            wedges, texts, autotexts = plt.pie(data, labels=None, autopct=autopct, startangle=startangle)
            plt.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        else:
            plt.pie(data, labels=labels, autopct=autopct, startangle=startangle)
        self._finalize(title or "Pie Chart", None, None)

    def plot_time_series(self, df, date_col, value_col, title=None, xlabel=None, ylabel=None):
        """Plot and save a time series line chart."""
        plt.figure()
        sns.lineplot(data=df, x=date_col, y=value_col, marker="o")
        plt.xticks(rotation=45)
        self._finalize(title or f"{value_col} over Time", xlabel or date_col, ylabel or value_col)

    def plot_box(self, df, y, x=None, title=None, xlabel=None, ylabel=None):
        """Plot and save a boxplot (optionally grouped by x)."""
        plt.figure()
        sns.boxplot(data=df, x=x, y=y)
        plt.xticks(rotation=45)
        self._finalize(title or f"Distribution of {y}", xlabel or (x if x else ""), ylabel or y)

    def plot_heatmap(self, corr_mat: pd.DataFrame, title: str | None = None, cmap: str = "coolwarm",
                     center: Optional[float] = 0, square: bool = True, annot: bool = False,
                     fmt: str = ".2f", figsize: tuple = (12, 8)):
        """Plot and save a correlation-style heatmap from a square DataFrame.

        Parameters
        - corr_mat: square DataFrame of pairwise values (e.g., correlations)
        - title: optional title used for display and filename
        - cmap, center, square, annot, fmt: forwarded to seaborn.heatmap
        - figsize: matplotlib figure size tuple
        """
        if corr_mat is None or corr_mat.empty:
            print("No matrix provided for heatmap.")
            return

        plt.figure(figsize=figsize)
        sns.heatmap(corr_mat, cmap=cmap, center=center, square=square, annot=annot, fmt=fmt)
        # rotate x tick labels if long, align right
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        self._finalize(title or "Correlation Heatmap", None, None)

    def plot_density_by_class(
        self,
        df: pd.DataFrame,
        x: str,
        class_col: str,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        bw_adjust: float = 1.0,
        fill: bool = True,
        alpha: float = 0.4,
    ) -> None:
        """Density plot (KDE) of feature `x` split by binary `class_col`.

        Parameters
        - df: DataFrame with data
        - x: feature/column to plot
        - class_col: binary class column (0/1)
        - title/xlabel/ylabel: optional labels
        - bw_adjust: KDE bandwidth adjustment
        - fill/alpha: area fill options
        """
        if x not in df.columns or class_col not in df.columns:
            print(f"Columns not found for density plot: x='{x}', class_col='{class_col}'")
            return

        series = pd.to_numeric(df[x], errors="coerce")
        local_df = df.assign(**{x: series}).dropna(subset=[x, class_col])
        if local_df.empty:
            print(f"No valid data to plot density for '{x}' by '{class_col}'.")
            return

        plt.figure()
        sns.kdeplot(
            data=local_df,
            x=x,
            hue=class_col,
            common_norm=False,
            fill=fill,
            alpha=alpha,
            bw_adjust=bw_adjust,
        )
        self._finalize(title or f"Density by {class_col}: {x}", xlabel or x, ylabel or "Density")

    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve"):
        """Plot and save the ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        self._finalize(title, "False Positive Rate", "True Positive Rate")

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot and save the confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        self._finalize(title, "Predicted Label", "True Label")

    def plot_roc_curve(self, y_true, y_proba, title="ROC Curve"):
        """Plot and save the ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        self._finalize(title, "False Positive Rate", "True Positive Rate")

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot and save the confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        self._finalize(title, "Predicted Label", "True Label")
