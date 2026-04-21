"""
src/visualization/eda_plots.py
───────────────────────────────
Reusable chart functions for exploratory data analysis.
Called by notebooks and potentially by the Streamlit dashboard.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid", palette="muted")

CHURN_PALETTE = {0: "#2ecc71", 1: "#e74c3c"}


def plot_churn_by_category(
    df: pd.DataFrame,
    col: str,
    title: Optional[str] = None,
    figsize: tuple = (9, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar chart of churn rate for each category in `col`."""
    stats = df.groupby(col)["churned"].agg(["mean", "count"]).reset_index()
    stats.columns = [col, "churn_rate", "count"]
    stats = stats.sort_values("churn_rate", ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(stats)))
    bars = ax.bar(stats[col].astype(str), stats["churn_rate"], color=colors, edgecolor="white")

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{row['churn_rate']:.1%}\n(n={int(row['count']):,})",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(title or f"Churn Rate by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Churn Rate")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_numeric_vs_churn(
    df: pd.DataFrame,
    col: str,
    plot_type: str = "box",
    title: Optional[str] = None,
    figsize: tuple = (8, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Box or violin plot of a numeric column split by churn label."""
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "violin":
        sns.violinplot(data=df, x="churned", y=col, palette=CHURN_PALETTE, ax=ax)
    elif plot_type == "hist":
        for label, color in CHURN_PALETTE.items():
            subset = df[df["churned"] == label][col]
            subset = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
            ax.hist(subset, bins=50, alpha=0.6, color=color, density=True,
                    label="Churned" if label else "Retained")
        ax.legend()
    else:
        sns.boxplot(data=df, x="churned", y=col, palette=CHURN_PALETTE, ax=ax)

    ax.set_title(title or f"{col} vs Churn")
    ax.set_xlabel("Churned (0=No, 1=Yes)")
    ax.set_ylabel(col)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[list] = None,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (12, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Triangular correlation heatmap."""
    if cols:
        df = df[cols]
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                ax=ax, annot_kws={"size": 8}, linewidths=0.3, cbar=True)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_missing_values(
    df: pd.DataFrame,
    figsize: tuple = (10, 5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Horizontal bar chart of missing value counts per column."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    if len(null_counts) == 0:
        print("No missing values found.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    null_counts.plot.barh(ax=ax, color="#e74c3c", alpha=0.8)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Count")
    for i, v in enumerate(null_counts.values):
        ax.text(v + 1, i, f"{v:,} ({v/len(df):.1%})", va="center", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str = "churned",
    figsize: tuple = (6, 4),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Pie + bar chart of target class distribution."""
    counts = df[target_col].value_counts()
    labels = ["Retained", "Churned"]
    colors = [CHURN_PALETTE[0], CHURN_PALETTE[1]]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[0].set_title("Class Distribution")

    axes[1].bar(labels, counts.values, color=colors, edgecolor="white")
    axes[1].set_title("Class Counts")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list,
    n_cols: int = 3,
    figsize_per_row: tuple = (14, 3.5),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grid of histograms for a list of numeric features, coloured by churn."""
    n_rows = -(-len(feature_cols) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows))
    axes = axes.flatten() if n_rows > 1 else axes

    for ax, col in zip(axes, feature_cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        for label, color in CHURN_PALETTE.items():
            subset = df[df["churned"] == label][col].dropna()
            subset = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
            ax.hist(subset, bins=40, alpha=0.55, color=color, density=True,
                    label="Churned" if label else "Retained")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide empty subplots
    for ax in axes[len(feature_cols):]:
        ax.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in [CHURN_PALETTE[0], CHURN_PALETTE[1]]]
    fig.legend(handles, ["Retained", "Churned"], loc="upper right", fontsize=9)
    plt.suptitle("Feature Distributions by Churn Label", y=1.01, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig
