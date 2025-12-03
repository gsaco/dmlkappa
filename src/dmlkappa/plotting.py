"""Simple plotting helpers for dmlkappa."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_kappa_vs_coverage(summary_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    # color by overlap if present
    if "overlap" in summary_df.columns:
        groups = summary_df.groupby("overlap")
        for name, grp in groups:
            ax.plot(grp["kappa_mean"], grp["coverage"], marker="o", linestyle="-", label=str(name))
        ax.legend()
    else:
        ax.plot(summary_df["kappa_mean"], summary_df["coverage"], marker="o")
    ax.set_xlabel("Mean κ_DML")
    ax.set_ylabel("Coverage")
    ax.set_title("κ_DML vs Coverage")
    return ax


def plot_kappa_hist(results_df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(results_df["kappa"].dropna(), bins=30, alpha=0.7)
    ax.set_xlabel("κ_DML")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of κ_DML")
    return ax
