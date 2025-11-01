from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss)
from scipy.stats import ttest_rel, wilcoxon, shapiro

def metrics_from_preds(y_true, y_proba, y_pred):
    out = dict(
        accuracy = accuracy_score(y_true, y_pred),
        precision= precision_score(y_true, y_pred, zero_division=0),
        recall   = recall_score(y_true, y_pred, zero_division=0),
        f1       = f1_score(y_true, y_pred, zero_division=0),
        roc_auc  = roc_auc_score(y_true, y_proba) if len(np.unique(y_true))==2 else np.nan,
        ap       = average_precision_score(y_true, y_proba) if len(np.unique(y_true))==2 else np.nan,
        brier    = brier_score_loss(y_true, y_proba) if len(np.unique(y_true))==2 else np.nan,
    )
    return out

def mean_sd(df: pd.DataFrame, by_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(by_cols)[metric_cols]
    m = g.mean().add_suffix("_mean")
    s = g.std(ddof=1).add_suffix("_sd")
    out = pd.concat([m, s], axis=1).reset_index()
    return out

def paired_test(delta: np.ndarray, alternative="greater"):
    # normalitas
    stat_sw, p_sw = shapiro(delta) if len(delta) >= 3 else (np.nan, np.nan)
    normal = bool(p_sw > 0.05) if not np.isnan(p_sw) else False

    if normal:
        # paired t-test (H1: mean(delta) > 0)
        t_stat, p_val = ttest_rel(delta, np.zeros_like(delta), alternative=alternative)
        effect = np.mean(delta) / (np.std(delta, ddof=1) + 1e-12)  # Cohen's d (paired, approx)
        method = "paired_t"
    else:
        # Wilcoxon signed-rank (H1: median(delta) > 0)
        try:
            s_stat, p_val = wilcoxon(delta, alternative=alternative, zero_method="wilcox")
        except ValueError:
            s_stat, p_val = (np.nan, np.nan)
        # Cliff's delta (approx): ( #pairs x_i>x_j - #pairs x_i<x_j ) / n^2
        n = len(delta)
        greater = np.sum(delta > 0); less = np.sum(delta < 0)
        effect = (greater - less) / (n + 1e-12)
        method = "wilcoxon"
    return dict(p_value=float(p_val), effect=float(effect), normal=normal, method=method)