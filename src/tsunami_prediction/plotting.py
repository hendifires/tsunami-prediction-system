from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, y_proba, out_path: Path, title: str):
    fig = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    _save(fig, out_path)

def plot_pr(y_true, y_proba, out_path: Path, title: str):
    fig = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    _save(fig, out_path)

def plot_cm(y_true, y_pred, out_path: Path, title: str):
    fig = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=None)
    plt.title(title)
    _save(fig, out_path)

def plot_reliability(y_true, y_proba, out_path: Path, title: str):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    fig = plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1], "--", alpha=0.6)
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.title(title)
    _save(fig, out_path)

def bar_delta(values: dict[str, float], out_path: Path, title: str):
    fig = plt.figure(figsize=(6,3))
    names = list(values.keys()); vals = [values[k] for k in names]
    plt.bar(names, vals)
    plt.axhline(0, color="black", lw=0.8)
    plt.title(title); plt.ylabel("Δ (SMOTE − NoSMOTE)")
    _save(fig, out_path)