from __future__ import annotations
# cSpell:ignore joblib multiclass

"""
Stacking Ensemble Pipeline (Multi-class 0/1/2) + SMOTE Ablation (simple)

Desain sesuai tesis (versi disederhanakan):

- Masalah:
    * Multi-class classification:
        0 = non-tsunami
        1 = tsunami tektonik
        2 = tsunami vulkanik

- Dataset:
    * Diasumsikan sudah melalui tahap:
        - penggabungan tektonik + vulkanik,
        - filter tahun (>= 1900),
        - labeling 0/1/2 dari kolom resmi (bukan rule manual),
        - imputasi missing,
        - encoding kategorikal seperlunya.
    * Tahap ini dilakukan di modul lain (mis. preprocessing.py).

- Fitur:
    * Hanya fitur yang umum:
        - magnitude, depth, latitude, longitude, alert, sig, VEI, elevation, dst.
      (tergantung hasil preprocessing — di sini kita tidak mengatur lagi).

- Model:
    * Base learners:
        - SVM dengan kernel polynomial (dibungkus StandardScaler)
        - RandomForestClassifier
    * Meta-learner:
        - LogisticRegression (multi-class, max_iter besar)
    * Arsitektur stacking:
        [SVM Polynomial + Random Forest] --> Logistic Regression
      dengan stack_method="predict_proba".

- Skenario imbalance:
    * no_smote : tanpa oversampling
    * smote    : SMOTE standar (imblearn.over_sampling.SMOTE) pada training set saja

- Evaluasi (per model & per skenario):
    * Accuracy
    * Precision (macro)
    * Recall (macro)
    * F1-score (macro)
    * ROC-AUC (multi-class, OVR, macro)
    * Average precision (multi-class, macro)
    * Confusion matrix 3x3
    * ROC curve multi-class
    * Precision-Recall curve multi-class

- Output:
    * reports/tables/stacking_results.csv
        - Satu baris per (scenario, model) dengan semua metrik.
    * reports/tables/stacking_classification_report_<scenario>.txt
        - Classification report per skenario.
    * reports/figures/cm_<scenario>_<model>.png
    * reports/figures/roc_<scenario>_<model>.png
    * reports/figures/pr_<scenario>_<model>.png
    * artifacts/<model>_<scenario>.joblib
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from joblib import dump

from imblearn.over_sampling import SMOTE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

from sklearn.exceptions import ConvergenceWarning

# ---- tame noisy warnings (tidak mempengaruhi hasil) ----
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
ART = ROOT / "artifacts"

for p in (PROCESSED, TAB, FIG, ART):
    p.mkdir(parents=True, exist_ok=True)


# ---------------- Utils sederhana ----------------
def _log(msg: str) -> None:
    """Log singkat ke stdout dengan flush."""
    print(msg, flush=True)


def _savetab(df: pd.DataFrame, path: Path) -> None:
    """Simpan DataFrame ke CSV dengan memastikan foldernya ada."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _savefig(path: Path) -> None:
    """Simpan figure matplotlib ke file PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------- Data loading ----------------
def load_train_test(
    train_path: Path,
    test_path: Path,
    target_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load dataset train & test yang sudah dipreproses.

    Diasumsikan:
    - train_path & test_path berisi kolom fitur + kolom target (target_col).
    - target_col sudah berupa angka: 0, 1, 2.

    Contoh default:
    - data/processed/events_train.csv
    - data/processed/events_test.csv
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    if target_col not in df_train.columns or target_col not in df_test.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in train/test CSV. "
            "Pastikan preprocessing.py sudah menyimpan kolom target dengan nama yang konsisten."
        )

    y_train = df_train.pop(target_col).astype(int)
    y_test = df_test.pop(target_col).astype(int)

    # Pastikan fitur konsisten di train/test
    common_cols = [c for c in df_train.columns if c in df_test.columns]
    X_train = df_train[common_cols].copy()
    X_test = df_test[common_cols].copy()

    _log(
        f"[Data] Loaded train={train_path.name} test={test_path.name} | "
        f"X_train={X_train.shape}, X_test={X_test.shape}"
    )
    return X_train, X_test, y_train, y_test


# ---------------- Model builders ----------------
def build_svm_poly(random_state: int = 42) -> Pipeline:
    """
    SVM dengan kernel polynomial, dibungkus StandardScaler.

    SVM peka terhadap skala, jadi scaling selalu dilakukan di dalam pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="poly",
                    degree=3,            # bisa kamu jelaskan di tesis (polynomial degree 3)
                    C=1.0,
                    gamma="scale",
                    probability=True,    # penting agar bisa pakai predict_proba untuk stacking
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """
    Random Forest sebagai base learner kedua.
    Cocok untuk data non-linear dan cukup robust terhadap outlier.
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
        # class_weight bisa diaktifkan jika mau:
        # class_weight="balanced_subsample",
    )


def build_stacking_classifier(random_state: int = 42) -> StackingClassifier:
    """
    Stacking:
        [SVM Polynomial + Random Forest] --> Logistic Regression

    Meta-learner Logistic Regression memakai input berupa probabilitas
    (stack_method="predict_proba").
    """
    svm_poly = build_svm_poly(random_state=random_state)
    rf = build_random_forest(random_state=random_state)

    estimators = [
        ("svm_poly", svm_poly),
        ("rf", rf),
    ]

    final_estimator = LogisticRegression(
        max_iter=1000,
        multi_class="auto",
        n_jobs=-1,
    )

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    return stacking


# ---------------- Metrics & plotting (multi-class) ----------------
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Non-tsunami", "Tsunami tektonik", "Tsunami vulkanik"]


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> Dict[str, float]:
    """Hitung metrik multi-class utama."""
    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    if y_proba is not None:
        # ROC-AUC multi-class (OVR, macro)
        try:
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            metrics["roc_auc_ovr_macro"] = float("nan")

        # Average precision (PR-AUC) multi-class (macro)
        try:
            y_true_bin = label_binarize(y_true, classes=CLASS_LABELS)
            metrics["avg_precision_macro"] = average_precision_score(
                y_true_bin,
                y_proba,
                average="macro",
            )
        except Exception:
            metrics["avg_precision_macro"] = float("nan")
    else:
        metrics["roc_auc_ovr_macro"] = float("nan")
        metrics["avg_precision_macro"] = float("nan")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    """Plot confusion matrix 3x3 untuk kelas 0/1/2."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    plt.figure(figsize=(5.5, 4.5))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_LABELS))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)

    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center", fontweight="bold")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    _savefig(out_png)


def plot_roc_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    """Plot ROC curve one-vs-rest untuk tiap kelas."""
    y_true_bin = label_binarize(y_true, classes=CLASS_LABELS)

    plt.figure(figsize=(6.0, 4.5))
    for idx, class_id in enumerate(CLASS_LABELS):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_proba[:, idx])
        except Exception:
            continue
        plt.plot(fpr, tpr, label=f"Class {class_id} - {CLASS_NAMES[idx]}")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    _savefig(out_png)


def plot_pr_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    """Plot Precision-Recall curve one-vs-rest untuk tiap kelas."""
    y_true_bin = label_binarize(y_true, classes=CLASS_LABELS)

    plt.figure(figsize=(6.0, 4.5))
    for idx, class_id in enumerate(CLASS_LABELS):
        try:
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, idx],
                y_proba[:, idx],
            )
        except Exception:
            continue
        plt.plot(recall, precision, label=f"Class {class_id} - {CLASS_NAMES[idx]}")

    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=8)
    plt.tight_layout()
    _savefig(out_png)


# ---------------- Train & Eval per model ----------------
def train_and_evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scenario: str,
) -> Dict[str, float]:
    """
    Train + evaluasi satu model pada satu skenario (no_smote / smote).

    Mengembalikan dictionary metrik + durasi training.
    """
    _log(f"[{scenario}] Fit {name} …")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_sec = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    metrics = compute_multiclass_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
    )
    metrics["model"] = name
    metrics["scenario"] = scenario
    metrics["fit_sec"] = round(fit_sec, 3)

    _log(
        f"[{scenario}] {name}: "
        f"acc={metrics['accuracy']:.3f} "
        f"f1_macro={metrics['f1_macro']:.3f} "
        f"roc_auc={metrics['roc_auc_ovr_macro']:.3f}"
    )

    # Simpan confusion matrix + ROC/PR curve jika probabilitas tersedia
    cm_png = FIG / f"cm_{scenario}_{name}.png"
    plot_confusion_matrix(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        title=f"{name.upper()} ({scenario}) - Confusion Matrix",
        out_png=cm_png,
    )

    if y_proba is not None:
        roc_png = FIG / f"roc_{scenario}_{name}.png"
        pr_png = FIG / f"pr_{scenario}_{name}.png"
        plot_roc_multiclass(
            y_true=y_test.to_numpy(),
            y_proba=y_proba,
            title=f"{name.upper()} ({scenario}) - ROC (OVR)",
            out_png=roc_png,
        )
        plot_pr_multiclass(
            y_true=y_test.to_numpy(),
            y_proba=y_proba,
            title=f"{name.upper()} ({scenario}) - Precision-Recall",
            out_png=pr_png,
        )

    # Simpan classification report per skenario (sekali saja, untuk stacking)
    if name == "stacking":
        report_txt = TAB / f"stacking_classification_report_{scenario}.txt"
        report_txt.parent.mkdir(parents=True, exist_ok=True)
        with report_txt.open("w", encoding="utf-8") as f:
            f.write(
                classification_report(
                    y_test,
                    y_pred,
                    labels=CLASS_LABELS,
                    target_names=CLASS_NAMES,
                    zero_division=0,
                )
            )

    return metrics


# ---------------- Skenario: no-SMOTE vs SMOTE ----------------
def run_scenario(
    scenario: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> List[Dict[str, float]]:
    """
    Jalankan 3 model (SVM poly, RF, Stacking) untuk satu skenario.

    scenario: "no_smote" atau "smote"
    """
    # 1) Handling imbalance (kalau skenario smote)
    if scenario == "smote":
        _log("[smote] Apply SMOTE on training set …")
        sm = SMOTE(random_state=random_state)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        _log(f"[smote] After SMOTE: X_train={X_train_bal.shape}")
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # 2) Build models
    svm_poly = build_svm_poly(random_state=random_state)
    rf = build_random_forest(random_state=random_state)
    stacking = build_stacking_classifier(random_state=random_state)

    # 3) Train & evaluate
    metrics_rows: List[Dict[str, float]] = []
    metrics_rows.append(
        train_and_evaluate_model(
            name="svm_poly",
            model=svm_poly,
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_test=X_test,
            y_test=y_test,
            scenario=scenario,
        )
    )
    metrics_rows.append(
        train_and_evaluate_model(
            name="rf",
            model=rf,
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_test=X_test,
            y_test=y_test,
            scenario=scenario,
        )
    )
    metrics_rows.append(
        train_and_evaluate_model(
            name="stacking",
            model=stacking,
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_test=X_test,
            y_test=y_test,
            scenario=scenario,
        )
    )

    # 4) Simpan model ke artifacts
    #    (hanya pakai model hasil training pada data balancing final)
    dump(
        svm_poly,
        ART / f"svm_poly_{scenario}.joblib",
        compress=3,
    )
    dump(
        rf,
        ART / f"rf_{scenario}.joblib",
        compress=3,
    )
    dump(
        stacking,
        ART / f"stacking_{scenario}.joblib",
        compress=3,
    )

    return metrics_rows


# ---------------- CLI utama ----------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stacking Ensemble (SVM poly + RF -> LR) untuk prediksi tsunami "
            "multi-class (0=non-tsunami, 1=tektonik, 2=vulkanik) "
            "dengan dua skenario: no-SMOTE dan SMOTE."
        )
    )
    ap.add_argument(
        "--train-csv",
        type=str,
        default=str(PROCESSED / "events_train.csv"),
        help="Path ke CSV train yang sudah dipreproses (default: data/processed/events_train.csv)",
    )
    ap.add_argument(
        "--test-csv",
        type=str,
        default=str(PROCESSED / "events_test.csv"),
        help="Path ke CSV test yang sudah dipreproses (default: data/processed/events_test.csv)",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="label",
        help="Nama kolom target multi-class (0/1/2). Default: label",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state untuk reprodusibilitas.",
    )
    ap.add_argument(
        "--no-smote",
        action="store_true",
        help="Jika di-set, hanya jalankan skenario tanpa SMOTE (no_smote).",
    )
    ap.add_argument(
        "--only-smote",
        action="store_true",
        help="Jika di-set, hanya jalankan skenario SMOTE (smote).",
    )

    args = ap.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)

    _log("[Main] Load train/test data …")
    X_train, X_test, y_train, y_test = load_train_test(
        train_path=train_path,
        test_path=test_path,
        target_col=args.target_col,
    )

    all_metrics: List[Dict[str, float]] = []

    # Tentukan skenario mana yang dijalankan
    if args.only_smote and args.no_smote:
        raise ValueError("Tidak bisa menggunakan --no-smote dan --only-smote bersamaan.")
    elif args.only_smote:
        scenarios = ["smote"]
    elif args.no_smote:
        scenarios = ["no_smote"]
    else:
        scenarios = ["no_smote", "smote"]

    for scenario in scenarios:
        _log(f"[Main] Run scenario: {scenario}")
        rows = run_scenario(
            scenario=scenario,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=args.random_state,
        )
        all_metrics.extend(rows)

    # Simpan ringkasan metrik semua skenario ke satu CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        out_csv = TAB / "stacking_results.csv"
        _savetab(metrics_df, out_csv)
        _log(f"[Main] Metrics written to: {out_csv}")

    _log(
        f"[DONE] Stacking pipeline finished.\n"
        f" - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}\n - DATA: {PROCESSED}"
    )


if __name__ == "__main__":
    main()