from __future__ import annotations
# cSpell:ignore nosmote smote joblib multiclass svm rbf poly

"""
Stacking Ensemble (multiclass 0/1/2) untuk prediksi tsunami berbasis dataset gabungan `events`.

Desain akhir (selaras metodologi tesis):

- Dataset:
    * Tanpa SMOTE (baseline) :
        data/processed/events_train.csv
        data/processed/events_test.csv
    * Dengan SMOTE :
        data/processed/events_train_smote.csv
        data/processed/events_test.csv

  Target multi-class:
    - kolom `label` (0=non-tsunami, 1=tsunami tektonik, 2=tsunami vulkanik)
      dibuat oleh preprocessing.py
    - fallback kompatibilitas: `tsunami_label`

- Base learners:
    * svm_rbf   : SVC(kernel="rbf") + StandardScaler (Pipeline)
    * svm_poly  : SVC(kernel="poly") + StandardScaler (Pipeline)
    * rf        : RandomForestClassifier

- Meta-learner (stacking):
    * stacking  : StackingClassifier dengan LogisticRegression (multi-class)
      menggunakan stack_method="predict_proba"

- Skenario yang dijalankan:
    * nosmote : train pada events_train.csv
    * smote   : train pada events_train_smote.csv

- Output utama per skenario:
    * artifacts/events_<variant>_svm_rbf.joblib
    * artifacts/events_<variant>_svm_poly.joblib
    * artifacts/events_<variant>_rf.joblib
    * artifacts/events_<variant>_stacking.joblib

    * reports/tables/events_<variant>_metrics.csv
    * reports/tables/events_<variant>_stacking_preds.csv

    * reports/figures/events_<variant>_<model>_cm.png   (confusion matrix)
    * reports/figures/events_<variant>_<model>_roc.png  (ROC macro)

Pipeline ini sengaja dibuat ringkas agar mudah dijelaskan di tesis,
namun tetap mencakup:
    - baseline model SVM & Random Forest,
    - dan model utama Stacking Ensemble (SVM+RF -> Logistic Regression).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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


# ---------------- Util umum ----------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _savetab(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _find_target_col(df: pd.DataFrame) -> str:
    """
    Cari kolom target multiclass.
    Versi final tesis mengasumsikan kolom 'label'.
    Disediakan fallback 'tsunami_label' untuk kompatibilitas minimal.
    """
    for cand in ("label", "tsunami_label"):
        if cand in df.columns:
            return cand
    raise KeyError(
        "Tidak menemukan kolom target 'label' atau 'tsunami_label' "
        f"di kolom: {list(df.columns)}"
    )


# ---------------- Load data events (nosmote / smote) ----------------
def load_events_dataset(
    variant: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load train/test untuk dataset gabungan 'events' sesuai varian imbalance.

    Parameters
    ----------
    variant : {"nosmote", "smote"}

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    variant = variant.lower()
    if variant not in {"nosmote", "smote"}:
        raise ValueError("variant harus 'nosmote' atau 'smote'.")

    if variant == "nosmote":
        p_train = PROCESSED / "events_train.csv"
    else:
        p_train = PROCESSED / "events_train_smote.csv"

    p_test = PROCESSED / "events_test.csv"

    if not p_train.exists():
        raise FileNotFoundError(f"[Stack] Train file tidak ditemukan: {p_train}")
    if not p_test.exists():
        raise FileNotFoundError(f"[Stack] Test file tidak ditemukan: {p_test}")

    df_tr = _read_csv(p_train)
    df_te = _read_csv(p_test)

    target = _find_target_col(df_tr)
    if target not in df_te.columns:
        raise KeyError(
            f"[Stack] Kolom target '{target}' tidak ada di test set. "
            f"Kolom test: {list(df_te.columns)}"
        )

    y_train = df_tr[target].astype("int64")
    y_test = df_te[target].astype("int64")

    X_train = df_tr.drop(columns=[target])
    X_test = df_te.drop(columns=[target])

    _log(
        f"[Stack] events ({variant}): "
        f"X_train={X_train.shape}, X_test={X_test.shape}, target='{target}'"
    )
    return X_train, X_test, y_train, y_test


# ---------------- Model builders ----------------
def build_base_models(random_state: int) -> Dict[str, object]:
    """
    Base learners final untuk tesis:
    - svm_rbf   : SVC(RBF) + StandardScaler
    - svm_poly  : SVC(poly, degree=3) + StandardScaler
    - rf        : RandomForestClassifier
    """
    base: Dict[str, object] = {}

    base["svm_rbf"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    base["svm_poly"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="poly",
                    degree=3,
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    base["rf"] = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return base


def build_stacking(
    base_models: Dict[str, object],
    random_state: int,
) -> StackingClassifier:
    """
    Stacking Ensemble:
        [SVM-RBF, SVM-Poly, Random Forest] -> Logistic Regression (multiclass)

    - Menggunakan stack_method="predict_proba"
    - Tanpa passthrough fitur asli (passthrough=False)
      agar arsitektur lebih sederhana untuk dijelaskan.
    """
    estimators = list(base_models.items())

    meta = LogisticRegression(
        solver="lbfgs",
        max_iter=10_000,
        multi_class="auto",
        class_weight="balanced",
        n_jobs=-1,
    )

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        passthrough=False,
        cv=5,
        stack_method="predict_proba",
        n_jobs=1,
    )
    return stack


# ---------------- Evaluasi & visualisasi ----------------
def eval_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Hitung metrik multiclass makro:
    - accuracy
    - precision_macro
    - recall_macro
    - f1_macro
    - roc_auc_macro (opsional, jika y_proba tersedia)
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics: Dict[str, float] = {
        "accuracy": float(acc),
        "prec_macro": float(prec),
        "rec_macro": float(rec),
        "f1_macro": float(f1m),
    }

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            metrics["roc_auc_macro"] = float(auc)
        except Exception:
            metrics["roc_auc_macro"] = float("nan")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_png: Path,
    title: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plot confusion matrix sederhana dan simpan sebagai PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        labels = [str(i) for i in range(cm.shape[0])]
    else:
        labels = class_names

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_multiclass(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_png: Path,
    title: str,
    n_classes: int,
) -> None:
    """
    Plot ROC multi-class (one-vs-rest) + macro-average.

    Catatan:
    - Memerlukan y_proba dengan shape (n_samples, n_classes).
    - y_true adalah label integer 0..(n_classes-1).
    """
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    for i in range(n_classes):
        try:
            RocCurveDisplay.from_predictions(
                y_bin[:, i],
                y_proba[:, i],
                name=f"class {i}",
                ax=ax,
            )
        except Exception:
            continue

    try:
        auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        ax.set_title(f"{title} (macro AUC={auc_macro:.3f})")
    except Exception:
        ax.set_title(title)

    ax.grid(True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fit_eval_and_save(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    variant: str,
) -> Tuple[Dict[str, float], object]:
    """
    Fit satu model, evaluasi di test set, simpan model + visualisasi.

    Return:
      (metrics_dict, fitted_model)
    """
    _log(f"[Stack]   > Fit {model_name} ({variant}) …")
    fitted = clone(model)
    fitted.fit(X_train, y_train)

    y_pred = fitted.predict(X_test)

    if hasattr(fitted, "predict_proba"):
        y_proba: Optional[np.ndarray] = fitted.predict_proba(X_test)
    else:
        y_proba = None

    metrics = eval_multiclass(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
    )
    metrics["model"] = model_name
    metrics["variant"] = variant

    _log(
        "[Stack]   > Done %s (%s) | acc=%.3f f1_macro=%.3f prec_macro=%.3f rec_macro=%.3f"
        % (
            model_name,
            variant,
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["prec_macro"],
            metrics["rec_macro"],
        )
    )

    model_path = ART / f"events_{variant}_{model_name}.joblib"
    dump(fitted, model_path)
    _log(f"[Stack]   > Saved model to {model_path}")

    cm_path = FIG / f"events_{variant}_{model_name}_cm.png"
    plot_confusion_matrix(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        out_png=cm_path,
        title=f"{model_name} ({variant}) - Confusion Matrix",
        class_names=["non-tsunami (0)", "tektonik (1)", "vulkanik (2)"],
    )
    _log(f"[Stack]   > Saved confusion matrix to {cm_path}")

    if y_proba is not None:
        roc_path = FIG / f"events_{variant}_{model_name}_roc.png"
        n_classes = len(np.unique(y_test.to_numpy()))
        try:
            plot_roc_multiclass(
                y_true=y_test.to_numpy(),
                y_proba=y_proba,
                out_png=roc_path,
                title=f"{model_name} ({variant}) - ROC",
                n_classes=n_classes,
            )
            _log(f"[Stack]   > Saved ROC curve to {roc_path}")
        except Exception as exc:
            _log(f"[Stack]   > Skip ROC plot for {model_name} ({variant}) ({exc})")

    return metrics, fitted


# ---------------- Pipeline per varian (nosmote/smote) ----------------
def run_variant(
    variant: str,
    random_state: int,
) -> pd.DataFrame:
    """
    Jalankan seluruh eksperimen untuk satu varian imbalance:
      - Train & evaluasi svm_rbf, svm_poly, rf, dan stacking.
      - Simpan model, metrik, preds (khusus stacking), dan visualisasi.
    """
    _log(f"[Stack] === events | variant={variant} ===")

    X_train, X_test, y_train, y_test = load_events_dataset(variant)

    base_models = build_base_models(random_state=random_state)
    stacking_model = build_stacking(base_models=base_models, random_state=random_state)

    rows: List[Dict[str, float]] = []

    # Base learners
    for name, est in base_models.items():
        m, _ = fit_eval_and_save(
            model_name=name,
            model=est,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            variant=variant,
        )
        rows.append(m)

    # Stacking (sekali fit saja, model dipakai juga untuk simpan preds)
    m_stack, stacking_fitted = fit_eval_and_save(
        model_name="stacking",
        model=stacking_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        variant=variant,
    )
    rows.append(m_stack)

    # Simpan preds stacking (untuk analisis detail)
    y_pred_stack = stacking_fitted.predict(X_test)
    preds_df = pd.DataFrame(
        {
            "y_true": y_test.to_numpy(),
            "y_pred": y_pred_stack,
        }
    )
    preds_path = TAB / f"events_{variant}_stacking_preds.csv"
    _savetab(preds_df, preds_path)
    _log(f"[Stack]   > Saved stacking predictions to {preds_path}")

    df_metrics = pd.DataFrame(rows)
    if not df_metrics.empty:
        df_metrics = df_metrics.sort_values(
            by=["f1_macro", "accuracy"],
            ascending=False,
        )
        p_metrics = TAB / f"events_{variant}_metrics.csv"
        _savetab(df_metrics, p_metrics)
        _log(f"[Stack] Saved metrics for variant='{variant}' to {p_metrics}")

    return df_metrics


# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Stacking Ensemble (SVM RBF + SVM Poly + RF -> Logistic Regression) "
            "untuk prediksi tsunami multi-class (0/1/2) pada dataset events."
        )
    )
    ap.add_argument(
        "--variant",
        type=str,
        default="both",
        choices=["nosmote", "smote", "both"],
        help=(
            "Varian imbalance yang dijalankan:\n"
            " - nosmote : tanpa oversampling (baseline)\n"
            " - smote   : dengan SMOTE pada train set\n"
            " - both    : jalankan keduanya"
        ),
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed untuk semua model (SVM, RF, dan Logistic Regression).",
    )

    args = ap.parse_args()

    _log("[Stack] Pipeline start …")

    if args.variant == "both":
        variants = ["nosmote", "smote"]
    else:
        variants = [args.variant]

    all_metrics: List[pd.DataFrame] = []
    for v in variants:
        df_v = run_variant(variant=v, random_state=args.random_state)
        all_metrics.append(df_v)

    if all_metrics:
        df_all = pd.concat(all_metrics, ignore_index=True)
        p_all = TAB / "events_all_variants_metrics.csv"
        _savetab(df_all, p_all)
        _log(f"[Stack] Saved combined metrics (all variants) to {p_all}")

    _log(
        f"[DONE] Stacking pipeline finished.\n"
        f" - FIG: {FIG}\n"
        f" - TAB: {TAB}\n"
        f" - ART: {ART}\n"
        f" - DATA: {PROCESSED}"
    )


if __name__ == "__main__":
    main()