from __future__ import annotations
# cSpell:ignore nosmote smoteenn tomek writeable joblib OOF

"""
Stacking Ensemble Pipeline + SMOTE Ablation (fast & robust)

- Compare: non-SMOTE vs SMOTE variants (smote, smote_tomek, smoteenn) per dataset.
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, FN.
- Outputs per run: metrics.csv, preds.csv, CM/ROC/PR figures, model.joblib
- Aggregated ablation: reports/tables/ablation_smote.csv

Tambahan:
- Kolom waktu: fit_sec (per model & stacking), grid_sec (waktu meta-grid), run_sec (durasi total run_one)
- Simpan confusion matrix sebagai gambar (reports/figures) dan tabel CSV (reports/tables)
- Cari decision_threshold optimal (F-beta) via OOF CV, disimpan ke artifact (dipakai serve_api).
"""

import argparse
import time
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import warnings
from sklearn.exceptions import ConvergenceWarning

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

# ---- tame noisy warnings (tidak mempengaruhi hasil) ----
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
ART = ROOT / "artifacts"
for p in [PROCESSED, TAB, FIG, ART]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- SMOTE tag helper ----------
_TAG_MAP = {"smote": "smote", "smote_tomek": "smote_tomek", "smoteenn": "smote_enn"}
def _variant_to_tag(variant: Optional[str]) -> str:
    return _TAG_MAP.get((variant or "smote").lower(), (variant or "smote").lower())

SMOTE_VARIANTS = ["smote", "smote_tomek", "smoteenn"]

# ---------------- Utils ----------------
def _log(msg: str) -> None:
    print(msg, flush=True)

def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def _savetab(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def _safe_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p
    if hasattr(model, "decision_function"):
        z = np.clip(model.decision_function(X), -20, 20)
        return 1.0 / (1.0 + np.exp(-z))
    return model.predict(X)

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    out: Dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        with suppress(Exception): out["roc_auc"] = roc_auc_score(y_true, y_prob)
        with suppress(Exception): out["pr_auc"] = average_precision_score(y_true, y_prob)
        out.setdefault("roc_auc", float("nan"))
        out.setdefault("pr_auc", float("nan"))
    else:
        out.update({"roc_auc": float("nan"), "pr_auc": float("nan")})
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return out

# -------- Sanitize feature names for XGBoost & Stacking --------
_BAD_CHARS = {"[": "(", "]": ")", "<": "_lt_", ">": "_gt_", "{": "(", "}": ")", "/": "_", "\\": "_", ":": "_", ";": "_", ",": "_", "=": "_"}
def _safe_name(name: object) -> str:
    s = str(name)
    for k, v in _BAD_CHARS.items():
        s = s.replace(k, v)
    return s.replace(" ", "_")

def sanitize_df_pair(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    ori_cols = list(Xtr.columns)
    new_cols: list[str] = []
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for c in ori_cols:
        base = _safe_name(c)
        name = base
        i = 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used.add(name)
        new_cols.append(name)
        mapping[str(c)] = name
    Xtr2 = Xtr.copy(); Xtr2.columns = new_cols
    xte_cols = [mapping.get(str(c), _safe_name(c)) for c in Xte.columns]
    Xte2 = Xte.copy(); Xte2.columns = xte_cols
    return Xtr2, Xte2, mapping

# ---------------- Data loading ----------------
def load_split(dataset: str, use_smote: bool, smote_variant: Optional[str] = None
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if use_smote:
        tag = _variant_to_tag(smote_variant)
        cand = PROCESSED / f"{dataset}_train_{tag}.csv"
        p_tr = cand if cand.exists() else (PROCESSED / f"{dataset}_train_smote.csv")
    else:
        p_tr = PROCESSED / f"{dataset}_train.csv"
    p_te = PROCESSED / f"{dataset}_test.csv"

    if not p_tr.exists() or not p_te.exists():
        missing = " ".join(n for p, n in [(p_tr, p_tr.name), (p_te, p_te.name)] if not p.exists())
        raise FileNotFoundError(f"[Stack] Split not found. Run smote_pipeline first. Missing: {missing}")

    df_tr, df_te = _read_csv(p_tr), _read_csv(p_te)
    if "tsu" not in df_tr.columns or "tsu" not in df_te.columns:
        raise KeyError("[Stack] Target column 'tsu' not found in splits.")

    y_train = df_tr.pop("tsu").astype(int)
    y_test  = df_te.pop("tsu").astype(int)

    common = [c for c in df_tr.columns if c in df_te.columns]
    X_train, X_test = df_tr[common], df_te[common]

    mode = _variant_to_tag(smote_variant) if use_smote else "no-smote"
    _log(f"[Stack] {dataset} ({mode}): X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test

# ---------------- Feature selection (opsional) ----------------
def pearson_topn(X: pd.DataFrame, y: pd.Series, top_n: int, out_tab: Path, out_fig: Path) -> List[str]:
    num_cols = X.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        _savetab(pd.DataFrame(columns=["feature", "abs_corr"]), out_tab)
        return list(X.columns)
    vals: List[tuple[str, float]] = []
    yv = y.to_numpy()
    for c in num_cols:
        xc = X[c].to_numpy()
        corr = 0.0 if np.nanstd(xc) == 0 else float(np.corrcoef(xc, yv)[0, 1])
        if not np.isfinite(corr): corr = 0.0
        vals.append((c, abs(corr)))
    corr_df = pd.DataFrame(vals, columns=["feature", "abs_corr"]).sort_values("abs_corr", ascending=False)
    _savetab(corr_df, out_tab)
    top = corr_df.head(top_n)
    plt.figure(figsize=(8, max(3.5, 0.25 * len(top))))
    plt.barh(top["feature"][::-1], top["abs_corr"][::-1])
    plt.title(f"Top-{top_n} Pearson |corr| vs tsu")
    _savefig(out_fig)
    return top["feature"].tolist()

def rfe_select(X: pd.DataFrame, y: pd.Series, top_n: int, out_tab: Path, out_fig: Path, random_state: int = 42) -> List[str]:
    n = min(top_n, X.shape[1]) if X.shape[1] else 0
    if n == 0:
        _savetab(pd.DataFrame(columns=["feature", "selected"]), out_tab)
        return list(X.columns)
    base = RandomForestClassifier(n_estimators=200, random_state=random_state,
                                 n_jobs=-1, class_weight="balanced_subsample")
    selector = RFE(base, n_features_to_select=n).fit(X, y)
    mask = selector.support_
    selected = X.columns[mask].tolist()
    _savetab(pd.DataFrame({"feature": X.columns, "selected": mask}), out_tab)
    plt.figure(figsize=(8, max(3.5, 0.25 * len(selected))))
    plt.barh(selected[::-1], np.arange(1, len(selected) + 1)[::-1])
    plt.title(f"RFE top-{n} (train)")
    _savefig(out_fig)
    return selected

def maybe_feature_select(Xtr: pd.DataFrame, ytr: pd.Series, Xte: pd.DataFrame,
                         dataset: str, suffix: str, method: str, top_n: int, random_state: int
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    method = method.lower()
    if method == "pearson":
        sel = pearson_topn(Xtr, ytr, top_n,
                           out_tab=TAB / f"{dataset}_stack_{suffix}_pearson.csv",
                           out_fig=FIG / f"{dataset}_stack_{suffix}_pearson.png")
    elif method == "rfe":
        sel = rfe_select(Xtr, ytr, top_n,
                         out_tab=TAB / f"{dataset}_stack_{suffix}_rfe.csv",
                         out_fig=FIG / f"{dataset}_stack_{suffix}_rfe.png",
                         random_state=random_state)
    else:
        return Xtr, Xte, list(Xtr.columns)
    Xtr_s = Xtr[sel]
    sel_test = [c for c in sel if c in Xte.columns]
    Xte_s = Xte[sel_test]
    return Xtr_s, Xte_s, sel_test

# ---------------- Helpers (non-lambda, picklable) ----------------
def to_np_writable(X):
    arr = np.asarray(X, dtype=float)
    if not arr.flags.writeable:
        arr = arr.copy()
    return arr

# ---------------- Models ----------------
def build_base_learners(random_state: int, with_xgb: bool, with_mlp: bool) -> List[Tuple[str, object]]:
    """Semua base learners dikemas Pipeline. NB diberi transformer agar array writable (hindari ValueError)."""
    to_np_tf = FunctionTransformer(to_np_writable, validate=False)

    learners: List[Tuple[str, object]] = [
        ("lr",  Pipeline([("scaler", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])),
        ("sgd", Pipeline([("scaler", StandardScaler()),
                          ("clf", SGDClassifier(loss="log_loss", alpha=1e-4,
                                                class_weight="balanced",
                                                max_iter=2000, early_stopping=True,
                                                n_iter_no_change=5, random_state=random_state))])),
        ("knn", Pipeline([("scaler", StandardScaler()),
                          ("clf", KNeighborsClassifier(n_neighbors=11))])),
        ("nb",  Pipeline([("to_np", to_np_tf),
                          ("clf", GaussianNB())])),
        ("dt",  DecisionTreeClassifier(random_state=random_state, class_weight="balanced")),
        ("rf",  RandomForestClassifier(n_estimators=300, random_state=random_state,
                                      n_jobs=-1, class_weight="balanced_subsample")),
        ("gb",  GradientBoostingClassifier(random_state=random_state)),
    ]
    if with_mlp:
        with suppress(Exception):
            from sklearn.neural_network import MLPClassifier
            learners.append(("mlp", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(50,), max_iter=300, random_state=random_state)),
            ])))
    if with_xgb:
        with suppress(Exception):
            from xgboost import XGBClassifier  # type: ignore
            learners.append(("xgb", XGBClassifier(
                n_estimators=200, learning_rate=0.08, max_depth=5,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=random_state, n_jobs=-1,
                tree_method="hist",
            )))
    return learners

def build_stacking(base_learners: List[Tuple[str, object]], cv: int) -> StackingClassifier:
    """n_jobs=1 untuk menghindari memmap read-only di proses paralel yang memicu WRITEABLE error."""
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(max_iter=8000, class_weight="balanced"),
        cv=cv,
        passthrough=True,
        stack_method="predict_proba",
        n_jobs=1,
    )

# ---------------- Threshold tuning ----------------
def _f_beta(y_true: np.ndarray, y_prob: np.ndarray, thr: float, beta: float) -> Tuple[float, float, float]:
    y_pred = (y_prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    if prec == 0 and rec == 0:
        return 0.0, prec, rec
    b2 = beta ** 2
    fbeta = (1 + b2) * (prec * rec) / (b2 * prec + rec) if (b2 * prec + rec) > 0 else 0.0
    return fbeta, prec, rec

def _plot_thr_sweep(df: pd.DataFrame, thr: float, title: str, out_png: Path) -> None:
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(df["threshold"], df["f_beta"], label="F-beta")
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.axvline(thr, linestyle="--")
    plt.title(title or "Threshold sweep")
    plt.xlabel("Threshold"); plt.ylabel("Score")
    plt.legend()
    _savefig(out_png)

def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 1.0,
    min_precision: Optional[float] = None,
    title: str = "",
    out_csv: Optional[Path] = None,
    out_png: Optional[Path] = None,
) -> float:
    """Sweep threshold & pilih yang memaksimalkan F-beta (opsional: syarat min precision)."""
    grid = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in grid:
        f, p, r = _f_beta(y_true, y_prob, t, beta)
        rows.append((t, f, p, r))
    df = pd.DataFrame(rows, columns=["threshold", "f_beta", "precision", "recall"])

    if min_precision is None:
        best = df.loc[df["f_beta"].idxmax()]
    else:
        cand = df[df["precision"] >= float(min_precision)]
        if cand.empty:
            best = df.loc[df["f_beta"].idxmax()]
        else:
            best = cand.loc[cand["f_beta"].idxmax()]

    thr = float(best["threshold"])
    if out_csv is not None:
        _savetab(df, out_csv)
    if out_png is not None:
        _plot_thr_sweep(df, thr, title, out_png)
    return thr

# ---------------- Plots ----------------
def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_png: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm): plt.text(j, i, int(v), ha="center", va="center", fontweight="bold")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); _savefig(out_png)

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_png: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5.6, 4.2)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
    plt.title(title); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); _savefig(out_png)

def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_png: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5.6, 4.2)); plt.plot(recall, precision)
    plt.title(title); plt.xlabel("Recall"); plt.ylabel("Precision"); _savefig(out_png)

# ---------------- Train & Eval ----------------
def fit_and_eval_single(name: str, clf, Xtr, ytr, Xte, yte) -> Dict[str, float]:
    t0 = time.time(); _log(f"[Stack]   > Fit {name} …")
    model = clone(clf); model.fit(Xtr, ytr)
    pred = model.predict(Xte); prob = _safe_proba(model, Xte)
    m = _metrics(yte, pred, prob); m["model"] = name
    m["fit_sec"] = round(time.time() - t0, 3)
    _log(f"[Stack]   > Done {name} in {m['fit_sec']:.2f}s | f1={m['f1']:.3f} rec={m['recall']:.3f}")
    return m

def run_one(
    dataset: str, use_smote: bool, smote_variant: Optional[str],
    cv: int, random_state: int, with_xgb: bool, with_mlp: bool,
    fast_sample: Optional[int], feature_select: str, top_n: int, meta_grid: bool,
) -> Dict[str, float]:
    start_run = time.time()
    suffix = _variant_to_tag(smote_variant) if use_smote else "nosmote"
    _log(f"[Stack] Start -> {dataset} ({suffix})")

    p_metrics = TAB / f"{dataset}_stack_{suffix}_metrics.csv"
    p_preds   = TAB / f"{dataset}_stack_{suffix}_preds.csv"
    p_cm_png  = FIG / f"{dataset}_stack_{suffix}_cm.png"
    p_cm_tab  = TAB / f"{dataset}_stack_{suffix}_cm.csv"
    p_roc     = FIG / f"{dataset}_stack_{suffix}_roc.png"
    p_pr      = FIG / f"{dataset}_stack_{suffix}_pr.png"
    p_thr_csv = TAB / f"{dataset}_stack_{suffix}_thr_sweep.csv"
    p_thr_png = FIG / f"{dataset}_stack_{suffix}_thr_sweep.png"
    p_model   = ART / f"{dataset}_stack_{suffix}.joblib"

    Xtr, Xte, ytr, yte = load_split(dataset, use_smote=use_smote, smote_variant=smote_variant)

    # sanitasi nama kolom (untuk XGB & konsistensi)
    Xtr, Xte, name_map = sanitize_df_pair(Xtr, Xte)

    if fast_sample is not None and len(Xtr) > fast_sample:
        _log(f"[Stack]   > FAST mode sample {fast_sample}/{len(Xtr)}")
        idx = np.random.RandomState(42).choice(len(Xtr), size=fast_sample, replace=False)
        Xtr = Xtr.iloc[idx]; ytr = ytr.iloc[idx]

    Xtr, Xte, sel_cols = maybe_feature_select(Xtr, ytr, Xte, dataset, suffix, feature_select, top_n, random_state)

    base_learners = build_base_learners(random_state, with_xgb, with_mlp)
    stack = build_stacking(base_learners, cv)

    grid_sec = 0.0
    best_meta = None
    if meta_grid:
        _log("[Stack]   > Grid meta-learner …")
        param_grid = {
            "final_estimator__C": [0.1, 1.0, 10.0],
            "final_estimator__solver": ["lbfgs", "liblinear"],
        }
        t_grid = time.time()
        grid = GridSearchCV(stack, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=1, verbose=1)
        grid.fit(Xtr, ytr)
        grid_sec = round(time.time() - t_grid, 3)
        best_meta = getattr(grid, "best_params_", None)
        _log(f"[Stack]   > Meta-grid done in {grid_sec:.2f}s | best={best_meta}")
        stack = grid.best_estimator_

    # --- Base learners (waktu per-model ikut ke metrics.csv) ---
    rows = [fit_and_eval_single(name, est, Xtr, ytr, Xte, yte) for name, est in base_learners]

    # --- Stacking: fit, proba test, threshold tuning (OOF) ---
    _log("[Stack]   > Fit Stacking …")
    t0 = time.time()
    stack.fit(Xtr, ytr)
    # Prob test
    y_prob_test = _safe_proba(stack, Xte)
    # OOF prob untuk cari threshold (lebih aman dari pada pakai test)
    _log("[Stack]   > Cross-val predict for threshold …")
    with suppress(Exception):
        oof = cross_val_predict(stack, Xtr, ytr, method="predict_proba", cv=cv)
        y_prob_oof = oof[:, 1] if oof.ndim == 2 else oof
    if "y_prob_oof" not in locals():
        _log("[WARN] cross_val_predict failed; fallback: in-sample prob for threshold")
        y_prob_oof = _safe_proba(stack, Xtr)

    # Dataset-specific preference: volcanic lebih pro-recall
    beta = 2.0 if dataset.lower() == "volcanic" else 1.0
    min_prec = 0.50 if dataset.lower() == "volcanic" else None
    thr = find_best_threshold(
        ytr.to_numpy(), y_prob_oof, beta=beta, min_precision=min_prec,
        title=f"{dataset.title()} ({suffix}) Threshold sweep (beta={beta})",
        out_csv=p_thr_csv, out_png=p_thr_png,
    )
    _log(f"[Stack]   > Best threshold = {thr:.3f} (beta={beta}, min_precision={min_prec})")

    # Prediksi test memakai threshold terbaik
    y_pred = (y_prob_test >= thr).astype(int)
    m_stack = _metrics(yte.to_numpy(), y_pred, y_prob_test); m_stack["model"] = "stacking_lr_meta"
    m_stack["fit_sec"] = round(time.time() - t0, 3)
    m_stack["grid_sec"] = grid_sec
    rows.append(m_stack)
    _log(f"[Stack]   > Done Stacking in {m_stack['fit_sec']:.2f}s | f1={m_stack['f1']:.3f} rec={m_stack['recall']:.3f}")

    # --- Tabel metrics lengkap (+fit_sec) ---
    metrics_df = pd.DataFrame(rows)
    if {"f1", "recall", "roc_auc"} <= set(metrics_df.columns):
        metrics_df = metrics_df.sort_values(["f1", "recall", "roc_auc"], ascending=False)
    _savetab(metrics_df, p_metrics)

    # --- Simpan preds ---
    preds_df = pd.DataFrame({"y_true": yte, "y_pred": y_pred, "y_prob": y_prob_test})
    _savetab(preds_df, p_preds)

    # --- Confusion matrix: gambar + tabel CSV ---
    cm = confusion_matrix(yte.to_numpy(), y_pred, labels=[0, 1])
    plot_confusion(yte.to_numpy(), y_pred, f"{dataset.title()} - {suffix}", p_cm_png)
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    _savetab(cm_df, p_cm_tab)

    # --- ROC & PR curve (berdasarkan prob test) ---
    with suppress(Exception):
        plot_roc(yte.to_numpy(), y_prob_test, f"{dataset.title()} - ROC ({suffix})", p_roc)
        plot_pr(yte.to_numpy(), y_prob_test, f"{dataset.title()} - PR ({suffix})", p_pr)

    # --- Simpan artifact model + metadata waktu + threshold ---
    run_sec = round(time.time() - start_run, 3)
    dump({
        "model": stack,
        "feature_columns": list(Xtr.columns),
        "selected_columns": sel_cols,
        "feature_select": feature_select,
        "col_name_map": name_map,
        "sanitized_feature_names": True,
        "decision_threshold": float(thr),
        "runtime": {"stack_fit_sec": m_stack["fit_sec"], "grid_sec": grid_sec, "run_sec": run_sec},
        "meta": {"dataset": dataset, "use_smote": use_smote, "smote_variant": smote_variant,
                 "cv": cv, "random_state": random_state, "best_meta": best_meta,
                 "beta": beta, "min_precision": min_prec},
    }, p_model, compress=3)

    # --- Nilai untuk ablation_smote.csv (ringkas) ---
    return {
        "dataset": dataset,
        "use_smote": use_smote,
        "variant": _variant_to_tag(smote_variant) if use_smote else "",
        "accuracy": m_stack["accuracy"], "precision": m_stack["precision"],
        "recall": m_stack["recall"], "f1": m_stack["f1"],
        "roc_auc": m_stack["roc_auc"], "pr_auc": m_stack["pr_auc"], "fn": m_stack["fn"],
        "stack_fit_sec": m_stack["fit_sec"], "grid_sec": grid_sec, "run_sec": run_sec,
        "thr": float(thr), "beta": beta,
    }

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Stacking Ensemble + SMOTE ablation")
    ap.add_argument("--datasets", nargs="+", default=["tectonic", "volcanic"])
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--random-state", type=int, default=42)

    # Toggles: default = ON; bisa dimatikan dengan --no-xgb / --no-meta-grid
    ap.add_argument("--with-xgb", dest="with_xgb", action="store_true")
    ap.add_argument("--no-xgb",  dest="with_xgb", action="store_false")
    ap.set_defaults(with_xgb=True)

    ap.add_argument("--meta-grid",    dest="meta_grid", action="store_true")
    ap.add_argument("--no-meta-grid", dest="meta_grid", action="store_false")
    ap.set_defaults(meta_grid=True)

    ap.add_argument("--with-mlp", action="store_true")
    ap.add_argument("--fast", action="store_true", help="subsample training rows for fast debugging")
    ap.add_argument("--fast-n", type=int, default=5000)
    ap.add_argument("--feature-select", choices=["none","pearson","rfe"], default="none")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--ablation", action="store_true",
                    help="run non-SMOTE + SMOTE + SMOTE_TOMEK + SMOTEENN and write ablation_smote.csv")
    ap.add_argument("--use-smote", choices=["auto","yes","no"], default="auto")
    args = ap.parse_args()

    with_xgb  = args.with_xgb
    meta_grid = args.meta_grid
    fast_n = args.fast_n if args.fast else None

    _log("[Stack] Pipeline start …")
    ablation_rows: List[Dict[str, float]] = []

    for ds in args.datasets:
        if args.ablation:
            # non-SMOTE
            ablation_rows.append(run_one(
                ds, use_smote=False, smote_variant=None,
                cv=args.cv, random_state=args.random_state,
                with_xgb=with_xgb, with_mlp=args.with_mlp,
                fast_sample=fast_n, feature_select=args.feature_select,
                top_n=args.top_n, meta_grid=meta_grid,
            ))
            # semua varian SMOTE (cek file)
            for var in SMOTE_VARIANTS:
                tag = _variant_to_tag(var)
                csv_path = PROCESSED / f"{ds}_train_{tag}.csv"
                if not csv_path.exists():
                    _log(f"[WARN] {csv_path.name} not found, skip {var} for {ds}."); continue
                ablation_rows.append(run_one(
                    ds, use_smote=True, smote_variant=var,
                    cv=args.cv, random_state=args.random_state,
                    with_xgb=with_xgb, with_mlp=args.with_mlp,
                    fast_sample=fast_n, feature_select=args.feature_select,
                    top_n=args.top_n, meta_grid=meta_grid,
                ))
        else:
            if args.use_smote == "auto":
                use_sm = (PROCESSED / f"{ds}_train_{_variant_to_tag('smote')}.csv").exists() \
                         or (PROCESSED / f"{ds}_train_smote.csv").exists()
            else:
                use_sm = args.use_smote == "yes"
            var = "smote" if use_sm else None
            ablation_rows.append(run_one(
                ds, use_smote=use_sm, smote_variant=var,
                cv=args.cv, random_state=args.random_state,
                with_xgb=with_xgb, with_mlp=args.with_mlp,
                fast_sample=fast_n, feature_select=args.feature_select,
                top_n=args.top_n, meta_grid=meta_grid,
            ))

    if ablation_rows:
        _savetab(pd.DataFrame(ablation_rows), TAB / "ablation_smote.csv")
        _log(f"[Stack] ablation_smote.csv written to {TAB}")

    _log(f"[DONE] Stacking pipeline finished.\n - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}\n - DATA: {PROCESSED}")

if __name__ == "__main__":
    main()