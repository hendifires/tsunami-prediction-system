from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone

from tsunami_prediction.utils import load_csv_smart, ensure_dirs, FIG, TAB, MODELS, ART
from tsunami_prediction.eda import run_full_eda
from tsunami_prediction.preprocessing import add_engineered_features, build_preprocessor
from tsunami_prediction.base_models import get_base_estimators
from tsunami_prediction.stacking_pipeline import build_single_pipeline, build_stacking_pipeline
from tsunami_prediction.plotting import plot_roc, plot_pr, plot_cm, plot_reliability, bar_delta
from tsunami_prediction.evaluation import metrics_from_preds, mean_sd, paired_test

RANDOM_STATE = 42
TARGET = "tsu"
KFOLDS = 10

def load_dataset(name: str) -> pd.DataFrame:
    """Kini otomatis fallback ke data/raw bila processed belum ada."""
    return load_csv_smart(name)

def main():
    ensure_dirs()
    for ds in ["tectonic", "volcanic"]:
        print(f"\n=== Dataset: {ds} ===")
        df = load_dataset(ds)

def eda_stage(df: pd.DataFrame, ds_name: str):
    run_full_eda(df, ds_name, target=TARGET)

def prepare_Xy(df: pd.DataFrame):
    df = add_engineered_features(df)
    num_cols, cat_cols = detect_columns(df, target=TARGET)
    X = df[num_cols + cat_cols]
    y = df[TARGET].astype(int).values
    pre = build_preprocessor(num_cols, cat_cols)
    return X, y, pre, num_cols, cat_cols

def evaluate_models_cv(X, y, pre, estimators: dict, use_smote: bool, ds_name: str):
    """CV identik untuk NoSMOTE & SMOTE (pairing). Return: per-fold results, and per-model holdout metrics via a final split."""
    skf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for model_name, est in estimators.items():
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y[tr], y[va]
            pipe = build_single_pipeline(pre, clone(est), use_smote=use_smote)
            pipe.fit(Xtr, ytr)
            proba = pipe.predict_proba(Xva)[:, 1]
            yhat = (proba >= 0.5).astype(int)  # evaluasi default @0.5
            m = metrics_from_preds(yva, proba, yhat)
            m.update(dict(model=model_name, fold=fold, smote=use_smote, dataset=ds_name))
            records.append(m)

    # Stacking (pakai pipeline yang sama untuk anti-leakage)
    stack_pipe = build_stacking_pipeline(pre, estimators, use_smote=use_smote)
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        sp = clone(stack_pipe)
        sp.fit(Xtr, ytr)
        proba = sp.predict_proba(Xva)[:, 1]
        # pilih τ* dari validasi fold (target recall tinggi)
        tau_star, _ = choose_tau_for_recall(yva, proba, target_recall=0.95)
        yhat = (proba >= tau_star).astype(int)
        m = metrics_from_preds(yva, proba, yhat)
        m.update(dict(model="stacking", fold=fold, smote=use_smote, dataset=ds_name, tau_star=tau_star))
        records.append(m)

    return pd.DataFrame.from_records(records)

def holdout_evaluation_and_save(X, y, pre, estimators: dict, use_smote: bool, ds_name: str):
    """Split holdout untuk gambar final & simpan model/artefak untuk UI/deployment."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Simpan base (opsional)
    base_pipe = {name: build_single_pipeline(pre, est, use_smote) for name, est in estimators.items()}
    for name, p in base_pipe.items():
        p.fit(Xtr, ytr)

    # Stacking
    sp = build_stacking_pipeline(pre, estimators, use_smote)
    sp.fit(Xtr, ytr)
    proba = sp.predict_proba(Xte)[:, 1]
    tau_star, _ = choose_tau_for_recall(yte, proba, target_recall=0.95)
    yhat = (proba >= tau_star).astype(int)

    # Plots
    suffix = f"{ds_name}_{'smote' if use_smote else 'nosmote'}"
    plot_roc(yte, proba, FIG / f"{suffix}_roc.png", f"ROC — {suffix}")
    plot_pr(yte, proba, FIG / f"{suffix}_pr.png", f"PR — {suffix}")
    plot_cm(yte, yhat, FIG / f"{suffix}_cm.png", f"CM — {suffix}")
    plot_reliability(yte, proba, FIG / f"{suffix}_reliability.png", f"Reliability — {suffix}")

    # Simpan model & spec
    save_joblib(base_pipe, MODELS / f"base_{suffix}.joblib")
    save_joblib(sp, MODELS / f"stack_meta_{suffix}.joblib")
    # simpan spec kolom + tau*
    num_cols = pre.transformers_[0][2]  # dari ColumnTransformer
    cat_cols = pre.transformers_[1][2]
    spec = {"num_cols": list(num_cols), "cat_cols": list(cat_cols), "tau_star": float(tau_star)}
    save_joblib(spec, ART / f"feature_spec_{suffix}.joblib")

    # Tabel metrik holdout stacking
    m = metrics_from_preds(yte, proba, yhat)
    m.update(dict(model="stacking", dataset=ds_name, smote=use_smote, tau_star=tau_star))
    pd.DataFrame([m]).to_csv(TAB / f"holdout_{suffix}.csv", index=False)

def smote_ablation(stats_nosmote: pd.DataFrame, stats_smote: pd.DataFrame, ds_name: str):
    """Bandingkan SMOTE vs NoSMOTE untuk Stacking dan RF (primer)."""
    # Ambil per-fold untuk model 'stacking' dan 'rf'
    def pick(df, model):
        return df.query("model == @model")[["fold", "recall", "f1", "ap", "roc_auc"]].sort_values("fold")

    rows = []
    for model in ["stacking", "rf"]:
        a = pick(stats_nosmote, model).reset_index(drop=True)
        b = pick(stats_smote, model).reset_index(drop=True)
        for metric in ["recall", "f1", "ap", "roc_auc"]:
            delta = (b[metric] - a[metric]).values  # SMOTE − NoSMOTE
            test = paired_test(delta, alternative="greater")
            rows.append({
                "dataset": ds_name,
                "model": model,
                "metric": metric,
                "delta_mean": float(np.mean(delta)),
                "delta_sd": float(np.std(delta, ddof=1)),
                "p_value": test["p_value"],
                "effect": test["effect"],
                "normal": test["normal"],
                "method": test["method"]
            })
            # plot bar kecil utk delta rata2
            bar_delta({metric: float(np.mean(delta))}, FIG / f"delta_{ds_name}_{model}_{metric}.png",
                      f"Δ (SMOTE − NoSMOTE) — {ds_name} / {model} / {metric}")
    out = pd.DataFrame(rows)
    out.to_csv(TAB / f"smote_ablation_{ds_name}.csv", index=False)

def run_for_dataset(ds_name: str):
    print(f"\n=== Dataset: {ds_name} ===")
    df = load_dataset(ds_name)
    eda_stage(df, ds_name)

    X, y, pre, _, _ = prepare_Xy(df)
    estimators = get_base_estimators(RANDOM_STATE)

    # CV NoSMOTE & SMOTE (pairing sama karena random_state sama di StratifiedKFold)
    cv_no = evaluate_models_cv(X, y, pre, estimators, use_smote=False, ds_name=ds_name)
    cv_sm = evaluate_models_cv(X, y, pre, estimators, use_smote=True, ds_name=ds_name)

    # Simpan tabel per-fold & ringkasan
    cv_no.to_csv(TAB / f"cv_{ds_name}_nosmote.csv", index=False)
    cv_sm.to_csv(TAB / f"cv_{ds_name}_smote.csv", index=False)

    metr_cols = ["accuracy","precision","recall","f1","roc_auc","ap","brier"]
    summary_no = mean_sd(cv_no, by_cols=["dataset","model","smote"], metric_cols=metr_cols)
    summary_sm = mean_sd(cv_sm, by_cols=["dataset","model","smote"], metric_cols=metr_cols)
    summary = pd.concat([summary_no, summary_sm]).reset_index(drop=True)
    summary.to_csv(TAB / f"summary_{ds_name}.csv", index=False)

    # Holdout test & simpan model/artefak untuk UI
    holdout_evaluation_and_save(X, y, pre, estimators, use_smote=False, ds_name=ds_name)
    holdout_evaluation_and_save(X, y, pre, estimators, use_smote=True, ds_name=ds_name)

    # SMOTE ablation (paired test)
    smote_ablation(cv_no, cv_sm, ds_name)

def main():
    for ds in ["tectonic", "volcanic"]:
        run_for_dataset(ds)

if __name__ == "__main__":
    main()