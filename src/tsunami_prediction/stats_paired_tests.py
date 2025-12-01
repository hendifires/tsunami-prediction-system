from __future__ import annotations
# cSpell:ignore nosmote smoteenn tomek cohen

"""
Paired statistical tests (paired t-test + Cohen's d) untuk pipeline stacking terbaru.

Fokus:
1) Perbandingan base learners vs stacking meta-learner (Logistic Regression)
   pada setiap dataset & varian sampling (nosmote / SMOTE / SMOTE-Tomek / SMOTE-ENN).
2) Ablation SMOTE: nosmote vs varian SMOTE KHUSUS untuk model stacking_lr_meta.

Input utama (per-fold metrics):

    reports/tables/<dataset>_stack_<variant>_cv<cv>_cvfolds.csv

File ini dihasilkan oleh stacking_pipeline.py versi terbaru, dengan kolom:

    dataset, variant, model, fold,
    accuracy, precision, recall, f1, roc_auc, pr_auc

Output:

  1) Base vs Stacking
     - reports/tables/paired_tests_base_vs_stack.csv
     - reports/tables/paired_tests_base_vs_stack.md

  2) SMOTE ablation (nosmote vs varian SMOTE) untuk stacking_lr_meta:
     - reports/tables/paired_tests_smote_ablation.csv
     - reports/tables/paired_tests_smote_ablation.md
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats  # pastikan sudah terinstall

from tsunami_prediction.utils import TAB, REPORTS, FIG, ensure_dirs


# ---------- KONSTANTA & PATH ----------
# pastikan struktur folder project sudah ada
ensure_dirs()

# default dataset mengikuti model utama: tektonik & vulkanik
# (jika nanti ada model global "events", bisa dipanggil via argumen --datasets events)
DATASETS_DEFAULT: List[str] = ["tectonic", "volcanic"]

# varian balancing sesuai pipeline SMOTE terbaru
VARIANTS_DEFAULT: List[str] = ["nosmote", "smote", "smote_tomek", "smote_enn"]

# metrik yang disimpan stacking_pipeline.py
METRICS_DEFAULT: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

# base learners yang dikomparasikan terhadap stacking
# (harus sama dengan nama model yang dipakai di stacking_pipeline.py)
BASE_MODELS_FOR_COMPARE: List[str] = ["rf", "gb", "dt", "nb", "knn"]
STACK_MODEL_NAME: str = "stacking_lr_meta"


# ---------- UTIL ----------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_md(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))


def _cohen_d_paired(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Cohen's d untuk desain berpasangan: d = mean(diff) / sd(diff),
    dengan diff = b - a (pasangan).

    Juga mengembalikan:
        - t-statistik
        - p-value (two-sided)
        - n (jumlah pasangan)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a dan b harus 1D dan memiliki shape yang sama.")

    diff = b - a
    n = diff.size
    if n < 2:
        return {
            "n": float(n),
            "mean_diff": np.nan,
            "t": np.nan,
            "p": np.nan,
            "d": np.nan,
        }

    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))

    if sd_diff == 0.0:
        # tidak ada variasi antarfold → t dan d tidak terdefinisi
        return {
            "n": float(n),
            "mean_diff": mean_diff,
            "t": np.nan,
            "p": np.nan,
            "d": np.nan,
        }

    # t-test berpasangan
    t_stat = mean_diff / (sd_diff / np.sqrt(n))
    dfree = n - 1
    p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=dfree))

    d = mean_diff / sd_diff
    return {
        "n": float(n),
        "mean_diff": mean_diff,
        "t": float(t_stat),
        "p": p_val,
        "d": float(d),
    }


def _load_cvfolds(dataset: str, variant: str, cv: int) -> Optional[pd.DataFrame]:
    """
    Membaca file per-fold metrics:
        reports/tables/<dataset>_stack_<variant>_cv<cv>_cvfolds.csv

    Contoh:
        tectonic_stack_smote_cv5_cvfolds.csv
        volcanic_stack_nosmote_cv5_cvfolds.csv
    """
    path = TAB / f"{dataset}_stack_{variant}_cv{cv}_cvfolds.csv"
    if not path.exists():
        _log(f"[SKIP] cvfolds file not found: {path.name}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        _log(f"[WARN] failed to read {path.name}: {exc}")
        return None

    if df.empty:
        _log(f"[WARN] empty cvfolds file: {path.name}")
        return None

    return df


# ---------- 1) BASE vs STACKING ----------
def compute_base_vs_stack_tests(
    datasets: List[str],
    variants: List[str],
    metrics: List[str],
    cv: int,
) -> pd.DataFrame:
    """
    Paired t-test dan Cohen's d:
        base_model (rf, gb, dt, nb, knn) vs stacking_lr_meta

    Pairing menggunakan fold yang sama di dalam (dataset, variant).
    """
    rows: List[Dict[str, object]] = []

    for ds in datasets:
        for var in variants:
            df_cv = _load_cvfolds(ds, var, cv=cv)
            if df_cv is None:
                continue

            if not {"model", "fold"}.issubset(df_cv.columns):
                _log(f"[SKIP] 'model' atau 'fold' tidak ada di {ds}/{var}")
                continue

            # subset stacking sebagai model referensi
            df_stack = df_cv[df_cv["model"] == STACK_MODEL_NAME]
            if df_stack.empty:
                _log(f"[INFO] No stacking model in {ds}/{var}, skip base-vs-stack.")
                continue

            for base in BASE_MODELS_FOR_COMPARE:
                df_base = df_cv[df_cv["model"] == base]
                if df_base.empty:
                    _log(f"[INFO] base model '{base}' missing in {ds}/{var}, skip.")
                    continue

                # align berdasarkan fold
                merged = pd.merge(
                    df_base,
                    df_stack,
                    on="fold",
                    suffixes=("_base", "_stack"),
                )
                if merged.empty:
                    _log(
                        f"[INFO] No overlapping folds for {ds}/{var} "
                        f"base={base} vs stacking."
                    )
                    continue

                for metric in metrics:
                    col_base = f"{metric}_base"
                    col_stack = f"{metric}_stack"
                    if col_base not in merged.columns or col_stack not in merged.columns:
                        _log(
                            f"[INFO] metric '{metric}' missing for {ds}/{var}, "
                            f"base={base}"
                        )
                        continue

                    a = merged[col_base].to_numpy(dtype=float)
                    b = merged[col_stack].to_numpy(dtype=float)

                    stats_dict = _cohen_d_paired(a, b)  # diff = stack - base
                    n = stats_dict["n"]
                    if n < 2:
                        _log(
                            f"[INFO] too few folds (n={n}) for {ds}/{var}, "
                            f"base={base}, metric={metric}"
                        )
                        continue

                    row: Dict[str, object] = {
                        "dataset": ds,
                        "variant": var,
                        "metric": metric,
                        "model_base": base,
                        "model_ref": STACK_MODEL_NAME,
                        "n_folds": n,
                        "mean_base": float(np.mean(a)),
                        "mean_ref": float(np.mean(b)),
                        # diff = ref - base = stacking - base
                        "mean_diff_ref_minus_base": stats_dict["mean_diff"],
                        "t_stat": stats_dict["t"],
                        "p_value": stats_dict["p"],
                        "cohen_d": stats_dict["d"],
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


# ---------- 2) SMOTE ABLATION (nosmote vs varian SMOTE) ----------
def compute_smote_ablation_tests(
    datasets: List[str],
    variants: List[str],
    metrics: List[str],
    cv: int,
) -> pd.DataFrame:
    """
    Paired t-test & Cohen's d untuk efek SMOTE terhadap baseline 'nosmote'
    KHUSUS model stacking_lr_meta.

    Pairing: fold yang sama (1..cv) antara:
        (dataset, model=stacking_lr_meta, variant='nosmote')
    dan
        (dataset, model=stacking_lr_meta, variant=<smote variant>)
    """
    rows: List[Dict[str, object]] = []

    if "nosmote" not in variants:
        _log("[WARN] 'nosmote' tidak ada di variants; SMOTE ablation tidak bisa dihitung.")
        return pd.DataFrame()

    smote_variants = [v for v in variants if v != "nosmote"]
    if not smote_variants:
        _log("[INFO] tidak ada varian SMOTE selain 'nosmote'; skip SMOTE ablation.")
        return pd.DataFrame()

    for ds in datasets:
        df_nosm = _load_cvfolds(ds, "nosmote", cv=cv)
        if df_nosm is None:
            _log(f"[SMOTE] baseline nosmote missing for dataset {ds}, skip.")
            continue

        df_nosm = df_nosm[df_nosm["model"] == STACK_MODEL_NAME]
        if df_nosm.empty:
            _log(f"[SMOTE] nosmote stacking missing for {ds}, skip.")
            continue

        for var in smote_variants:
            df_var = _load_cvfolds(ds, var, cv=cv)
            if df_var is None:
                _log(f"[SMOTE] variant {var} missing for dataset {ds}, skip.")
                continue

            df_var = df_var[df_var["model"] == STACK_MODEL_NAME]
            if df_var.empty:
                _log(f"[SMOTE] stacking missing for {ds}/{var}, skip.")
                continue

            merged = pd.merge(
                df_nosm,
                df_var,
                on="fold",
                suffixes=("_nosm", "_var"),
            )
            if merged.empty:
                _log(
                    f"[SMOTE] No overlapping folds for {ds}: nosmote vs {var}, skip."
                )
                continue

            for metric in metrics:
                col_base = f"{metric}_nosm"
                col_var = f"{metric}_var"
                if col_base not in merged.columns or col_var not in merged.columns:
                    _log(
                        f"[SMOTE] metric '{metric}' missing for {ds}: "
                        f"nosmote vs {var}"
                    )
                    continue

                a = merged[col_base].to_numpy(dtype=float)  # baseline nosmote
                b = merged[col_var].to_numpy(dtype=float)   # SMOTE variant

                stats_dict = _cohen_d_paired(a, b)  # diff = variant - baseline
                n = stats_dict["n"]
                if n < 2:
                    _log(
                        f"[SMOTE] too few folds (n={n}) for {ds}: nosmote vs {var}, "
                        f"metric={metric}"
                    )
                    continue

                row: Dict[str, object] = {
                    "dataset": ds,
                    "model": STACK_MODEL_NAME,
                    "baseline_variant": "nosmote",
                    "variant": var,
                    "metric": metric,
                    "n_folds": n,
                    "mean_baseline": float(np.mean(a)),
                    "mean_variant": float(np.mean(b)),
                    # diff = variant - baseline (SMOTE effect)
                    "mean_diff_variant_minus_baseline": stats_dict["mean_diff"],
                    "t_stat": stats_dict["t"],
                    "p_value": stats_dict["p"],
                    "cohen_d": stats_dict["d"],
                }
                rows.append(row)

    return pd.DataFrame(rows)


# ---------- MAIN ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Paired t-test & Cohen's d untuk base vs stacking dan SMOTE ablation "
            "menggunakan metrics per-fold (cv=5)."
        )
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS_DEFAULT,
        help="list nama dataset (default: tectonic volcanic)",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=VARIANTS_DEFAULT,
        help="list varian sampling (default: nosmote smote smote_tomek smote_enn)",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=METRICS_DEFAULT,
        help="list metrik (default: accuracy precision recall f1 roc_auc pr_auc)",
    )
    ap.add_argument(
        "--cv",
        type=int,
        default=5,
        help="jumlah fold CV yang dipakai stacking_pipeline (default=5)",
    )
    args = ap.parse_args()

    datasets = args.datasets
    variants = args.variants
    metrics = args.metrics
    cv = args.cv

    _log("=== Paired tests: Base vs Stacking ===")
    df_bvs = compute_base_vs_stack_tests(datasets, variants, metrics, cv=cv)
    if df_bvs.empty:
        _log("[RESULT] Tidak ada hasil untuk base vs stacking.")
    else:
        df_bvs = df_bvs.sort_values(
            ["dataset", "variant", "metric", "model_base"]
        ).reset_index(drop=True)

        out_csv_bvs = TAB / "paired_tests_base_vs_stack.csv"
        out_md_bvs = TAB / "paired_tests_base_vs_stack.md"
        _save_csv(df_bvs, out_csv_bvs)
        _save_md(df_bvs, out_md_bvs)
        _log(f"[SAVE] Base vs stacking CSV: {out_csv_bvs}")
        _log(f"[SAVE] Base vs stacking MD : {out_md_bvs}")

    _log("\n=== Paired tests: SMOTE ablation (nosmote vs variants) for stacking ===")
    df_smote = compute_smote_ablation_tests(datasets, variants, metrics, cv=cv)
    if df_smote.empty:
        _log("[RESULT] Tidak ada hasil untuk SMOTE ablation.")
    else:
        df_smote = df_smote.sort_values(
            ["dataset", "variant", "metric"]
        ).reset_index(drop=True)

        out_csv_sm = TAB / "paired_tests_smote_ablation.csv"
        out_md_sm = TAB / "paired_tests_smote_ablation.md"
        _save_csv(df_smote, out_csv_sm)
        _save_md(df_smote, out_md_sm)
        _log(f"[SAVE] SMOTE ablation CSV: {out_csv_sm}")
        _log(f"[SAVE] SMOTE ablation MD : {out_md_sm}")

    _log("\n[DONE] Paired tests selesai. Lihat hasil di folder reports/tables.")


if __name__ == "__main__":
    main()