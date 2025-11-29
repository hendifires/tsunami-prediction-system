from __future__ import annotations
# cSpell:ignore nosmote tomek smoteenn utama dari sekaligus membandingkan

"""
Compare multiple stacking runs (nosmote / smote / smote_tomek / smote_enn)
dari output stacking_pipeline.py (cv=5), dan sekaligus membandingkan
kinerja base learners tertentu terhadap meta-learner Logistic Regression
di dalam stacking ensemble.

Outputs utama:

1) Ringkasan stacking (per dataset & varian SMOTE)
   - reports/tables/compare_runs.csv
   - reports/tables/compare_runs.md
   - reports/figures/compare_stacking_<metric>.png
       (untuk SEMUA metric di _METRICS_MAIN, grouped bar antar varian
        dan antar domain tectonic/volcanic)

   Setiap baris = (dataset, variant), berisi metrik dari model
   'stacking_lr_meta' (atau fallback baris dengan F1 tertinggi).

2) Efek SMOTE terhadap baseline 'nosmote' (khusus stacking)
   - reports/tables/smote_effects_by_variant.csv
   - reports/figures/smote_effects_f1.png

   Tambahan:
   - reports/tables/smote_paired_stats.csv
   - reports/tables/smote_paired_stats.md
     berisi ringkasan paired t-test & Cohen's d:
     nosmote vs (smote / smote_tomek / smote_enn) untuk tiap metric.

3) Perbandingan base learners vs meta-learner (stacking_lr_meta)
   Base learner yang dibandingkan:
       RF, GB, DT, NB, KNN  vs  stacking_lr_meta (meta-learner Logistic Regression)

   - reports/tables/base_vs_stack.csv
        format long: satu baris per (dataset, variant, model)
        kolom:
        accuracy, precision, recall, f1, f1_macro, roc_auc, pr_auc, brier,
        tn, fp, fn, tp, model, fit_sec, grid_sec
   - reports/tables/base_vs_stack.md
        versi markdown (siap copas ke tesis)
   - reports/figures/base_vs_stack_<dataset>_<metric>.png
        grouped bar per dataset: x = model, hue = varian sampling,
        untuk SEMUA metric di _METRICS_MAIN.

Catatan penting:
- Script ini mengasumsikan adanya file metrics:
    reports/tables/<dataset>_stack_<variant>_cv5_metrics.csv
  yang dibuat oleh stacking_pipeline.py.
- Untuk kompatibilitas ke belakang, jika file tersebut tidak ada,
  script akan mencoba membaca:
    reports/tables/<dataset>_stack_<variant>_metrics.csv
"""

import argparse
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
for p in (TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Constants ----------------
VARIANT_TAGS: List[str] = ["nosmote", "smote", "smote_tomek", "smote_enn"]
DATASETS: List[str] = ["tectonic", "volcanic"]

# Metrik utama (sinkron dengan stacking_pipeline._metrics)
_METRIC_COLS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "roc_auc",
    "pr_auc",
    "brier",
    "tn",
    "fp",
    "fn",
    "tp",
    "fit_sec",
    "grid_sec",
]

# Metrik utama untuk ringkasan & plotting
_METRICS_MAIN: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "roc_auc",
    "pr_auc",
    "brier",
]

# Model base yang akan dibandingkan dengan meta-learner (stacking)
BASE_FOR_COMPARE: List[str] = ["rf", "gb", "dt", "nb", "knn"]
STACK_MODEL_NAME = "stacking_lr_meta"


# ---------------- Utils ----------------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_md_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------- Core helpers (stacking summary) ----------------
def _pick_row(df: pd.DataFrame) -> pd.Series:
    """
    Ambil baris representatif dari metrics_df satu run:

    1) Jika ada model 'stacking_lr_meta' → ambil baris itu.
    2) Jika tidak ada → pilih baris dengan F1 tertinggi.
    3) Jika tetap tidak ada info F1 → pakai baris pertama.
    """
    if "model" in df.columns:
        sel = df[df["model"] == STACK_MODEL_NAME]
        if not sel.empty:
            return sel.iloc[0]

    df2 = df.copy()
    if "f1" in df2.columns:
        df2["f1"] = pd.to_numeric(df2["f1"], errors="coerce")
        df2 = df2.sort_values("f1", ascending=False)
        return df2.iloc[0]

    return df.iloc[0]


def _find_metrics_path(dataset: str, variant: str) -> Optional[Path]:
    """
    Cari file metrics untuk (dataset, variant) dengan urutan prioritas:
    1) <dataset>_stack_<variant>_cv5_metrics.csv
    2) <dataset>_stack_<variant>_metrics.csv  (fallback kompatibilitas)
    """
    candidates = [
        TAB / f"{dataset}_stack_{variant}_cv5_metrics.csv",
        TAB / f"{dataset}_stack_{variant}_metrics.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def _read_metrics(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _log(f"[WARN] failed to read {path.name}: {exc}")
        return None

    if df.empty:
        _log(f"[WARN] metrics file is empty: {path.name}")
        return None

    return df


def collect_stacking_summary(
    datasets: List[str],
    variants: List[str],
) -> pd.DataFrame:
    """
    Kumpulkan satu baris stacking per (dataset, variant).

    Sumber: metrics.csv masing-masing run.
    """
    rows: List[Dict[str, object]] = []
    for ds in datasets:
        for var in variants:
            path = _find_metrics_path(ds, var)
            if path is None:
                _log(f"[SKIP] metrics not found for {ds} / {var}")
                continue

            df = _read_metrics(path)
            if df is None:
                continue

            row = _pick_row(df)
            out: Dict[str, object] = {
                "dataset": ds,
                "variant": var,
            }
            for col in _METRIC_COLS:
                out[col] = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            rows.append(out)

    return pd.DataFrame(rows)


def _order_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns:
        out["dataset"] = pd.Categorical(
            out["dataset"],
            categories=DATASETS,
            ordered=True,
        )
    if "variant" in out.columns:
        out["variant"] = pd.Categorical(
            out["variant"],
            categories=VARIANT_TAGS,
            ordered=True,
        )
    return out.sort_values(["dataset", "variant"])


def _plot_bar(df: pd.DataFrame, metric: str, path: Path) -> None:
    """
    Grouped bar: x = variant, hue = dataset (untuk stacking summary)
    untuk satu metric tertentu.
    """
    if metric not in df.columns or df.empty:
        _log(f"[INFO] cannot plot: metric '{metric}' missing or df empty.")
        return

    df_plot = _order_categories(df)
    present_variants = [
        v for v in VARIANT_TAGS if v in df_plot["variant"].astype(str).unique()
    ]
    if not present_variants:
        _log("[INFO] no variants present for plotting.")
        return

    x = np.arange(len(present_variants))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    for i, ds in enumerate(DATASETS):
        sub = df_plot[df_plot["dataset"] == ds]
        if sub.empty:
            continue

        vals: List[float] = []
        for v in present_variants:
            sub_v = sub[sub["variant"] == v]
            if sub_v.empty or pd.isna(sub_v[metric].values[0]):
                vals.append(np.nan)
            else:
                vals.append(float(sub_v[metric].values[0]))

        plt.bar(x + (i - 0.5) * width, vals, width, label=ds)

    plt.xticks(x, present_variants)
    plt.ylabel(metric)
    plt.title(f"Stacking ({STACK_MODEL_NAME}) by {metric}")
    plt.legend()
    _savefig(path)


# ---------- SMOTE effect helpers (stacking only) ----------
def compute_smote_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung efek SMOTE dibanding baseline 'nosmote' untuk setiap dataset & varian.

    Output kolom:
    - dataset, variant, baseline_variant
    - <metric>_base, <metric>_variant, <metric>_delta
      untuk setiap metric utama (_METRICS_MAIN).
    """
    if df.empty:
        return pd.DataFrame()

    df_ord = _order_categories(df)

    base = df_ord[df_ord["variant"] == "nosmote"].set_index("dataset")
    if base.empty:
        _log("[SMOTE] No baseline 'nosmote' rows found; cannot compute effects.")
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for ds in df_ord["dataset"].unique():
        if ds not in base.index:
            continue
        base_row = base.loc[ds]

        sub = df_ord[
            (df_ord["dataset"] == ds) & (df_ord["variant"] != "nosmote")
        ]
        for _, r in sub.iterrows():
            out: Dict[str, object] = {
                "dataset": ds,
                "variant": r["variant"],
                "baseline_variant": "nosmote",
            }
            for metric in _METRICS_MAIN:
                base_val = float(base_row.get(metric, np.nan))
                var_val = float(r.get(metric, np.nan))
                if np.isfinite(base_val) and np.isfinite(var_val):
                    delta = var_val - base_val
                else:
                    delta = np.nan
                out[f"{metric}_base"] = base_val
                out[f"{metric}_variant"] = var_val
                out[f"{metric}_delta"] = delta
            rows.append(out)

    return pd.DataFrame(rows)


def _plot_smote_effects_f1(df_eff: pd.DataFrame, path: Path) -> None:
    """Plot delta F1 (varian - nosmote) per dataset & varian."""
    if df_eff.empty or "f1_delta" not in df_eff.columns:
        _log("[SMOTE] No data to plot F1 effects.")
        return

    df_plot = df_eff.copy()
    df_plot["dataset"] = pd.Categorical(
        df_plot["dataset"],
        categories=DATASETS,
        ordered=True,
    )
    df_plot["variant"] = pd.Categorical(
        df_plot["variant"],
        categories=[v for v in VARIANT_TAGS if v != "nosmote"],
        ordered=True,
    )
    df_plot = df_plot.sort_values(["dataset", "variant"])

    variants = list(df_plot["variant"].cat.categories)
    x = np.arange(len(variants))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    for i, ds in enumerate(DATASETS):
        sub = df_plot[df_plot["dataset"] == ds]
        if sub.empty:
            continue

        vals: List[float] = []
        for v in variants:
            sub_v = sub[sub["variant"] == v]
            if sub_v.empty:
                vals.append(np.nan)
            else:
                vals.append(float(sub_v["f1_delta"].iloc[0]))

        plt.bar(x + (i - 0.5) * width, vals, width, label=ds)

    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, variants)
    plt.ylabel("Δ F1 (variant − nosmote)")
    plt.title(f"Effect of SMOTE variants on F1 ({STACK_MODEL_NAME})")
    plt.legend()
    _savefig(path)


def compute_smote_paired_stats(df_eff: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung statistik paired t-test dan Cohen's d untuk ablation SMOTE.

    Input: df_eff dari compute_smote_effects (berisi *_base, *_variant, *_delta)
    Output: satu baris per (variant, metric):
        variant, metric, n,
        mean_base, mean_variant, mean_delta, sd_delta,
        cohen_d, t_stat, p_value
    """
    if df_eff.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    variants = sorted(df_eff["variant"].unique())
    for var in variants:
        sub = df_eff[df_eff["variant"] == var]
        if sub.empty:
            continue

        for metric in _METRICS_MAIN:
            base_col = f"{metric}_base"
            var_col = f"{metric}_variant"

            if base_col not in sub.columns or var_col not in sub.columns:
                continue

            base_vals = pd.to_numeric(sub[base_col], errors="coerce")
            var_vals = pd.to_numeric(sub[var_col], errors="coerce")
            mask = base_vals.notna() & var_vals.notna()

            base_arr = base_vals[mask].to_numpy()
            var_arr = var_vals[mask].to_numpy()
            if base_arr.size == 0:
                continue

            diff = var_arr - base_arr
            n = diff.size
            mean_base = float(np.mean(base_arr))
            mean_var = float(np.mean(var_arr))
            mean_delta = float(np.mean(diff))

            if n > 1:
                sd_delta = float(np.std(diff, ddof=1))
                cohen_d = mean_delta / sd_delta if sd_delta > 0 else np.nan
            else:
                sd_delta = np.nan
                cohen_d = np.nan

            t_stat = np.nan
            p_value = np.nan
            if n > 1:
                with suppress(Exception):
                    from scipy import stats

                    t_res = stats.ttest_rel(var_arr, base_arr, nan_policy="omit")
                    t_stat = float(t_res.statistic)
                    p_value = float(t_res.pvalue)

            rows.append(
                {
                    "variant": var,
                    "metric": metric,
                    "n": int(n),
                    "mean_base": mean_base,
                    "mean_variant": mean_var,
                    "mean_delta": mean_delta,
                    "sd_delta": sd_delta,
                    "cohen_d": cohen_d,
                    "t_stat": t_stat,
                    "p_value": p_value,
                }
            )

    return pd.DataFrame(rows)


def handle_smote_effects(df: pd.DataFrame) -> None:
    """
    Wrapper terpisah untuk menghitung & menyimpan efek SMOTE
    dan statistik paired t-test + Cohen's d.
    """
    eff_df = compute_smote_effects(df)
    if eff_df.empty:
        _log("[Compare] SMOTE effects could not be computed (missing baseline nosmote?).")
        return

    eff_csv = TAB / "smote_effects_by_variant.csv"
    _save_csv(eff_df, eff_csv)
    _log(f"[Compare] Saved SMOTE effects: {eff_csv}")

    eff_fig = FIG / "smote_effects_f1.png"
    _plot_smote_effects_f1(eff_df, eff_fig)
    _log(f"[Compare] Saved SMOTE F1 effects: {eff_fig}")

    # ---- Paired t-test & Cohen's d ----
    stats_df = compute_smote_paired_stats(eff_df)
    if stats_df.empty:
        _log("[Compare] Paired stats not computed (insufficient data).")
        return

    stats_csv = TAB / "smote_paired_stats.csv"
    stats_md = TAB / "smote_paired_stats.md"
    _save_csv(stats_df, stats_csv)
    _save_md_table(stats_df, stats_md)
    _log(f"[Compare] Saved SMOTE paired stats CSV: {stats_csv}")
    _log(f"[Compare] Saved SMOTE paired stats MD : {stats_md}")


# ---------- Base vs Stacking helpers ----------
def collect_base_vs_stack(
    datasets: List[str],
    variants: List[str],
) -> pd.DataFrame:
    """
    Ambil metrik untuk base learners tertentu (RF, GB, DT, NB, KNN)
    dan stacking_lr_meta (meta-learner Logistic Regression),
    untuk setiap (dataset, variant).

    Output long-format:
    - dataset, variant, model
    - accuracy, precision, recall, f1, f1_macro, roc_auc, pr_auc, brier,
      tn, fp, fn, tp, fit_sec, grid_sec
    """
    rows: List[Dict[str, object]] = []

    models_interest = set(BASE_FOR_COMPARE + [STACK_MODEL_NAME])

    for ds in datasets:
        for var in variants:
            path = _find_metrics_path(ds, var)
            if path is None:
                _log(f"[SKIP] metrics not found for base-vs-stack {ds} / {var}")
                continue

            df = _read_metrics(path)
            if df is None or "model" not in df.columns:
                _log(f"[SKIP] no 'model' column in {path.name}")
                continue

            df_loc = df.copy()
            df_loc["model"] = df_loc["model"].astype(str)
            df_loc = df_loc[df_loc["model"].isin(models_interest)]
            if df_loc.empty:
                _log(f"[INFO] no selected models in {path.name}")
                continue

            for _, r in df_loc.iterrows():
                model_name = str(r["model"])
                out: Dict[str, object] = {
                    "dataset": ds,
                    "variant": var,
                    "model": model_name,
                }
                # isi semua kolom metrik & runtime yang diminta
                for col in _METRIC_COLS:
                    out[col] = pd.to_numeric(r.get(col, np.nan), errors="coerce")
                rows.append(out)

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    df_out["dataset"] = pd.Categorical(
        df_out["dataset"],
        categories=DATASETS,
        ordered=True,
    )
    df_out["variant"] = pd.Categorical(
        df_out["variant"],
        categories=VARIANT_TAGS,
        ordered=True,
    )
    model_order = BASE_FOR_COMPARE + [STACK_MODEL_NAME]
    df_out["model"] = pd.Categorical(
        df_out["model"],
        categories=model_order,
        ordered=True,
    )
    return df_out.sort_values(["dataset", "variant", "model"])


def _plot_base_vs_stack_per_dataset(
    df_bvs: pd.DataFrame,
    dataset: str,
    metric: str,
    path: Path,
) -> None:
    """
    Grouped bar untuk satu dataset & satu metric:
    x = model (RF, GB, DT, NB, KNN, stacking_lr_meta)
    hue = varian sampling (nosmote, smote, smote_tomek, smote_enn)
    """
    if df_bvs.empty or metric not in df_bvs.columns:
        _log(f"[BaseVsStack] Nothing to plot for {dataset} / metric '{metric}'.")
        return

    df_ds = df_bvs[df_bvs["dataset"] == dataset].copy()
    if df_ds.empty:
        _log(f"[BaseVsStack] No data for dataset {dataset}.")
        return

    models = list(df_ds["model"].cat.categories)
    variants = [v for v in VARIANT_TAGS if v in df_ds["variant"].cat.categories]

    x = np.arange(len(models))
    if not variants:
        _log(f"[BaseVsStack] No variants for plotting in dataset {dataset}.")
        return

    width = 0.8 / max(len(variants), 1)

    plt.figure(figsize=(9.0, 4.8))
    for i, var in enumerate(variants):
        sub = df_ds[df_ds["variant"] == var]
        vals: List[float] = []
        for m in models:
            sub_m = sub[sub["model"] == m]
            if sub_m.empty or pd.isna(sub_m[metric].values[0]):
                vals.append(np.nan)
            else:
                vals.append(float(sub_m[metric].values[0]))

        offset = (i - (len(variants) - 1) / 2) * width
        plt.bar(x + offset, vals, width, label=str(var))

    plt.xticks(x, models)
    plt.ylabel(metric)
    plt.title(f"{dataset} – Base learners vs {STACK_MODEL_NAME} ({metric})")
    plt.legend(title="variant")
    _savefig(path)


# ---------------- CLI ----------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Aggregate & compare stacking (SMOTE variants) + "
            "base learners (RF, GB, DT, NB, KNN) vs meta-learner Logistic Regression"
        )
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--variants", nargs="+", default=VARIANT_TAGS, choices=VARIANT_TAGS)
    ap.add_argument(
        "--sort-metric",
        default="accuracy",
        choices=_METRICS_MAIN,
        help="metric untuk mengurutkan compare_runs.csv di dalam tiap dataset",
    )
    ap.add_argument(
        "--out-csv",
        default=str(TAB / "compare_runs.csv"),
        help="output CSV ringkasan stacking",
    )
    ap.add_argument(
        "--out-md",
        default=str(TAB / "compare_runs.md"),
        help="output Markdown ringkasan stacking",
    )
    ap.add_argument(
        "--plot-metric",
        default="all",
        choices=_METRICS_MAIN + ["none", "all"],
        help=(
            "metric untuk grouped bar stacking. "
            "'all' = buat plot untuk semua metric, 'none' = tidak digambar"
        ),
    )
    ap.add_argument(
        "--base-metric",
        default="all",
        choices=_METRICS_MAIN + ["all"],
        help=(
            "metric utama untuk plot base vs stacking. "
            "'all' = buat plot untuk semua metric."
        ),
    )
    args = ap.parse_args()

    # --------- Stacking summary (per dataset & variant) ---------
    _log("[Compare] Collecting stacking summaries …")
    df_stack = collect_stacking_summary(args.datasets, args.variants)
    if df_stack.empty:
        _log("[Compare] Nothing to compare (no stacking metrics files found).")
        return

    df_stack = _order_categories(df_stack)
    if args.sort_metric in df_stack.columns:
        df_stack = df_stack.sort_values(
            ["dataset", args.sort_metric],
            ascending=[True, False],
        )

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    _save_csv(df_stack, out_csv)
    _save_md_table(df_stack, out_md)
    _log(f"[Compare] Saved stacking summary CSV: {out_csv}")
    _log(f"[Compare] Saved stacking summary MD : {out_md}")

    # --- Plot stacking summary untuk semua metric (atau satu metric) ---
    if args.plot_metric != "none":
        metrics_to_plot = (
            _METRICS_MAIN if args.plot_metric == "all" else [args.plot_metric]
        )
        for m in metrics_to_plot:
            if m not in df_stack.columns:
                continue
            fig_path = FIG / f"compare_stacking_{m}.png"
            _plot_bar(df_stack, m, fig_path)
            _log(f"[Compare] Saved stacking figure: {fig_path}")

    # --------- SMOTE vs nosmote (stacking only) + paired stats ---------
    handle_smote_effects(df_stack)

    # --------- Base vs meta-learner (RF, GB, DT, NB, KNN vs stacking_lr_meta) ---------
    _log("[Compare] Collecting base vs stacking …")
    df_bvs = collect_base_vs_stack(args.datasets, args.variants)
    if df_bvs.empty:
        _log("[Compare] No data for base-vs-stacking comparison.")
        return

    base_csv = TAB / "base_vs_stack.csv"
    base_md = TAB / "base_vs_stack.md"
    _save_csv(df_bvs, base_csv)
    _save_md_table(df_bvs, base_md)
    _log(f"[Compare] Saved base vs stacking CSV: {base_csv}")
    _log(f"[Compare] Saved base vs stacking MD : {base_md}")

    # --- Plot base vs stacking untuk setiap dataset & setiap metric utama ---
    metrics_for_base = _METRICS_MAIN if args.base_metric == "all" else [args.base_metric]
    for ds in args.datasets:
        for m in metrics_for_base:
            if m not in df_bvs.columns:
                continue
            fig_bvs = FIG / f"base_vs_stack_{ds}_{m}.png"
            _plot_base_vs_stack_per_dataset(df_bvs, ds, m, fig_bvs)
            _log(f"[Compare] Saved base-vs-stacking figure: {fig_bvs}")


if __name__ == "__main__":
    main()