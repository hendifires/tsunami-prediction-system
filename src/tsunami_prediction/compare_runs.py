from __future__ import annotations
# cSpell:ignore nosmote tomek smoteenn utama Tambahan

"""
Compare multiple stacking runs (nosmote / smote / smote_tomek / smote_enn)
by collecting metrics CSVs emitted by stacking_pipeline.py.

Outputs utama:
- reports/tables/compare_runs.csv
    Satu baris per (dataset, variant), berisi metrik stacking ensemble.
- reports/tables/compare_runs.md
    Versi tabel dalam format markdown (siap copas ke tesis).
- reports/figures/compare_<metric>.png
    Grouped bar chart per metriks (opsional, dipilih via CLI).

Selection rule per metrics file:
1) Utamakan baris dengan model == 'stacking_lr_meta'
2) Jika tidak ada, fallback ke baris dengan F1 tertinggi
3) Jika tetap tidak bisa, pakai baris pertama

Tambahan untuk analisis SMOTE:
- reports/tables/smote_effects_by_variant.csv
    Efek SMOTE per varian terhadap baseline nosmote:
    berisi nilai baseline & varian serta delta untuk accuracy,
    precision, recall, f1, roc_auc, pr_auc.
- reports/figures/smote_effects_f1.png
    Bar plot delta F1 per varian & dataset.

Catatan:
- File runtime per base learner & stacking (semua varian)
  sudah disiapkan oleh stacking_pipeline.py dalam
  reports/tables/runtime_summary.csv, sehingga script ini
  fokus hanya pada komparasi metrik kinerja stacking.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Matplotlib only for chart; safe for headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------- PATHS (only what's used) ----------------
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
for p in [TAB, FIG]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Constants ----------------
VARIANT_TAGS = ["nosmote", "smote", "smote_tomek", "smote_enn"]
DATASETS = ["tectonic", "volcanic"]

# Expected metric columns (sinkron dengan stacking_pipeline.py)
_METRIC_COLS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "fit_sec",
    "grid_sec",  # waktu meta-grid untuk stacking (NaN/0 untuk model lain)
]

_METRICS_MAIN = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

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


# ---------------- Core helpers ----------------
def _pick_row(df: pd.DataFrame) -> pd.Series:
    """
    Ambil baris representatif dari metrics_df satu run:

    1) Jika ada model 'stacking_lr_meta' → ambil baris itu.
    2) Jika tidak ada → pilih baris dengan F1 tertinggi.
    3) Jika tetap tidak ada info F1 → pakai baris pertama.
    """
    if "model" in df.columns:
        sel = df[df["model"] == "stacking_lr_meta"]
        if not sel.empty:
            return sel.iloc[0]

    df2 = df.copy()
    if "f1" in df2.columns:
        df2["f1"] = pd.to_numeric(df2["f1"], errors="coerce")
        df2 = df2.sort_values("f1", ascending=False)
        return df2.iloc[0]

    return df.iloc[0]


def _read_metrics(path: Path) -> pd.Series | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:  # hanya logging
        _log(f"[WARN] failed to read {path.name}: {e}")
        return None

    if df.empty:
        _log(f"[WARN] metrics file is empty: {path.name}")
        return None

    row = _pick_row(df)

    # pastikan kolom metrik & waktu ada; isi NaN jika tidak
    for c in _METRIC_COLS:
        if c not in row.index:
            row[c] = np.nan
    return row


def collect(datasets: List[str], variants: List[str]) -> pd.DataFrame:
    """
    Kumpulkan satu baris stacking per (dataset, variant) dari:
    reports/tables/<dataset>_stack_<variant}_metrics.csv
    """
    rows: List[Dict] = []
    for ds in datasets:
        for var in variants:
            metrics_path = TAB / f"{ds}_stack_{var}_metrics.csv"
            if not metrics_path.exists():
                _log(f"[SKIP] missing: {metrics_path.name}")
                continue

            row = _read_metrics(metrics_path)
            if row is None:
                continue

            out: Dict[str, object] = {"dataset": ds, "variant": var}
            for c in _METRIC_COLS:
                out[c] = pd.to_numeric(row.get(c, np.nan), errors="coerce")
            rows.append(out)

    return pd.DataFrame(rows)


def _order_categories(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" in df.columns:
        df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    if "variant" in df.columns:
        df["variant"] = pd.Categorical(
            df["variant"], categories=VARIANT_TAGS, ordered=True
        )
    return df.sort_values(["dataset", "variant"])


def _plot_bar(df: pd.DataFrame, metric: str, path: Path) -> None:
    """Grouped bar: x = variant, hue = dataset."""
    if metric not in df.columns or df.empty:
        _log(f"[INFO] cannot plot: metric '{metric}' missing or df empty.")
        return

    present_variants = [v for v in VARIANT_TAGS if v in set(df["variant"].astype(str))]
    if not present_variants:
        _log("[INFO] no variants present for plotting.")
        return

    x = np.arange(len(present_variants))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    for i, ds in enumerate(DATASETS):
        sub = df[df["dataset"] == ds]
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

    plt.xticks(x, present_variants, rotation=0)
    plt.ylabel(metric)
    plt.title(f"Comparison by {metric}")
    plt.legend()
    _savefig(path)


# ---------- SMOTE effect helpers ----------
def compute_smote_effects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung efek SMOTE dibanding baseline 'nosmote' untuk setiap dataset & varian.

    Output kolom:
    - dataset, variant, baseline_variant
    - <metric>_base, <metric>_variant, <metric>_delta
      untuk setiap metric utama (accuracy, precision, recall, f1, roc_auc, pr_auc).
    """
    if df.empty:
        return pd.DataFrame()

    df = _order_categories(df.copy())

    # baseline per dataset (variant == nosmote)
    base = df[df["variant"] == "nosmote"].set_index("dataset")
    if base.empty:
        _log("[SMOTE] No baseline 'nosmote' rows found; cannot compute effects.")
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for ds in df["dataset"].unique():
        if ds not in base.index:
            continue
        base_row = base.loc[ds]

        # semua varian selain nosmote
        sub = df[(df["dataset"] == ds) & (df["variant"] != "nosmote")]
        for _, r in sub.iterrows():
            out: Dict[str, object] = {
                "dataset": ds,
                "variant": r["variant"],
                "baseline_variant": "nosmote",
            }
            for m in _METRICS_MAIN:
                b_val = float(base_row.get(m, np.nan))
                v_val = float(r.get(m, np.nan))
                delta = (
                    v_val - b_val
                    if np.isfinite(b_val) and np.isfinite(v_val)
                    else np.nan
                )
                out[f"{m}_base"] = b_val
                out[f"{m}_variant"] = v_val
                out[f"{m}_delta"] = delta
            rows.append(out)

    return pd.DataFrame(rows)


def _plot_smote_effects_f1(df_eff: pd.DataFrame, path: Path) -> None:
    """Plot delta F1 (varian - nosmote) per dataset & varian."""
    if df_eff.empty or "f1_delta" not in df_eff.columns:
        _log("[SMOTE] No data to plot F1 effects.")
        return

    # order kategori
    df_eff = df_eff.copy()
    df_eff["dataset"] = pd.Categorical(
        df_eff["dataset"], categories=DATASETS, ordered=True
    )
    df_eff["variant"] = pd.Categorical(
        df_eff["variant"],
        categories=[v for v in VARIANT_TAGS if v != "nosmote"],
        ordered=True,
    )
    df_eff = df_eff.sort_values(["dataset", "variant"])

    variants = list(df_eff["variant"].cat.categories)
    x = np.arange(len(variants))
    width = 0.35

    plt.figure(figsize=(8.5, 4.8))
    for i, ds in enumerate(DATASETS):
        sub = df_eff[df_eff["dataset"] == ds]
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

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, variants)
    plt.ylabel("Δ F1 (variant − nosmote)")
    plt.title("Effect of SMOTE variants on F1 (stacking_lr_meta)")
    plt.legend()
    _savefig(path)


def handle_smote_effects(df: pd.DataFrame) -> None:
    """
    Wrapper terpisah untuk menghitung & menyimpan efek SMOTE,
    agar blok kode di main() tetap ringkas.
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


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate & compare stacking results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--variants", nargs="+", default=VARIANT_TAGS, choices=VARIANT_TAGS)
    ap.add_argument(
        "--sort-metric",
        # Fokus tesis: accuracy & precision → default sort pakai accuracy
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"],
    )
    ap.add_argument("--out-csv", default=str(TAB / "compare_runs.csv"))
    ap.add_argument("--out-md", default=str(TAB / "compare_runs.md"))
    ap.add_argument(
        "--plot-metric",
        # Default visualisasi juga pakai accuracy (bisa diubah via CLI)
        default="accuracy",
        choices=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "none"],
    )
    args = ap.parse_args()

    _log("[Compare] Collecting …")
    df = collect(args.datasets, args.variants)
    if df.empty:
        _log("[Compare] Nothing to compare (no metrics files found).")
        return

    df = _order_categories(df)
    if args.sort_metric in df.columns:
        # urutkan per dataset, lalu berdasarkan sort_metric (desc)
        df = df.sort_values(["dataset", args.sort_metric], ascending=[True, False])

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    _save_csv(df, out_csv)
    _save_md_table(df, out_md)
    _log(f"[Compare] Saved: {out_csv}")
    _log(f"[Compare] Saved: {out_md}")

    # ---- optional grouped bar untuk metric utama ----
    if args.plot_metric != "none":
        fig_path = FIG / f"compare_{args.plot_metric}.png"
        _plot_bar(df, args.plot_metric, fig_path)
        _log(f"[Compare] Saved: {fig_path}")

    # ---- SMOTE effects vs nosmote ----
    handle_smote_effects(df)


if __name__ == "__main__":
    main()