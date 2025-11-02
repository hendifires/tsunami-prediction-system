from __future__ import annotations
"""
Compare multiple stacking runs (nosmote / smote / smote_tomek / smote_enn)
by collecting metrics CSVs emitted by stacking_pipeline.py.

Outputs:
- reports/tables/compare_runs.csv   (one row per dataset+variant)
- reports/tables/compare_runs.md    (markdown table)
- reports/figures/compare_<metric>.png (grouped bar chart; optional)

Selection rule per metrics file:
1) Prefer row where model == 'stacking_lr_meta'
2) Fallback to the row with the highest F1
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
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "tn", "fp", "fn", "tp",
    "fit_sec", "grid_sec",  # waktu training per-model/stacking dan waktu meta-grid (NaN utk base models)
]

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
    """Ambil baris model stacking; fallback ke F1 tertinggi; terakhir baris pertama."""
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
    except Exception as e:
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
        df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_TAGS, ordered=True)
    return df.sort_values(["dataset", "variant"])

def _plot_bar(df: pd.DataFrame, metric: str, path: Path) -> None:
    """Grouped bar: x = variant, hue = dataset."""
    if metric not in df.columns or df.empty:
        _log(f"[INFO] cannot plot: metric '{metric}' missing or df empty.")
        return
    present_variants = [v for v in VARIANT_TAGS if v in set(df["variant"].astype(str))]
    x = np.arange(len(present_variants))
    width = 0.35
    plt.figure(figsize=(8.5, 4.8))
    for i, ds in enumerate(DATASETS):
        sub = df[df["dataset"] == ds]
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

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate & compare stacking results")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--variants", nargs="+", default=VARIANT_TAGS, choices=VARIANT_TAGS)
    ap.add_argument("--sort-metric", default="f1",
                    choices=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    ap.add_argument("--out-csv", default=str(TAB / "compare_runs.csv"))
    ap.add_argument("--out-md", default=str(TAB / "compare_runs.md"))
    ap.add_argument("--plot-metric", default="f1",
                    choices=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "none"])
    args = ap.parse_args()

    _log("[Compare] Collecting …")
    df = collect(args.datasets, args.variants)
    if df.empty:
        _log("[Compare] Nothing to compare (no metrics files found).")
        return

    df = _order_categories(df)
    if args.sort_metric in df.columns:
        df = df.sort_values(["dataset", args.sort_metric], ascending=[True, False])

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    _save_csv(df, out_csv)
    _save_md_table(df, out_md)
    _log(f"[Compare] Saved: {out_csv}")
    _log(f"[Compare] Saved: {out_md}")

    if args.plot_metric != "none":
        fig_path = FIG / f"compare_{args.plot_metric}.png"
        _plot_bar(df, args.plot_metric, fig_path)
        _log(f"[Compare] Saved: {fig_path}")

if __name__ == "__main__":
    main()