from __future__ import annotations
# cSpell:ignore nosmote smote

"""
Compare stacking runs multi-year & multi-metric (dataset `events`).

Sinkron dengan pipeline baru:

- Metrik dibaca dari file-file:
    reports/tables/events_*_metrics.csv

  Contoh nama file yang didukung:
    events_nosmote_metrics.csv                  -> exp_tag = 'latest'
    events_smote_metrics.csv                    -> exp_tag = 'latest'
    events_nosmote_y1900_2024_metrics.csv       -> exp_tag = 'y1900_2024'
    events_smote_y1900_2024_metrics.csv         -> exp_tag = 'y1900_2024'
    events_smote_y2000_2024_metrics.csv         -> exp_tag = 'y2000_2024'
    dst.

- Kompatibel baik dengan format metrik lama maupun baru:
    Lama : kolom sudah punya ['dataset', 'scenario', 'exp_tag', 'model', ...]
    Baru : kolom minimal ['model', 'variant', 'accuracy', 'f1_macro', ...]

  Untuk format baru:
    * dataset   -> diisi 'events'
    * scenario  -> diambil dari kolom 'variant' atau dari nama file (nosmote/smote)
    * exp_tag   -> diambil dari nama file (bagian tengah setelah scenario), atau 'latest'

- Penandaan `role`:
    * 'main'     : model stacking terbaik (F1-macro maksimum)
                   untuk dataset='events', scenario='smote', model mengandung 'stacking'
    * 'baseline' : model non-stacking terbaik (F1-macro maksimum)
                   untuk dataset='events', scenario='nosmote'

- Output:
    * reports/tables/stacking_experiments_all_metrics.csv
    * reports/figures/events_stacking_multi_metric_multi_year.png
    * reports/figures/events_rf_multi_metric_multi_year.png
"""

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"

for p in (TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


# =========================
# Discover & load metrics
# =========================
def discover_metric_files(dataset: str = "events") -> List[Path]:
    """
    Cari semua file metrik bertipe:
        {dataset}_*_metrics.csv
    """
    files = sorted(TAB.glob(f"{dataset}_*_metrics.csv"))
    return files


def _parse_name_to_fields(dataset: str, stem: str) -> tuple[str, str, str]:
    """
    Parse nama file (tanpa .csv) menjadi:
        (dataset, scenario, exp_tag)

    Contoh:
        events_nosmote_metrics                -> ('events', 'nosmote', 'latest')
        events_smote_y1900_2024_metrics       -> ('events', 'smote', 'y1900_2024')
        events_smote_y2000_2024_metrics       -> ('events', 'smote', 'y2000_2024')
    """
    if not stem.endswith("_metrics"):
        raise ValueError(f"Nama file metrik tidak sesuai pola: {stem}")
    core = stem[: -len("_metrics")]
    parts = core.split("_")
    if len(parts) < 2:
        raise ValueError(f"Nama file metrik terlalu pendek: {stem}")

    ds = parts[0]
    sc = parts[1]
    if ds != dataset:
        raise ValueError(f"Dataset mismatch di nama file: {stem}")

    if len(parts) == 2:
        exp_tag = "latest"
    else:
        # gabungkan sisa sebagai exp_tag (mis. 'y1900_2024')
        exp_tag = "_".join(parts[2:])

    return ds, sc, exp_tag


def load_all_metrics(dataset: str = "events") -> pd.DataFrame:
    """
    Parse semua file metrik menjadi satu DataFrame.

    Mendukung:
        events_nosmote_metrics.csv
        events_smote_metrics.csv
        events_nosmote_y1900_2024_metrics.csv
        events_smote_y1900_2024_metrics.csv
        dst.
    """
    metric_files = discover_metric_files(dataset)
    rows = []

    for p in metric_files:
        try:
            ds, scenario_from_name, exp_tag = _parse_name_to_fields(dataset, p.stem)
        except ValueError:
            # abaikan file yang tidak match pola
            continue

        df = _read_csv(p).copy()

        # Normalisasi nama kolom agar konsisten
        # ------------------------------------
        # 1) dataset
        if "dataset" not in df.columns:
            df["dataset"] = ds

        # 2) scenario (nosmote / smote)
        if "scenario" in df.columns:
            pass
        elif "variant" in df.columns:
            df = df.rename(columns={"variant": "scenario"})
        else:
            df["scenario"] = scenario_from_name

        # 3) exp_tag (mis. 'y1900_2024' atau 'latest')
        if "exp_tag" not in df.columns:
            df["exp_tag"] = exp_tag
        else:
            df["exp_tag"] = exp_tag  # override agar sesuai nama file

        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"Tidak menemukan file metrik yang cocok untuk dataset='{dataset}' di {TAB}"
        )

    df_all = pd.concat(rows, ignore_index=True)
    return df_all


# =========================
# Role: main vs baseline
# =========================
def mark_main_and_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kolom 'role' untuk menandai:
    - 'main'      : model stacking terbaik (F1-macro maksimum) pada
                    dataset='events', scenario='smote', model mengandung 'stacking'
    - 'baseline'  : model non-stacking terbaik (F1-macro maksimum) pada
                    dataset='events', scenario='nosmote'
    """
    df = df.copy()
    if "role" not in df.columns:
        df["role"] = ""

    # Main model: stacking terbaik (smote)
    mask_main_pool = (
        (df["dataset"] == "events")
        & (df["scenario"] == "smote")
        & (df["model"].str.contains("stacking", case=False))
    )
    if mask_main_pool.any():
        idx_main = df.loc[mask_main_pool, "f1_macro"].idxmax()
        df.loc[idx_main, "role"] = "main"

    # Baseline: non-stacking terbaik (nosmote)
    mask_base_pool = (
        (df["dataset"] == "events")
        & (df["scenario"] == "nosmote")
        & (~df["model"].str.contains("stacking", case=False))
    )
    if mask_base_pool.any():
        idx_base = df.loc[mask_base_pool, "f1_macro"].idxmax()
        if df.loc[idx_base, "role"] == "":
            df.loc[idx_base, "role"] = "baseline"
        else:
            # kalau kebetulan sama barisnya (jarang), tandai gabungan
            df.loc[idx_base, "role"] = df.loc[idx_base, "role"] + "+baseline"

    return df


# =========================
# Plot multi-metric / multi-year
# =========================
def plot_multi_metric_multi_year(
    df: pd.DataFrame,
    dataset: str = "events",
    model: str = "stacking",
    out_png: Path | None = None,
) -> None:
    """
    Buat grafik komparasi performa (multi-metrik, multi-year) untuk 1 model.

    - x-axis  : exp_tag (mis. 'y1900_2024', 'y2000_2024', 'latest')
    - bar     : dua warna -> nosmote & smote (scenario)
    - subplot : accuracy, f1_macro, prec_macro, rec_macro
    """
    df = df.copy()
    df = df[df["dataset"] == dataset]
    df = df[df["model"] == model]

    if df.empty:
        _log(f"[Compare] Tidak ada data untuk model='{model}' pada dataset='{dataset}'.")
        return

    metrics = ["accuracy", "f1_macro", "prec_macro", "rec_macro"]
    scenarios = sorted(df["scenario"].unique().tolist())
    exp_tags = sorted(df["exp_tag"].unique().tolist())

    n_metrics = len(metrics)
    ncols = 2
    nrows = int(np.ceil(n_metrics / ncols))

    plt.figure(figsize=(10, 4 * nrows))

    x = np.arange(len(exp_tags))
    width = 0.8 / max(1, len(scenarios))  # bagi rata lebar bar

    for i, metric in enumerate(metrics, start=1):
        ax = plt.subplot(nrows, ncols, i)

        for j, sc in enumerate(scenarios):
            sub = df[df["scenario"] == sc]
            # buat mapping exp_tag -> nilai metrik
            vals = []
            for tag in exp_tags:
                row = sub[sub["exp_tag"] == tag]
                if row.empty:
                    vals.append(np.nan)
                else:
                    vals.append(float(row.iloc[0][metric]))
            xpos = x + (j - (len(scenarios) - 1) / 2) * width
            ax.bar(xpos, vals, width, label=sc.capitalize())

            # tulis angka di atas bar
            for xx, v in zip(xpos, vals):
                if np.isnan(v):
                    continue
                ax.text(
                    xx,
                    v + 0.005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(exp_tags, rotation=0)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} – {model}")
        ax.legend(fontsize=8)

    plt.tight_layout()

    if out_png is None:
        out_png = FIG / f"{dataset}_{model}_multi_metric_multi_year.png"

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    _log(f"[Compare] Saved multi-metric multi-year plot to {out_png}")


# =========================
# MAIN CLI
# =========================
def main() -> None:
    _log("[Compare] Start stacking runs comparison …")

    # 1) Load & gabungkan semua metrik
    df_all = load_all_metrics(dataset="events")
    _log(f"[Compare] Loaded {len(df_all)} rows of metrics.")

    # 2) Tandai main model & baseline
    df_all = mark_main_and_baseline(df_all)

    # 3) Simpan tabel gabungan
    out_csv = TAB / "stacking_experiments_all_metrics.csv"
    df_all_sorted = df_all.sort_values(
        by=["dataset", "exp_tag", "scenario", "model"]
    ).reset_index(drop=True)
    df_all_sorted.to_csv(out_csv, index=False)
    _log(f"[Compare] Saved combined metrics to {out_csv}")

    # 4) Grafik multi-metrik, multi-year untuk model utama (stacking)
    plot_multi_metric_multi_year(
        df_all_sorted,
        dataset="events",
        model="stacking",
        out_png=FIG / "events_stacking_multi_metric_multi_year.png",
    )

    # 5) Grafik untuk baseline Random Forest
    plot_multi_metric_multi_year(
        df_all_sorted,
        dataset="events",
        model="rf",
        out_png=FIG / "events_rf_multi_metric_multi_year.png",
    )

    _log(
        f"[DONE] Compare stacking runs finished.\n"
        f" - FIG: {FIG}\n"
        f" - TAB: {TAB}"
    )


if __name__ == "__main__":
    main()