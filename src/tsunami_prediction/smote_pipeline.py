# src/tsunami_prediction/smote_pipeline.py
from __future__ import annotations
# cSpell:ignore oversampling imbalanced

"""
SMOTE pipeline sederhana untuk dataset gabungan multi-class (events):

Desain baru (sinkron dengan preprocessing & stacking):

- Dataset utama:
    * data/processed/events_train.csv  -> fitur + label (0=non,1=tektonik,2=vulkanik)
    * data/processed/events_test.csv   -> test set (tidak disentuh SMOTE)

- Tugas file ini:
    * Membaca events_train.csv
    * Menerapkan SMOTE standar (imblearn.over_sampling.SMOTE) pada training set
      dengan target multi-class 'label'.
    * Menyimpan hasil oversampling ke:
          data/processed/events_train_smote.csv
    * Membuat ringkasan imbalance:
          reports/tables/events_smote_summary.csv
    * Membuat visual perbandingan jumlah sampel per kelas sebelum/sesudah:
          reports/figures/events_smote_bar.png
          reports/figures/events_smote_pie.png

- Tujuan ilmiah:
    * Menyediakan skenario "dengan SMOTE" untuk dibandingkan dengan
      baseline (tanpa SMOTE) di Stacking Ensemble:
        - baseline   : events_train.csv       (no SMOTE)
        - SMOTE      : events_train_smote.csv (SMOTE di training set)

Catatan:
- Test set TIDAK pernah di-oversampling (anti-data-leakage).
- Hanya pakai SMOTE standar; tidak ada SMOTE-Tomek / SMOTEENN / ADASYN
  dan tidak ada split ulang (semua split dilakukan di preprocessing.py).
"""

import argparse
import ast
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE

from matplotlib.ticker import ScalarFormatter


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


# ---------- Utils kecil ----------

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


def _format_axis_plain(ax) -> None:
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))


def _replace_non_finite(X: pd.DataFrame) -> pd.DataFrame:
    """Ganti inf/-inf dengan NaN, lalu isi NaN dengan 0 (fitur sudah di-scale sebelumnya)."""
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0.0)


def _class_counts(y: pd.Series) -> Dict[int, int]:
    """Hitung jumlah sampel per kelas (dikembalikan sebagai dict)."""
    cnt = Counter(int(v) for v in y)
    # sort by class label
    return dict(sorted(cnt.items(), key=lambda kv: kv[0]))


# ---------- Visualisasi imbalance ----------

def plot_bar(before: Dict[int, int], after: Dict[int, int], out_png: Path) -> None:
    """
    Bar plot jumlah sampel per kelas sebelum dan sesudah SMOTE.
    Label kelas:
        0 = non-tsunami
        1 = tsunami tektonik
        2 = tsunami vulkanik
    """
    labels = sorted(set(before.keys()) | set(after.keys()))
    labels_str = [f"Class {c}" for c in labels]

    before_vals = [before.get(c, 0) for c in labels]
    after_vals = [after.get(c, 0) for c in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    ax = plt.gca()

    ax.bar(x - width / 2, before_vals, width, label="Before SMOTE")
    ax.bar(x + width / 2, after_vals, width, label="After SMOTE")

    _format_axis_plain(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_str)
    ax.set_ylabel("Jumlah Sampel")
    ax.set_title("Kelas sebelum dan sesudah SMOTE (events_train)")
    ax.legend()

    # Tulis angka di atas bar
    ymax = max(before_vals + after_vals) if (before_vals + after_vals) else 0
    offset = max(1, int(0.02 * max(ymax, 1)))
    for i, v in enumerate(before_vals):
        ax.text(x[i] - width / 2, v + offset, str(int(v)), ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(after_vals):
        ax.text(x[i] + width / 2, v + offset, str(int(v)), ha="center", va="bottom", fontsize=8)

    _savefig(out_png)


def plot_pie(before: Dict[int, int], after: Dict[int, int], out_png: Path) -> None:
    """
    Pie chart proporsi kelas sebelum & sesudah SMOTE dalam satu kanvas (2 subplots).
    """
    labels = sorted(set(before.keys()) | set(after.keys()))
    label_names = {
        0: "Non-tsunami (0)",
        1: "Tsunami tektonik (1)",
        2: "Tsunami vulkanik (2)",
    }
    label_texts = [label_names.get(c, f"Class {c}") for c in labels]

    before_vals = [before.get(c, 0) for c in labels]
    after_vals = [after.get(c, 0) for c in labels]

    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    if sum(before_vals) > 0:
        plt.pie(before_vals, labels=label_texts, autopct="%.1f%%", startangle=90)
    plt.title("Before SMOTE")

    plt.subplot(1, 2, 2)
    if sum(after_vals) > 0:
        plt.pie(after_vals, labels=label_texts, autopct="%.1f%%", startangle=90)
    plt.title("After SMOTE")

    _savefig(out_png)


# ---------- Core: SMOTE pada events_train ----------

def run_smote_events(
    overwrite: bool = False,
    random_state: int = 42,
    sampling_strategy="not majority",
    k_neighbors: int = 5,
) -> None:
    """
    Jalankan SMOTE pada data training gabungan (events_train):

    - Input  : data/processed/events_train.csv (fitur + 'label')
    - Output : data/processed/events_train_smote.csv

    Parameter penting:
    - random_state      : seed untuk reproducibility
    - sampling_strategy : default 'not majority' (upsample semua kelas minoritas)
    - k_neighbors       : tetangga SMOTE (default=5)
    """
    events_train_path = PROCESSED / "events_train.csv"
    events_smote_path = PROCESSED / "events_train_smote.csv"
    summary_path = TAB / "events_smote_summary.csv"
    bar_path = FIG / "events_smote_bar.png"
    pie_path = FIG / "events_smote_pie.png"

    if (
        events_smote_path.exists()
        and summary_path.exists()
        and bar_path.exists()
        and pie_path.exists()
        and not overwrite
    ):
        print(
            "[SMOTE] events_train_smote.csv & ringkasan sudah ada, "
            "gunakan --overwrite untuk regenerasi."
        )
        return

    if not events_train_path.exists():
        raise FileNotFoundError(
            f"[SMOTE] {events_train_path} tidak ditemukan. "
            "Pastikan preprocessing.py sudah dijalankan."
        )

    df = _read_csv(events_train_path)
    if "label" not in df.columns:
        raise KeyError(
            "[SMOTE] Kolom 'label' tidak ditemukan di events_train.csv. "
            "Pastikan preprocessing.py menghasilkan label 0/1/2."
        )

    # Pisahkan fitur & target
    y = df["label"].astype("int64")
    X = df.drop(columns=["label"])

    # Bersihkan nilai inf/NaN (preprocessing sebelumnya sudah scaling dan imputasi,
    # tetapi langkah ini menjaga agar SMOTE tidak error).
    X = _replace_non_finite(X)

    print("[SMOTE] events_train shape sebelum SMOTE:", X.shape)
    print("[SMOTE] Distribusi label sebelum SMOTE:", _class_counts(y))

    # Sesuaikan k_neighbors bila kelas minoritas sangat sedikit
    class_counts = _class_counts(y)
    min_count = min(class_counts.values())
    k_eff = min(k_neighbors, max(1, min_count - 1))
    if k_eff < k_neighbors:
        print(
            f"[SMOTE] k_neighbors={k_neighbors} terlalu besar untuk kelas minoritas, "
            f"disesuaikan menjadi {k_eff}."
        )

    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_eff,
    )

    X_res, y_res = smote.fit_resample(X, y)
    print("[SMOTE] events_train shape sesudah SMOTE:", X_res.shape)

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res["label"] = y_res.astype("int64")

    # Simpan data hasil resampling
    df_res.to_csv(events_smote_path, index=False)

    # Ringkasan kelas sebelum & sesudah
    before = _class_counts(y)
    after = _class_counts(y_res)

    summary_df = pd.DataFrame(
        {
            "class": sorted(set(before.keys()) | set(after.keys())),
            "count_before": [before.get(c, 0) for c in sorted(before.keys() | after.keys())],
            "count_after": [after.get(c, 0) for c in sorted(before.keys() | after.keys())],
        }
    )
    _savetab(summary_df, summary_path)

    # Plot
    plot_bar(before, after, bar_path)
    plot_pie(before, after, pie_path)

    # Simpan metadata config sederhana
    from joblib import dump
    dump(
        {
            "random_state": random_state,
            "sampling_strategy": sampling_strategy,
            "k_neighbors_requested": k_neighbors,
            "k_neighbors_used": k_eff,
            "before_counts": before,
            "after_counts": after,
            "inputs": str(events_train_path),
            "outputs": str(events_smote_path),
        },
        ART / "events_smote_config.joblib",
    )

    print("[SMOTE] Selesai oversampling events_train dengan SMOTE standar.")
    print("        Output:", events_smote_path.name)
    print("        Summary:", summary_path.name)
    print("        Figures:", bar_path.name, ",", pie_path.name)


# ---------- CLI ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SMOTE pipeline sederhana untuk events_train (multi-class 0/1/2)."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite events_train_smote.csv dan ringkasan jika sudah ada",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="seed random untuk SMOTE (default=42)",
    )
    ap.add_argument(
        "--sampling-strategy",
        type=str,
        default="'not majority'",
        help=(
            "strategi sampling SMOTE (default='not majority'). "
            "Bisa 'auto', 'not majority', atau literal Python yang valid, "
            "misalnya '{1: 500, 2: 300}'."
        ),
    )
    ap.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="jumlah tetangga SMOTE (default=5)",
    )
    args = ap.parse_args()

    # parsing aman untuk sampling_strategy
    sampling_strategy = args.sampling_strategy
    try:
        # Jika string mengandung dict / list / string literal, gunakan ast.literal_eval
        sampling_strategy = ast.literal_eval(sampling_strategy)
    except Exception:
        # fallback: pakai string apa adanya (mis. 'not majority', 'auto')
        sampling_strategy = args.sampling_strategy

    print("[SMOTE] Pipeline start (events, multi-class)…")
    run_smote_events(
        overwrite=args.overwrite,
        random_state=args.random_state,
        sampling_strategy=sampling_strategy,
        k_neighbors=args.k_neighbors,
    )
    print(
        f"[DONE] SMOTE pipeline selesai.\n"
        f" - DATA : {PROCESSED}\n"
        f" - TAB  : {TAB}\n"
        f" - FIG  : {FIG}\n"
        f" - ART  : {ART}"
    )


if __name__ == "__main__":
    main()