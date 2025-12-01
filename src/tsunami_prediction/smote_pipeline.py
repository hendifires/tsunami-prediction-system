from __future__ import annotations
# cSpell:ignore oversampling imbalanced

"""
SMOTE pipeline sederhana untuk dataset gabungan multi-class (events).

Sinkron dengan preprocessing & stacking (multi-class: 0=non,1=tektonik,2=vulkanik).

------------------------------------
1. DATASET & FILE YANG DIGUNAKAN
------------------------------------
Prefix dataset (default: "events"):

- Input utama (dari preprocessing.py):
    * data/processed/{dataset}_train.csv  -> fitur + label (0/1/2)
    * data/processed/{dataset}_test.csv   -> test set (TIDAK disentuh SMOTE)

- Output dari script ini:
    * data/processed/{dataset}_train_smote.csv
    * reports/tables/{dataset}_smote_summary.csv
    * reports/figures/{dataset}_smote_bar.png
    * reports/figures/{dataset}_smote_pie.png
    * artifacts/{dataset}_smote_config.joblib  (metadata konfigurasi SMOTE)

------------------------------------
2. PERAN DALAM PENELITIAN
------------------------------------
Script ini menyiapkan **skenario "dengan SMOTE"** untuk dibandingkan dengan
baseline (tanpa SMOTE) pada Stacking Ensemble, yaitu:

- Baseline (no-SMOTE)   : {dataset}_train.csv
- Dengan SMOTE standar  : {dataset}_train_smote.csv

Test set TIDAK pernah di-oversample (anti data leakage); split train/test
selalu dilakukan di preprocessing.py.

------------------------------------
3. STRATEGI SAMPLING
------------------------------------
Argumen CLI --sampling-strategy mendukung dua mode:

- "not majority" (default, dan yang digunakan di tesis):
    - Menggunakan SMOTE(sampling_strategy="not majority")
      → semua kelas minoritas di-oversample hingga menyamai kelas mayoritas.

- "auto":
    - Menggunakan SMOTE(sampling_strategy="auto")
      → penentuan strategi diserahkan ke implementasi SMOTE.

Untuk kebutuhan eksplorasi lanjutan (di luar tesis), fungsi run_smote_events
juga menerima dict custom {class_label: target_count} bila dipanggil langsung
dari kode Python, tetapi mode ini tidak dibahas di naskah tesis.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Union

import matplotlib

matplotlib.use("Agg")  # non-GUI backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
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

# Nama kolom target yang didukung (multi-class 0/1/2)
TARGET_CANDIDATES: List[str] = ["tsunami_label", "label", "tsu"]


# ---------- Utils kecil ----------
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


def _format_axis_plain(ax) -> None:
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))


def _replace_non_finite(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ganti inf/-inf dengan NaN, lalu isi NaN dengan 0.

    Asumsi: preprocessing.py sudah melakukan imputasi dan scaling.
    Langkah ini hanya untuk jaga-jaga agar SMOTE tidak error karena nilai ekstrem.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0.0)


def _class_counts(y: pd.Series) -> Dict[int, int]:
    """Hitung jumlah sampel per kelas (dikembalikan sebagai dict terurut)."""
    cnt = Counter(int(v) for v in y)
    return dict(sorted(cnt.items(), key=lambda kv: kv[0]))


def _find_target_column(df: pd.DataFrame) -> str:
    """
    Cari nama kolom target berdasarkan kandidat yang sudah ditentukan.
    Raise error kalau tidak ditemukan.
    """
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        f"[SMOTE] Kolom target tidak ditemukan. "
        f"Diharapkan salah satu dari: {TARGET_CANDIDATES}, "
        f"tetapi kolom yang ada: {list(df.columns)}"
    )


# ---------- Visualisasi imbalance ----------
def plot_bar(
    before: Dict[int, int],
    after: Dict[int, int],
    dataset: str,
    out_png: Path,
) -> None:
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
    ax.set_title(f"Kelas sebelum dan sesudah SMOTE ({dataset}_train)")
    ax.legend()

    ymax = max(before_vals + after_vals) if (before_vals + after_vals) else 0
    offset = max(1, int(0.02 * max(ymax, 1)))
    for i, v in enumerate(before_vals):
        ax.text(
            x[i] - width / 2,
            v + offset,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for i, v in enumerate(after_vals):
        ax.text(
            x[i] + width / 2,
            v + offset,
            str(int(v)),
            ha="center",
            va="bottom",
            fontsize=8,
        )

    _savefig(out_png)


def plot_pie(
    before: Dict[int, int],
    after: Dict[int, int],
    dataset: str,
    out_png: Path,
) -> None:
    """
    Pie chart proporsi kelas sebelum & sesudah SMOTE
    dalam satu kanvas (2 subplots).
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
    plt.title(f"Before SMOTE ({dataset})")

    plt.subplot(1, 2, 2)
    if sum(after_vals) > 0:
        plt.pie(after_vals, labels=label_texts, autopct="%.1f%%", startangle=90)
    plt.title(f"After SMOTE ({dataset})")

    _savefig(out_png)


# ---------- Resolusi strategi sampling ----------
def _resolve_sampling_strategy(
    before_counts: Dict[int, int],
    sampling_strategy: Union[str, Dict[int, int]],
) -> Union[str, Dict[int, int]]:
    """
    Resolusi parameter sampling_strategy untuk SMOTE.

    - Jika dict:
        -> dipakai apa adanya (mode advanced, tidak dibahas di tesis).
    - Jika string:
        * 'not majority' / 'not_majority'  -> 'not majority'
        * 'auto'                           -> 'auto'
        * selain itu                        -> fallback ke 'not majority'
    """
    if isinstance(sampling_strategy, dict):
        _log(f"[SMOTE] Menggunakan custom dict sampling_strategy: {sampling_strategy}")
        return sampling_strategy

    mode = sampling_strategy.strip().lower()

    if mode in {"not majority", "not_majority"}:
        _log("[SMOTE] sampling_strategy='not majority' (SMOTE standar multi-class).")
        return "not majority"

    if mode == "auto":
        _log("[SMOTE] sampling_strategy='auto' (delegasi penuh ke SMOTE).")
        return "auto"

    _log(
        f"[SMOTE] sampling_strategy='{sampling_strategy}' tidak dikenal, "
        "fallback ke 'not majority'."
    )
    return "not majority"


# ---------- Core: SMOTE pada {dataset}_train ----------
def run_smote_events(
    dataset: str = "events",
    overwrite: bool = False,
    random_state: int = 42,
    sampling_strategy: Union[str, Dict[int, int]] = "not majority",
    k_neighbors: int = 5,
) -> None:
    """
    Jalankan SMOTE pada data training gabungan ({dataset}_train):

    - Input  : data/processed/{dataset}_train.csv  (fitur + label multi-class)
    - Output : data/processed/{dataset}_train_smote.csv

    Parameter penting:
    - dataset           : prefix nama dataset (default 'events')
    - overwrite         : jika True, regenerasi meskipun file hasil sudah ada
    - random_state      : seed untuk reproducibility
    - sampling_strategy :
          * 'not majority' (default, digunakan di tesis)
          * 'auto'
          * dict custom {class_label: target_count} jika dipanggil dari Python
    - k_neighbors       : jumlah tetangga SMOTE (default=5)
    """
    train_path = PROCESSED / f"{dataset}_train.csv"
    smote_path = PROCESSED / f"{dataset}_train_smote.csv"
    summary_path = TAB / f"{dataset}_smote_summary.csv"
    bar_path = FIG / f"{dataset}_smote_bar.png"
    pie_path = FIG / f"{dataset}_smote_pie.png"
    config_path = ART / f"{dataset}_smote_config.joblib"

    if (
        smote_path.exists()
        and summary_path.exists()
        and bar_path.exists()
        and pie_path.exists()
        and not overwrite
    ):
        _log(
            f"[SMOTE] {dataset}_train_smote.csv & ringkasan sudah ada, "
            "gunakan --overwrite untuk regenerasi."
        )
        return

    if not train_path.exists():
        raise FileNotFoundError(
            f"[SMOTE] {train_path} tidak ditemukan. "
            "Pastikan preprocessing.py sudah dijalankan."
        )

    df = _read_csv(train_path)

    target_col = _find_target_column(df)
    y = df[target_col].astype("int64")
    X = df.drop(columns=[target_col])

    # Bersihkan nilai inf/NaN (jaga-jaga)
    X = _replace_non_finite(X)

    _log(f"[SMOTE] {dataset}_train shape sebelum SMOTE: {X.shape}")
    before_counts = _class_counts(y)
    _log(f"[SMOTE] Distribusi label sebelum SMOTE: {before_counts}")

    # Sesuaikan k_neighbors bila kelas minoritas sangat sedikit
    min_count = min(before_counts.values())
    k_eff = min(k_neighbors, max(1, min_count - 1))
    if k_eff < k_neighbors:
        _log(
            f"[SMOTE] k_neighbors={k_neighbors} terlalu besar untuk kelas minoritas, "
            f"disesuaikan menjadi {k_eff}."
        )

    # Resolusi strategi sampling (string preset -> string/dict final)
    sampling_resolved = _resolve_sampling_strategy(before_counts, sampling_strategy)

    smote = SMOTE(
        random_state=random_state,
        sampling_strategy=sampling_resolved,
        k_neighbors=k_eff,
    )

    X_res, y_res = smote.fit_resample(X, y)
    _log(f"[SMOTE] {dataset}_train shape sesudah SMOTE: {X_res.shape}")

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res[target_col] = y_res.astype("int64")

    # Simpan data hasil resampling
    df_res.to_csv(smote_path, index=False)

    # Ringkasan kelas sebelum & sesudah
    after_counts = _class_counts(y_res)

    all_classes = sorted(set(before_counts.keys()) | set(after_counts.keys()))
    summary_df = pd.DataFrame(
        {
            "class": all_classes,
            "count_before": [before_counts.get(c, 0) for c in all_classes],
            "count_after": [after_counts.get(c, 0) for c in all_classes],
        }
    )
    _savetab(summary_df, summary_path)

    # Plot
    plot_bar(before_counts, after_counts, dataset, bar_path)
    plot_pie(before_counts, after_counts, dataset, pie_path)

    # Simpan metadata config sederhana
    dump(
        {
            "dataset": dataset,
            "target_column": target_col,
            "random_state": random_state,
            "sampling_strategy_raw": sampling_strategy,
            "sampling_strategy_resolved": sampling_resolved,
            "k_neighbors_requested": k_neighbors,
            "k_neighbors_used": k_eff,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "input_path": str(train_path),
            "output_path": str(smote_path),
        },
        config_path,
    )

    _log(f"[SMOTE] Selesai oversampling {dataset}_train dengan SMOTE.")
    _log(
        f"        Output  : {smote_path.name}\n"
        f"        Summary : {summary_path.name}\n"
        f"        Figures : {bar_path.name}, {pie_path.name}\n"
        f"        Config  : {config_path.name}"
    )


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "SMOTE pipeline sederhana untuk {dataset}_train (multi-class 0/1/2). "
            "Default dataset='events'."
        )
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="events",
        help="prefix nama dataset di data/processed (default: 'events').",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite *_train_smote.csv dan ringkasan jika sudah ada.",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="seed random untuk SMOTE (default=42).",
    )
    ap.add_argument(
        "--sampling-strategy",
        type=str,
        default="not majority",
        help=(
            "strategi sampling SMOTE. Pilihan utama:\n"
            "  - 'not majority' (default): upsample semua minoritas ke mayoritas\n"
            "  - 'auto'         : delegasi ke SMOTE\n"
            "Untuk eksperimen lanjutan, run_smote_events bisa dipanggil dari "
            "Python dengan dict custom {class_label: target_count}."
        ),
    )
    ap.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="jumlah tetangga SMOTE (default=5).",
    )
    args = ap.parse_args()

    _log("[SMOTE] Pipeline start (multi-class)…")

    run_smote_events(
        dataset=args.dataset,
        overwrite=args.overwrite,
        random_state=args.random_state,
        sampling_strategy=args.sampling_strategy,
        k_neighbors=args.k_neighbors,
    )

    _log(
        f"[DONE] SMOTE pipeline selesai.\n"
        f" - DATA : {PROCESSED}\n"
        f" - TAB  : {TAB}\n"
        f" - FIG  : {FIG}\n"
        f" - ART  : {ART}"
    )


if __name__ == "__main__":
    main()