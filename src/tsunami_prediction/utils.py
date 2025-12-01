# src/tsunami_prediction/utils.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

# ================== Project paths (shared) ==================
# File ini berada di: src/tsunami_prediction/utils.py
# Jadi ROOT adalah folder repo utama (yang berisi data/, reports/, artifacts/, dll.)
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TAB = REPORTS / "tables"

# models lama masih dipertahankan (mis. untuk baseline),
# sedangkan artefak utama stacking (joblib) sekarang di ART.
MODELS = ROOT / "models"
ART = ROOT / "artifacts"


def ensure_dirs() -> None:
    """
    Pastikan semua folder penting ada.

    Dipakai oleh berbagai modul:
        - preprocessing.py
        - feature_engineering.py
        - smote_pipeline.py
        - stacking_pipeline.py
        - eda.py
        - analyze_stacking_results.py
        - compare_stacking_runs.py
        - serve_api.py (melalui artifacts)

    Struktur minimum yang dipastikan:

        data/raw
        data/processed
        reports/figures
        reports/tables
        models
        artifacts
    """
    for d in [DATA, RAW, PROCESSED, REPORTS, FIG, TAB, MODELS, ART]:
        d.mkdir(parents=True, exist_ok=True)


def find_dataset_path(name: str) -> Path:
    """
    Cari file dataset dengan urutan prioritas yang sinkron dengan
    pipeline preprocessing/feature engineering terbaru.

    Prioritas:

    1) data/processed/  (untuk data yang sudah dibersihkan / diproses)
       - {name}_fe.csv           → output feature_engineering (mis. tectonic_fe, events_fe)
       - {name}_fe_ohe.csv       → FE + OHE (jika dimaterialisasi)
       - {name}.csv              → clean utama (tectonic.csv, volcanic.csv, events.csv, dst.)
       - {name}_preprocessed.csv → hasil preprocessing global
       - {name}_cleaned.csv      → fallback lama
       - {name}_biner.csv        → fallback lama (biner label)

    2) data/raw/
       - {name}.csv              → jika processed belum tersedia

    3) Glob fallback:
       - processed: {name}*.csv  (mis. tectonic_train.csv, events_subset.csv, dst.)
       - raw      : {name}*.csv

    Catatan:
    - Jika name sudah spesifik (mis. 'tectonic_preprocessed' atau 'events_fe'),
      maka entri paling awal adalah PROCESSED/{name}.csv sehingga tetap aman.
    """
    # kandidat eksplisit (tidak pakai glob dulu supaya tidak salah pilih file *train* / *smote*)
    candidates = [
        PROCESSED / f"{name}_fe.csv",
        PROCESSED / f"{name}_fe_ohe.csv",
        PROCESSED / f"{name}.csv",
        PROCESSED / f"{name}_preprocessed.csv",
        PROCESSED / f"{name}_cleaned.csv",
        PROCESSED / f"{name}_biner.csv",
        RAW / f"{name}.csv",
    ]

    # glob fallback (processed & raw) – diurutkan setelah kandidat eksplisit
    candidates += list(PROCESSED.glob(f"{name}*.csv"))
    candidates += list(RAW.glob(f"{name}*.csv"))

    # buang duplikat sambil mempertahankan urutan
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Dataset for '{name}' not found. Tried: {unique_candidates}"
    )


def load_csv_smart(name: str) -> pd.DataFrame:
    """
    Load CSV berdasarkan find_dataset_path(name).

    Fitur:
    - Otomatis mencari di data/processed dan data/raw dengan prioritas yang konsisten
      dengan pipeline terbaru (FE → clean → preprocessed → raw).
    - Setelah baca, kolom duplikat akan di-drop
      (align dengan EDA & preprocessing supaya tidak ada kolom ganda).
    """
    p = find_dataset_path(name)
    df = pd.read_csv(p)

    # buang kolom duplikat yang mungkin muncul akibat normalisasi nama kolom di tahap sebelumnya
    if pd.Index(df.columns).duplicated().any():
        df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    return df