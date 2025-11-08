# src/tsunami_prediction/utils.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

# --- Project paths ---
ROOT = Path(__file__).resolve().parents[2]   # <repo root> (karena file ini di src/tsunami_prediction/)
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TAB = REPORTS / "tables"
MODELS = ROOT / "models"
ART = ROOT / "artifacts"


def ensure_dirs() -> None:
    """
    Pastikan semua folder penting ada.

    Dipakai oleh berbagai modul (preprocessing, FE, SMOTE, stacking, EDA)
    supaya struktur:
      - data/raw
      - data/processed
      - reports/{figures,tables}
      - models
      - artifacts
    selalu tersedia.
    """
    for d in [DATA, RAW, PROCESSED, REPORTS, FIG, TAB, MODELS, ART]:
        d.mkdir(parents=True, exist_ok=True)


def find_dataset_path(name: str) -> Path:
    """
    Cari file dataset dengan urutan prioritas yang sinkron dengan pipeline terbaru:

    1) data/processed/
       - {name}.csv                → mis. tectonic.csv, volcanic.csv (CLEAN utama)
       - {name}_preprocessed.csv   → output preprocessing (setelah scaling/OHE global)
       - {name}_fe_ohe.csv         → hasil FE + OHE (jika dimaterialisasi)
       - {name}_fe.csv             → hasil feature_engineering
       - {name}_cleaned.csv        → fallback lama
       - {name}_biner.csv          → fallback lama

    2) data/raw/{name}.csv         → jika processed belum ada

    3) Fallback glob:
       - processed: semua file yang diawali prefix {name}
       - raw      : idem

    Catatan:
    - Kalau name sudah mengandung suffix (mis. 'tectonic_preprocessed'), prioritas
      tetap PROCESSED/{name}.csv terlebih dulu sehingga aman dipakai untuk file spesifik.
    """
    # kandidat eksplisit (tidak pakai glob dulu supaya tidak salah pilih *train*/*smote* dsb.)
    candidates = [
        PROCESSED / f"{name}.csv",
        PROCESSED / f"{name}_preprocessed.csv",
        PROCESSED / f"{name}_fe_ohe.csv",
        PROCESSED / f"{name}_fe.csv",
        PROCESSED / f"{name}_cleaned.csv",
        PROCESSED / f"{name}_biner.csv",
        RAW / f"{name}.csv",
    ]

    # glob fallback (processed & raw) – tetap diurutkan setelah kandidat spesifik
    # contoh: tectonic_train.csv, tectonic_train_smote.csv, dll.
    candidates += list(PROCESSED.glob(f"{name}*.csv"))
    candidates += list(RAW.glob(f"{name}*.csv"))

    # buang duplikat sambil mempertahankan urutan pertama
    seen = set()
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

    - Otomatis mencari di data/processed dan data/raw dengan prioritas yang benar.
    - Setelah baca, kolom duplikat akan di-drop (selaras dengan EDA & preprocessing).
    """
    p = find_dataset_path(name)
    df = pd.read_csv(p)
    # buang kolom duplikat yang mungkin muncul akibat normalisasi nama kolom di tahap sebelumnya
    if pd.Index(df.columns).duplicated().any():
        df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    return df