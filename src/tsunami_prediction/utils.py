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

def ensure_dirs():
    """Pastikan semua folder penting ada."""
    for d in [DATA, RAW, PROCESSED, REPORTS, FIG, TAB, MODELS, ART]:
        d.mkdir(parents=True, exist_ok=True)

def find_dataset_path(name: str) -> Path:
    """
    Cari file dataset dengan urutan prioritas:
    1) data/processed/{name}.csv | {name}_cleaned.csv | {name}_biner.csv
    2) data/raw/{name}.csv
    3) glob fallback: processed/raw yang mengandung prefix {name}
    """
    candidates = [
        PROCESSED / f"{name}.csv",
        PROCESSED / f"{name}_cleaned.csv",
        PROCESSED / f"{name}_biner.csv",
        RAW / f"{name}.csv",
    ]
    # glob fallback
    candidates += list(PROCESSED.glob(f"{name}*.csv"))
    candidates += list(RAW.glob(f"{name}*.csv"))

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Dataset for '{name}' not found. Tried: {candidates}")

def load_csv_smart(name: str) -> pd.DataFrame:
    """Load CSV berdasarkan find_dataset_path."""
    p = find_dataset_path(name)
    return pd.read_csv(p)