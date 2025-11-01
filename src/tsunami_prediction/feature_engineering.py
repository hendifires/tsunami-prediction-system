"""Feature engineering: domain features & safe helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Placeholder: jika kolom belum ada, buat default.
    if "is_subduction_zone" not in out.columns:
        out["is_subduction_zone"] = 0

    if "distance_to_coast_km" not in out.columns:
        out["distance_to_coast_km"] = np.nan  # isi kemudian jika tersedia

    if "days_since_prev" not in out.columns:
        # butuh event berdasar waktu; jika tak ada, isi NaN
        out["days_since_prev"] = np.nan

    return out

def select_columns(df: pd.DataFrame, num_cols, cat_cols, target: str):
    cols = [c for c in num_cols + cat_cols if c in df.columns]
    X = df[cols].copy()
    y = df[target].astype(int).values
    return X, y, cols
