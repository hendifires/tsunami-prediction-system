# tests/tests_pipeline.py
# cSpell:ignore joblib smoteenn tomek artefak belum
"""
Smoke tests untuk memastikan pipeline berjalan setelah perubahan:
- split & artefak tersedia,
- artefak punya decision_threshold valid,
- serve_api bisa prediksi 1 baris (tectonic & volcanic),
- kolom distance_to_coast_km muncul jika FE dijalankan dengan shapefile.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import pytest
from joblib import load as joblib_load

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DATA = ROOT / "data"
PROC = DATA / "processed"


def _first_artifact(dataset: str) -> Path | None:
    """Ambil artefak pertama untuk dataset (jika ada)."""
    cand = sorted(ART.glob(f"{dataset}_stack_*.joblib"))
    return cand[0] if cand else None


def _require_file(p: Path) -> None:
    """Skip test bila file prasyarat belum dibuat (agar suite tetap hijau)."""
    if not p.exists():
        pytest.skip(f"File belum ada: {p.name} — jalankan pipeline dulu.")


# ---------- 1) Struktur dasar tersedia ----------
def test_required_dirs_exist():
    for p in (ART, PROC, ROOT / "reports" / "tables", ROOT / "reports" / "figures"):
        assert p.exists(), f"Folder hilang: {p}"


def test_processed_splits_exist_minimal():
    # Minimal: train/test non-SMOTE tersedia
    for ds in ("tectonic", "volcanic"):
        _require_file(PROC / f"{ds}_train.csv")
        _require_file(PROC / f"{ds}_test.csv")

    # Opsional: cek apakah ada setidaknya salah satu varian SMOTE
    smote_names = [
        "tectonic_train_smote.csv",
        "tectonic_train_smote_enn.csv",
        "tectonic_train_smote_tomek.csv",
        "volcanic_train_smote.csv",
        "volcanic_train_smote_enn.csv",
        "volcanic_train_smote_tomek.csv",
    ]
    any_smote = any((PROC / name).exists() for name in smote_names)
    if not any_smote:
        # Bukan kegagalan; hanya info agar kamu tahu SMOTE split belum dibuat.
        print("[tests] Optional: no SMOTE splits found; run smote_pipeline if needed.")


# ---------- 2) Artefak & threshold ----------
@pytest.mark.parametrize("dataset", ["tectonic", "volcanic"])
def test_artifact_has_threshold_and_features(dataset: str):
    arte = _first_artifact(dataset)
    if arte is None:
        pytest.skip(f"Artefak {dataset} belum dibuat.")
    obj = joblib_load(arte)
    assert "model" in obj and "feature_columns" in obj, "Artefak tidak lengkap."
    thr = float(obj.get("decision_threshold", 0.5))
    assert 0.0 <= thr <= 1.0, f"decision_threshold out of range: {thr}"


# ---------- 3) serve_api prediksi satu baris ----------
def test_serve_api_predict_tectonic_smoke():
    arte = _first_artifact("tectonic")
    if arte is None:
        pytest.skip("Artefak tectonic belum ada.")

    api = importlib.import_module("tsunami_prediction.serve_api")
    df = pd.DataFrame(
        {
            "mag": [6.8],
            "depth": [25],
            "latitude": [-3.2],
            "longitude": [100.5],
            "country": ["Indonesia"],
            "is_subduction_zone": ["Yes"],
            "distance_to_coast_km": [10],
        }
    )
    out = api.predict_tectonic_stacking(df, artifact_path=str(arte))
    assert {"prediction", "probability"} <= set(out.columns)
    p = float(out.loc[0, "probability"])
    assert 0.0 <= p <= 1.0


def test_serve_api_predict_volcanic_smoke():
    arte = _first_artifact("volcanic")
    if arte is None:
        pytest.skip("Artefak volcanic belum ada.")

    api = importlib.import_module("tsunami_prediction.serve_api")
    df = pd.DataFrame(
        {
            "eq": [5.2],
            "elevation": [1200],
            "vei": [3],
            "latitude": [-7.9],
            "longitude": [112.3],
            "country": ["Indonesia"],
            "type": ["Caldera"],
            "is_subduction_zone": ["Yes"],
            "distance_to_coast_km": [15],
        }
    )
    out = api.predict_volcanic_stacking(df, artifact_path=str(arte))
    assert {"prediction", "probability"} <= set(out.columns)
    p = float(out.loc[0, "probability"])
    assert 0.0 <= p <= 1.0


# ---------- 4) Coastline integration (opsional tapi tegas) ----------
def test_coastline_distance_present_if_shapefile_used():
    shp = DATA / "coastline" / "ne_10m_coastline.shp"
    if not shp.exists():
        pytest.skip("Shapefile coastline belum ada, lewati cek kolom jarak pantai.")

    # Jika FE dijalankan dengan --coast, kolom ini seharusnya ada.
    p = PROC / "tectonic_fe.csv"
    _require_file(p)
    df = pd.read_csv(p)
    assert "distance_to_coast_km" in df.columns, (
        "Kolom distance_to_coast_km tidak ada. "
        "Jalankan FE dengan argumen --coast data/coastline/ne_10m_coastline.shp"
    )
    # Tidak harus semuanya terisi, tapi jangan semuanya NaN.
    frac_nan = float(df["distance_to_coast_km"].isna().mean())
    assert frac_nan < 1.0, "Semua nilai distance_to_coast_km NaN — periksa CRS/FE."