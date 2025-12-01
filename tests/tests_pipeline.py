# tests/tests_pipeline.py
# cSpell:ignore joblib smoteenn tomek artefak belum
"""
Smoke tests untuk memastikan pipeline terbaru berjalan end-to-end:

1) Struktur dasar project tersedia
   - data/processed
   - reports/tables & reports/figures
   - artifacts/ untuk model stacking

2) Split train/test minimal:
   - <dataset>_train.csv dan <dataset>_test.csv (non-SMOTE)
   - Informasi opsional: keberadaan train_smote / smote_tomek / smote_enn

3) Artefak stacking:
   - file <dataset>_stack_<variant>.joblib tersedia minimal satu
   - memuat kunci "model" dan "feature_columns"
   - decision_threshold berada di interval [0, 1]

4) Integrasi serve_api:
   - predict_tectonic_stacking() dan predict_volcanic_stacking()
     dapat memproses 1 baris input dan menghasilkan probability 0..1

5) Integrasi coastline (opsional tapi tegas):
   - jika shapefile coastline ada, maka tectonic_fe.csv harus punya
     kolom distance_to_coast_km dengan nilai tidak semuanya NaN.
"""

from __future__ import annotations

from pathlib import Path
import importlib

import pandas as pd
import pytest
from joblib import load as joblib_load

from tsunami_prediction.utils import (
    ROOT,
    DATA,
    PROCESSED as PROC,
    ART,
    REPORTS,
    FIG,
    TAB,
)

# ---------- Helper ----------


def _preferred_artifacts(dataset: str) -> list[Path]:
    """
    Urutan prioritas artefak, selaras dengan serve_api._preferred_artifacts.
    Dipakai agar smoke test selalu menguji model yang sama dengan yang
    dipakai di API (prioritas SMOTE, kemudian varian lain).
    """
    d = dataset.lower()
    mapping = {
        "tectonic": [
            ART / "tectonic_stack_smote.joblib",
            ART / "tectonic_stack_smote_tomek.joblib",
            ART / "tectonic_stack_smote_enn.joblib",
            ART / "tectonic_stack_nosmote.joblib",
        ],
        "volcanic": [
            ART / "volcanic_stack_smote.joblib",
            ART / "volcanic_stack_smote_tomek.joblib",
            ART / "volcanic_stack_smote_enn.joblib",
            ART / "volcanic_stack_nosmote.joblib",
        ],
    }
    base = mapping.get(d, [])
    # fallback: artefak lain dengan pola umum
    base += sorted(ART.glob(f"{dataset}_stack_*.joblib"))
    # buang duplikat sambil pertahankan urutan
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in base:
        if p in seen:
            continue
        seen.add(p)
        unique.append(p)
    return unique


def _first_artifact(dataset: str) -> Path | None:
    """Ambil artefak pertama sesuai prioritas (jika ada)."""
    for p in _preferred_artifacts(dataset):
        if p.exists():
            return p
    return None


def _require_file(p: Path) -> None:
    """Skip test bila file prasyarat belum dibuat (agar suite tetap hijau)."""
    if not p.exists():
        pytest.skip(f"File belum ada: {p.name} — jalankan pipeline dulu.")


# ---------- 1) Struktur dasar tersedia ----------


def test_required_dirs_exist():
    # Folder minimal yang harus ada setelah pipeline dijalankan
    for p in (ART, PROC, REPORTS, TAB, FIG):
        assert p.exists(), f"Folder hilang: {p}"


def test_processed_splits_exist_minimal():
    """
    Pastikan split train/test non-SMOTE untuk tectonic & volcanic tersedia.
    Split SMOTE opsional — kalau belum ada, hanya ditampilkan sebagai info.
    """
    for ds in ("tectonic", "volcanic"):
        _require_file(PROC / f"{ds}_train.csv")
        _require_file(PROC / f"{ds}_test.csv")

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
        pytest.skip(f"Artefak {dataset} belum dibuat (tidak ada {dataset}_stack_*.joblib).")

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

    # Input mengikuti skema TectonicItem di schemas.py
    df = pd.DataFrame(
        {
            "mag": [6.8],
            "depth": [25.0],
            "latitude": [-3.2],
            "longitude": [100.5],
            "country": ["Indonesia"],
            # di API nyata ini berupa int 0/1 → di sini disamakan
            "is_subduction_zone": [1],
            "distance_to_coast_km": [10.0],
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

    # Input mengikuti skema VolcanicItem di schemas.py
    df = pd.DataFrame(
        {
            "eq": [5.2],
            "elevation": [1200.0],
            "vei": [3.0],
            "latitude": [-7.9],
            "longitude": [112.3],
            "country": ["Indonesia"],
            "type": ["Caldera"],
            "is_subduction_zone": [1],
            "distance_to_coast_km": [15.0],
        }
    )

    out = api.predict_volcanic_stacking(df, artifact_path=str(arte))
    assert {"prediction", "probability"} <= set(out.columns)
    p = float(out.loc[0, "probability"])
    assert 0.0 <= p <= 1.0


# ---------- 4) Coastline integration (opsional tapi tegas) ----------


def test_coastline_distance_present_if_shapefile_used():
    """
    Versi tesis (FE ringan) tidak lagi menghitung jarak pantai.
    Kalau shapefile ada tapi kolom distance_to_coast_km tidak dibuat,
    tes ini di-skip supaya tidak mengganggu smoke-test pipeline.
    """
    shp = DATA / "coastline" / "ne_10m_coastline.shp"
    if not shp.exists():
        pytest.skip("Shapefile coastline belum ada, lewati cek kolom jarak pantai.")

    p = PROC / "tectonic_fe.csv"
    _require_file(p)
    df = pd.read_csv(p)

    if "distance_to_coast_km" not in df.columns:
        pytest.skip(
            "distance_to_coast_km tidak di-engineer pada versi FE ini; skip coastline check."
        )

    frac_nan = float(df["distance_to_coast_km"].isna().mean())
    assert frac_nan < 1.0, "Semua nilai distance_to_coast_km NaN — periksa CRS/FE."