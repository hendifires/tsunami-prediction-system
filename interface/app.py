# interface/app.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib

# =====================================================================
#   Path & model loading
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MODEL_FILENAME = "events_smote_stacking.joblib"
MODEL_PATH = ARTIFACTS_DIR / MODEL_FILENAME

# urutan fitur persis seperti saat training
FEATURE_COLS = [
    "abs_lat",
    "depth",
    "depth_log1p",
    "elev_log1p",
    "elevation",
    "is_tropic",
    "latitude",
    "longitude",
    "mag",
    "mag_sq",
    "vei",
    "vei_sq",
]

try:
    MODEL = joblib.load(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    # kalau gagal load, biarkan None dan nanti di endpoint dilempar error 500
    print(f"[ERROR] Cannot load model from {MODEL_PATH}: {exc}")
    MODEL = None

# =====================================================================
#   Pydantic schemas (payload dari UI)
# =====================================================================
class EventItem(BaseModel):
    mag: float
    depth: float
    latitude: float
    longitude: float
    vei: float
    elevation: float


class EventRequest(BaseModel):
    datas: List[EventItem]


# =====================================================================
#   FastAPI app & config
# =====================================================================
app = FastAPI(
    title="Tsunami Disaster Prediction API",
    description=(
        "API for predicting tsunami class (non-tsunami, tectonic-tsunami, "
        "volcanic-tsunami) using a Stacking Ensemble + SMOTE model "
        "trained on global tectonic & volcanic events."
    ),
    version="3.0",
    docs_url="/docs",
)

# CORS (dev/demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static landing (index.html)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# =====================================================================
#   Helpers
# =====================================================================
def _build_features(items: List[EventItem]) -> pd.DataFrame:
    """Bangun DataFrame fitur 12 kolom dari input mentah UI."""
    rows = []
    for it in items:
        lat = float(it.latitude)
        lon = float(it.longitude)
        mag = float(it.mag)
        depth = float(it.depth)
        elev = float(it.elevation)
        vei = float(it.vei)

        # jaga-jaga nilai negatif untuk log1p
        depth_pos = max(depth, 0.0)
        elev_pos = max(elev, 0.0)

        row = {
            "abs_lat": abs(lat),
            "depth": depth,
            "depth_log1p": float(np.log1p(depth_pos)),
            "elev_log1p": float(np.log1p(elev_pos)),
            "elevation": elev,
            "is_tropic": 1.0 if -23.5 <= lat <= 23.5 else 0.0,
            "latitude": lat,
            "longitude": lon,
            "mag": mag,
            "mag_sq": mag**2,
            "vei": vei,
            "vei_sq": vei**2,
        }
        rows.append(row)

    # pastikan urutan kolom konsisten
    df = pd.DataFrame(rows, columns=FEATURE_COLS)
    return df


def _build_api_response(
    y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, Any]:
    """
    Formatkan output model menjadi JSON standar untuk UI.

    - predictions   : list[int] (kelas 0/1/2)
    - probabilities : list[list[float]] (P(class0), P(class1), P(class2))
    - metadata statis (nama model, artifact, dsb.)
    """
    preds = [int(p) for p in y_pred]
    prob_list = [[float(p) for p in row] for row in y_proba]

    resp: Dict[str, Any] = {
        "predictions": preds,
        "probabilities": prob_list,
        "model_name": "events_smote_stacking",
        "smote_variant": "smote",
        "artifact": MODEL_FILENAME,
        # multi-class -> kita pakai argmax, threshold di sini tidak relevan
    }
    return resp


# =====================================================================
#   Health & Landing
# =====================================================================
@app.get("/healthz", tags=["Health"])
def healthz() -> Dict[str, str]:
    """Health check sederhana."""
    status = "ok" if MODEL is not None else "model-not-loaded"
    return {"status": status}


@app.get("/", response_class=HTMLResponse, tags=["Landing"])
async def root() -> HTMLResponse:
    """
    Render interface/static/index.html bila ada;
    jika tidak, kembalikan HTML sederhana.
    """
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return HTMLResponse(
            content=index_html.read_text(encoding="utf-8"),
            status_code=200,
        )
    return HTMLResponse(
        "<h1>Tsunami Prediction API</h1><p>Service is running.</p>",
        status_code=200,
    )


# =====================================================================
#   Single unified prediction endpoint
# =====================================================================
@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict tsunami class from event parameters (tectonic + volcanic)",
)
def predict_event(req: EventRequest) -> Dict[str, Any]:
    """
    Endpoint utama prediksi 3-kelas:

      - Class 0: Non-tsunami
      - Class 1: Tsunami Tektonik
      - Class 2: Tsunami Vulkanik

    Input: daftar EventItem dengan parameter:
      mag, depth, latitude, longitude, vei, elevation
    """
    if MODEL is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model not loaded from {MODEL_PATH}.",
        )

    if not req.datas:
        raise HTTPException(status_code=400, detail="Input 'datas' is empty.")

    try:
        X = _build_features(req.datas)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Failed to build features: {exc}",
        ) from exc

    try:
        proba = MODEL.predict_proba(X)
        preds = np.argmax(proba, axis=1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {exc}",
        ) from exc

    return _build_api_response(preds, proba)


# =====================================================================
#   Entry point (dev)
# =====================================================================
if __name__ == "__main__":
    # Jalankan langsung: python interface/app.py
    # (alternatif: uvicorn interface.app:app --reload)
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)