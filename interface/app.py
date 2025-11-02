# interface/app.py
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from tsunami_prediction.serve_api import (
    predict_tectonic_stacking,
    predict_volcanic_stacking,
)
from tsunami_prediction.schemas import TectonicRequest, VolcanicRequest

app = FastAPI(
    title="Tsunami Disaster Prediction API",
    description="API for predicting tsunamigenic events using stacking ensemble ML.",
    version="2.0",
    docs_url="/docs",
)

# --- CORS (dev) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static landing (index.html optional) ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/healthz", tags=["Health"])
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, tags=["Landing"])
async def main():
    """
    Serve interface/static/index.html kalau ada.
    """
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse("<h1>Tsunami Prediction API</h1><p>Service is running.</p>", status_code=200)


# ─────────────────────────────────────────
# 1) Predict TECTONIC
# ─────────────────────────────────────────
@app.post(
    "/v1/predict/tectonic",
    tags=["Prediction"],
    summary="Predict Tsunami from Tectonic Earthquake Data",
)
def predict_tectonic(req: TectonicRequest):
    if not req.datas:
        raise HTTPException(status_code=400, detail="Input 'datas' is empty.")

    df_input = pd.DataFrame(
        [
            {
                "mag": i.mag,
                "depth": i.depth,
                "latitude": i.latitude,
                "longitude": i.longitude,
                "country": i.country,
                "zone": getattr(i, "zone", None),
                "distance_to_coast_km": i.distance_to_coast_km,
                "is_subduction_zone": i.is_subduction_zone,
            }
            for i in req.datas
        ]
    )

    try:
        preds = predict_tectonic_stacking(df_input)
    except FileNotFoundError as e:
        # artefak model belum ada
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error (tectonic): {e}")

    if preds is None or preds.empty:
        raise HTTPException(status_code=500, detail="Empty prediction result.")

    return {
        "predictions": [int(p) for p in preds["prediction"]],
        "probabilities": [float(round(p, 4)) for p in preds["probability"]],
    }


# ─────────────────────────────────────────
# 2) Predict VOLCANIC
# ─────────────────────────────────────────
@app.post(
    "/v1/predict/volcanic",
    tags=["Prediction"],
    summary="Predict Tsunami from Volcanic Eruption Data",
)
def predict_volcanic(req: VolcanicRequest):
    if not req.datas:
        raise HTTPException(status_code=400, detail="Input 'datas' is empty.")

    df_input = pd.DataFrame(
        [
            {
                "eq": i.eq,
                "elevation": i.elevation,
                "latitude": i.latitude,
                "longitude": i.longitude,
                "country": i.country,
                "type": i.type,
                "vei": i.vei,
                "distance_to_coast_km": i.distance_to_coast_km,
                "is_subduction_zone": i.is_subduction_zone,
            }
            for i in req.datas
        ]
    )

    try:
        preds = predict_volcanic_stacking(df_input)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error (volcanic): {e}")

    if preds is None or preds.empty:
        raise HTTPException(status_code=500, detail="Empty prediction result.")

    return {
        "predictions": [int(p) for p in preds["prediction"]],
        "probabilities": [float(round(p, 4)) for p in preds["probability"]],
    }


if __name__ == "__main__":
    # 0.0.0.0 = listen all interfaces; bukanya di browser pakai http://127.0.0.1:8000
    import uvicorn
    uvicorn.run("interface.app:app", host="0.0.0.0", port=8000, reload=True)