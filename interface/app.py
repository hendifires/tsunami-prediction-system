# interface/app.py

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import pandas as pd

from tsunami_prediction.serve_api import (
    predict_tectonic_stacking,
    predict_volcanic_stacking,
)
# Lebih portable (tanpa prefix "src.")
from tsunami_prediction.schemas import (
    TectonicRequest,
    VolcanicRequest,
)

app = FastAPI(
    title="Tsunami Disaster Prediction API",
    description="API for predicting tsunamigenic events from tectonic and volcanic data using stacking ensemble ML.",
    version="2.0",
    docs_url="/docs"
)

# --- CORS (dev) ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*"  # DEV ONLY
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static / Landing page ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse, tags=["Landing"])
async def main():
    index_html = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_html):
        with open(index_html, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(
        content="<h1>Tsunami Prediction API</h1><p>Service is running.</p>",
        status_code=200
    )


# ─────────────────────────────────────────────
# 1) Predict TECTONIC
# ─────────────────────────────────────────────
@app.post(
    "/v1/predict/tectonic",
    tags=["Prediction"],
    summary="Predict Tsunami from Tectonic Earthquake Data",
)
def predict_tectonic(req: TectonicRequest):
    """
    Input: list of tectonic events
    Output: prediction (0/1) + probability
    """
    try:
        if not req.datas:
            raise HTTPException(status_code=400, detail="Input 'datas' is empty.")

        df_input = pd.DataFrame([
            {
                "mag": item.mag,
                "depth": item.depth,
                "latitude": item.latitude,
                "longitude": item.longitude,
                "country": item.country,
                "is_subduction_zone": item.is_subduction_zone,
                "distance_to_coast_km": item.distance_to_coast_km,
                "zone": getattr(item, "zone", None),
            }
            for item in req.datas
        ])

        preds = predict_tectonic_stacking(df_input)
        return {
            "predictions": [int(p) for p in preds["prediction"]],
            "probabilities": [float(round(p, 4)) for p in preds["probability"]],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error (tectonic): {str(e)}")


# ─────────────────────────────────────────────
# 2) Predict VOLCANIC
# ─────────────────────────────────────────────
@app.post(
    "/v1/predict/volcanic",
    tags=["Prediction"],
    summary="Predict Tsunami from Volcanic Eruption Data",
)
def predict_volcanic(req: VolcanicRequest):
    """
    Input: list of volcanic events
    Output: prediction (0/1) + probability
    """
    try:
        if not req.datas:
            raise HTTPException(status_code=400, detail="Input 'datas' is empty.")

        df_input = pd.DataFrame([
            {
                "eq": item.eq,
                "elevation": item.elevation,
                "latitude": item.latitude,
                "longitude": item.longitude,
                "country": item.country,
                "type": item.type,
                "vei": item.vei,
                "distance_to_coast_km": item.distance_to_coast_km,
                "is_subduction_zone": item.is_subduction_zone,
            }
            for item in req.datas
        ])

        preds = predict_volcanic_stacking(df_input)
        return {
            "predictions": [int(p) for p in preds["prediction"]],
            "probabilities": [float(round(p, 4)) for p in preds["probability"]],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error (volcanic): {str(e)}")


# --- MAIN run for local ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("interface.app:app", host="0.0.0.0", port=8000, reload=True)