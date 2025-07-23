from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import os
import pandas as pd

from src.predict import predict_tectonic_stacking, predict_volcanic_stacking
from pred_type.types import TektonikRequest, VulRequest

app = FastAPI(
    title="Tsunami Disaster Prediction API",
    description="API for predicting tsunamigenic events from tectonic and volcanic data using stacking ensemble machine learning.",
    version="1.1",
    docs_url="/docs"
)

# --- CORS setup for local/frontend access (Edit for production!) ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",  # For dev/test only, NOT for production!
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static HTML (Landing Page) ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse, tags=["Landing"])
async def main():
    html_path = os.path.join("static", "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse(content="<h1>Tsunami Disaster Prediction API</h1><p>API is running.</p>", status_code=200)
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# --- ENDPOINT: Predict Tsunami Potential from Tectonic Earthquake ---
@app.post("/v1/predict/tectonic", tags=["Prediction"], summary="Predict Tsunami from Tectonic Earthquake Data")
def predict_tectonic(request: TektonikRequest):
    """
    Predict tsunami potential for tectonic earthquakes using stacking ensemble ML.
    - **mag**: Magnitude (float)
    - **depth**: Focal Depth (km) (float)
    - **latitude**: (float)
    - **longitude**: (float)
    - **country**: (str)
    - **is_subduction_zone**: (int, 1=subduction, 0=non-subduction)
    - **distance_to_coast_km**: (float)
    - **zone**: Subduction type/categorical (str, optional)
    """
    try:
        df_input = pd.DataFrame([
            {
                "mag": s.mag,
                "depth": s.depth,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "country": s.country,
                "is_subduction_zone": s.is_subduction_zone,
                "distance_to_coast_km": s.distance_to_coast_km,
                "zone": getattr(s, "zone", None)  # zone opsional
            }
            for s in request.datas
        ])
        preds = predict_tectonic_stacking(df_input)
        return {
            "predictions": [int(p) for p in preds["prediction"]],
            "probabilities": [float(round(p, 3)) for p in preds["probability"]]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# --- ENDPOINT: Predict Tsunami Potential from Volcanic Eruption ---
@app.post("/v1/predict/volcanic", tags=["Prediction"], summary="Predict Tsunami from Volcanic Eruption Data")
def predict_volcanic(request: VulRequest):
    """
    Predict tsunami potential for volcanic eruptions using stacking ensemble ML.
    - **eq**: Earthquake Magnitude (float)
    - **elevation**: Elevation (m) (float)
    - **latitude**: (float)
    - **longitude**: (float)
    - **country**: (str)
    - **type**: Volcano type (categorical, str)
    - **vei**: Volcanic Explosivity Index (int/float)
    - **distance_to_coast_km**: (float)
    - **is_subduction_zone**: (int, 1=subduction, 0=non-subduction)
    """
    try:
        df_input = pd.DataFrame([
            {
                "eq": s.eq,
                "elevation": s.elevation,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "country": s.country,
                "type": s.type,
                "vei": s.vei,
                "distance_to_coast_km": s.distance_to_coast_km,
                "is_subduction_zone": s.is_subduction_zone,
            }
            for s in request.datas
        ])
        preds = predict_volcanic_stacking(df_input)
        return {
            "predictions": [int(p) for p in preds["prediction"]],
            "probabilities": [float(round(p, 3)) for p in preds["probability"]]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# --- MAIN RUN ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
