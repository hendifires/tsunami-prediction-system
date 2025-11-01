# src/tsunami_prediction/predict.py

import os
import joblib
import pandas as pd

ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"

def _load_artifact(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def predict_tectonic_stacking(df: pd.DataFrame):
    """
    df: kolom2 mentah dari API
    return: dict {"prediction": y_pred, "probability": y_proba}
    """
    model_path = os.path.join(MODEL_DIR, "stacking_tectonic.joblib")
    enc_path   = os.path.join(ARTIFACT_DIR, "encoder_tectonic.joblib")
    scaler_path= os.path.join(ARTIFACT_DIR, "scaler_tectonic.joblib")
    feat_path  = os.path.join(ARTIFACT_DIR, "feature_cols_tectonic.joblib")

    model = _load_artifact(model_path)
    encoder = _load_artifact(enc_path)
    scaler = _load_artifact(scaler_path)
    feature_cols = _load_artifact(feat_path)

    if model is None or feature_cols is None:
        # biar app.py bisa ngasih error yang ramah
        return None

    # --- preprocessing ringkas (ini nanti kita ganti dgn preprocessing.py) ---
    if encoder is not None and "country" in df.columns:
        df[["country"]] = encoder.transform(df[["country"]])

    X = df.reindex(columns=feature_cols, fill_value=0)

    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {"prediction": pred, "probability": proba}


def predict_volcanic_stacking(df: pd.DataFrame):
    model_path = os.path.join(MODEL_DIR, "stacking_volcanic.joblib")
    enc_path   = os.path.join(ARTIFACT_DIR, "encoder_volcanic.joblib")
    scaler_path= os.path.join(ARTIFACT_DIR, "scaler_volcanic.joblib")
    feat_path  = os.path.join(ARTIFACT_DIR, "feature_cols_volcanic.joblib")

    model = _load_artifact(model_path)
    encoder = _load_artifact(enc_path)
    scaler = _load_artifact(scaler_path)
    feature_cols = _load_artifact(feat_path)

    if model is None or feature_cols is None:
        return None

    if encoder is not None and "type" in df.columns:
        df[["type"]] = encoder.transform(df[["type"]])

    X = df.reindex(columns=feature_cols, fill_value=0)

    if scaler is not None:
        X = scaler.transform(X)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {"prediction": pred, "probability": proba}