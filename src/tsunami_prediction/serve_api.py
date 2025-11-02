# src/tsunami_prediction/serve_api.py
from __future__ import annotations

# cSpell:ignore nosmote joblib cand glob

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load


# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


# ---------------- Utilities ----------------
_BAD_CHARS = {
    "[": "(", "]": ")", "<": "_lt_", ">": "_gt_",
    "{": "(", "}": ")", "/": "_", "\\": "_",
    ":": "_", ";": "_", ",": "_", "=": "_",
}

def _safe_name(name: object) -> str:
    s = str(name)
    for k, v in _BAD_CHARS.items():
        s = s.replace(k, v)
    return s.replace(" ", "_")

def _to_float_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna(0.0)

def _find_artifact_path(dataset: str) -> Path:
    """
    Prefer SMOTE model if exists, else nosmote; fallback ke file apa pun yang cocok.
    """
    cand = [
        ART / f"{dataset}_stack_smote.joblib",
        ART / f"{dataset}_stack_nosmote.joblib",
    ]
    for p in cand:
        if p.exists():
            return p
    # fallback: first any matching file (pakai walrus untuk hilangkan warning Sourcery)
    if any_files := list(ART.glob(f"{dataset}_stack_*.joblib")):
        return any_files[0]
    raise FileNotFoundError(
        f"Model artifact for dataset='{dataset}' not found in {ART}. "
        f"Train with stacking_pipeline.py first."
    )

def _load_artifact(path: Path) -> Dict:
    obj = joblib_load(path)
    if "model" not in obj or "feature_columns" not in obj:
        raise ValueError(f"Invalid artifact content in {path.name}.")
    return obj

def _align_input_to_features(
    df_new: pd.DataFrame,
    artifact: Dict
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    1) lowercase kolom, 2) sanitasi, 3) rename sesuai mapping training,
    4) reindex ke urutan fitur training; kolom hilang -> 0.
    """
    df = df_new.copy()
    df.columns = [str(c).lower() for c in df.columns]
    df.columns = [_safe_name(c) for c in df.columns]

    mapping = artifact.get("col_name_map") or {}
    ci_map = {str(k).lower(): str(v) for k, v in mapping.items()}
    # dict-comprehension untuk hilangkan warning Sourcery
    rename_map = {c: ci_map.get(c.lower(), c) for c in df.columns}
    df = df.rename(columns=rename_map)

    # pilih selected_columns kalau ada, selain itu pakai feature_columns
    feat_cols = (sel_cols if (sel_cols := artifact.get("selected_columns")) else artifact["feature_columns"])

    X = df[feat_cols]
    X = _to_float_df(X)
    return X, pd.Index(feat_cols)

def _predict_with_artifact(
    df_new: pd.DataFrame,
    dataset: str,
    artifact_path: Optional[str] = None
) -> pd.DataFrame:
    path = Path(artifact_path) if artifact_path else _find_artifact_path(dataset)
    art = _load_artifact(path)

    X, _ = _align_input_to_features(df_new, art)
    model = art["model"]

    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        y_prob = proba[:, 1] if proba.ndim == 2 else proba
    elif hasattr(model, "decision_function"):
        z = np.clip(model.decision_function(X), -20, 20)
        y_prob = 1.0 / (1.0 + np.exp(-z))
    else:
        y_prob = np.asarray(y_pred, dtype=float)

    out = df_new.copy()
    out["prediction"] = np.asarray(y_pred, dtype=int)
    out["probability"] = np.asarray(y_prob, dtype=float)
    return out

def predict_tectonic_stacking(
    df_new: pd.DataFrame,
    artifact_path: Optional[str] = None
) -> pd.DataFrame:
    if not isinstance(df_new, pd.DataFrame):
        raise TypeError("df_new must be a pandas DataFrame.")
    if df_new.empty:
        raise ValueError("df_new is empty.")
    return _predict_with_artifact(df_new, dataset="tectonic", artifact_path=artifact_path)

def predict_volcanic_stacking(
    df_new: pd.DataFrame,
    artifact_path: Optional[str] = None
) -> pd.DataFrame:
    if not isinstance(df_new, pd.DataFrame):
        raise TypeError("df_new must be a pandas DataFrame.")
    if df_new.empty:
        raise ValueError("df_new is empty.")
    return _predict_with_artifact(df_new, dataset="volcanic", artifact_path=artifact_path)