# src/tsunami_prediction/serve_api.py
from __future__ import annotations
# cSpell:ignore writeable subduction Vanuatu Tonga Nicaragua Rica nosmote joblib

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import types
import sys

import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ============================== PATHS ==============================
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


# =================== Compat for pickled FunctionTransformer ===================
def to_np_writable(X):
    """Helper yang dipakai saat training; harus tersedia saat unpickle."""
    arr = np.asarray(X, dtype=float)
    if not getattr(arr, "flags", None) or not arr.flags.writeable:
        arr = arr.copy()
    return arr


def _ensure_unpickle_compat() -> None:
    """Pastikan simbol to_np_writable bisa diimpor oleh joblib saat unpickle."""
    # __main__
    main_mod = sys.modules.get("__main__") or types.ModuleType("__main__")
    sys.modules["__main__"] = main_mod
    setattr(main_mod, "to_np_writable", to_np_writable)

    # modul training
    try:
        import tsunami_prediction.stacking_pipeline as _  # noqa: F401
    except Exception:
        sp = types.ModuleType("tsunami_prediction.stacking_pipeline")
        sys.modules["tsunami_prediction.stacking_pipeline"] = sp
    sys.modules["tsunami_prediction.stacking_pipeline"].to_np_writable = to_np_writable  # type: ignore[attr-defined]


# ============================== Small FE ==============================
_BAD_CHARS = {
    "[": "(", "]": ")", "<": "_lt_", ">": "_gt_", "{": "(", "}": ")",
    "/": "_", "\\": "_", ":": "_", ";": "_", ",": "_", "=": "_"
}


def _safe_name(s: object) -> str:
    t = str(s)
    for k, v in _BAD_CHARS.items():
        t = t.replace(k, v)
    return t.replace(" ", "_")


_SUBDUCTION = {
    "Indonesia", "Japan", "Philippines", "Taiwan", "Papua New Guinea", "Solomon Islands",
    "Vanuatu", "Tonga", "New Zealand", "Fiji", "United States", "Canada", "Mexico",
    "Guatemala", "El Salvador", "Honduras", "Nicaragua", "Costa Rica", "Panama", "Colombia",
    "Ecuador", "Peru", "Chile", "Russia", "Greece", "Argentina", "Bolivia",
}
_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "myanmar (burma)": "Myanmar",
    "burma": "Myanmar",
    "russian federation": "Russia",
}


def _norm_country(x: object) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    low = s.lower()
    return _ALIASES.get(low, s.title())


def _coerce_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _bool01(s: pd.Series) -> pd.Series:
    v = s.astype(str).str.strip().str.lower()
    return v.isin(["1", "true", "yes", "y"]).astype(float)


def _fe_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "latitude" in out.columns:
        out["abs_lat"] = _coerce_float(out["latitude"]).abs()
        out["is_tropic"] = (out["abs_lat"] <= 23.5).astype("Int64")
    if {"latitude", "longitude"}.issubset(out.columns):
        out["lat_lon_prod"] = _coerce_float(out["latitude"]) * _coerce_float(out["longitude"])
    return out


def _add_subduction_if_missing(out: pd.DataFrame) -> pd.DataFrame:
    if "is_subduction_zone" in out.columns:
        return out
    src = "country_norm" if "country_norm" in out.columns else ("country" if "country" in out.columns else None)
    if src is not None:
        out = out.copy()
        out["country_norm"] = out[src].astype("object").map(_norm_country)
        out["is_subduction_zone"] = out["country_norm"].isin(_SUBDUCTION).astype("Int64")
    return out


def _fe_tectonic(df: pd.DataFrame) -> pd.DataFrame:
    out = _fe_common(df)
    out = _add_subduction_if_missing(out)
    if "depth" in out.columns:
        out["depth_log1p"] = np.log1p(_coerce_float(out["depth"]).clip(lower=0))
    if "mag" in out.columns:
        out["mag_sq"] = _coerce_float(out["mag"]) ** 2
    if {"mag", "depth"}.issubset(out.columns):
        out["mag_over_depth1p"] = _coerce_float(out["mag"]) / (_coerce_float(out["depth"]).abs() + 1.0)
    return out


def _fe_volcanic(df: pd.DataFrame) -> pd.DataFrame:
    out = _fe_common(df)
    out = _add_subduction_if_missing(out)
    if "elevation" in out.columns:
        out["elev_log1p"] = np.log1p(_coerce_float(out["elevation"]).clip(lower=0))
    if "vei" in out.columns:
        out["vei_sq"] = _coerce_float(out["vei"]) ** 2
    if "eq" in out.columns:
        out["eq_log1p"] = np.log1p(_coerce_float(out["eq"]).clip(lower=0))
    if {"vei", "elevation"}.issubset(out.columns):
        out["vei_x_elev"] = _coerce_float(out["vei"]) * _coerce_float(out["elevation"])
    return out


# =========================== Artifact helpers ===========================
def _preferred_artifacts(dataset: str) -> List[Path]:
    d = dataset.lower()
    return {
        "tectonic": [
            ART / "tectonic_stack_nosmote.joblib",
            ART / "tectonic_stack_smote.joblib",
            ART / "tectonic_stack_smote_tomek.joblib",
            ART / "tectonic_stack_smote_enn.joblib",
        ],
        "volcanic": [
            ART / "volcanic_stack_smote_enn.joblib",
            ART / "volcanic_stack_smote_tomek.joblib",
            ART / "volcanic_stack_smote.joblib",
            ART / "volcanic_stack_nosmote.joblib",
        ],
    }.get(d, [])


def _find_artifact_path(dataset: str) -> Path:
    for p in _preferred_artifacts(dataset):
        if p.exists():
            return p
    if (any_files := list(ART.glob(f"{dataset}_stack_*.joblib"))):
        return any_files[0]
    raise FileNotFoundError(
        f"Model artifact for dataset='{dataset}' not found in {ART}. "
        "Train with stacking_pipeline.py first."
    )


def _load_artifact(path: Path) -> Dict:
    _ensure_unpickle_compat()
    obj = joblib_load(path)
    if "model" not in obj or "feature_columns" not in obj:
        raise ValueError(f"Invalid artifact content in {path.name}.")
    return obj


# ============ Build X persis seperti saat training (align fitur) ============
def _inject_onehot_rowwise(
    X: pd.DataFrame,
    df_src: pd.DataFrame,
    src_col: str,
    prefix: str,
) -> None:
    if src_col not in df_src.columns:
        return
    vals = df_src[src_col].astype(str).fillna("").map(_safe_name).tolist()
    for i, v in enumerate(vals):
        col = f"{prefix}_{v}"
        if col in X.columns:
            X.iat[i, X.columns.get_loc(col)] = 1.0


def _align_input_to_features(
    df_in: pd.DataFrame,
    artifact: Dict,
    dataset: str,
) -> Tuple[pd.DataFrame, pd.Index]:
    if not isinstance(df_in, pd.DataFrame) or df_in.empty:
        raise ValueError("df_in must be a non-empty DataFrame.")

    # normalisasi kolom
    df = df_in.copy()
    df.columns = [_safe_name(str(c).lower()) for c in df.columns]

    # small FE seperti di training
    df = _fe_tectonic(df) if dataset.lower() == "tectonic" else _fe_volcanic(df)

    # mapping nama kolom dari artifact (hasil sanitize training)
    if (mapping := artifact.get("col_name_map") or {}):
        lower_map = {str(k).lower(): str(v) for k, v in mapping.items()}
        df = df.rename(columns={c: lower_map.get(c.lower(), c) for c in df.columns})

    feat_cols = list(artifact.get("selected_columns") or artifact["feature_columns"])
    X = pd.DataFrame(0.0, index=df.index, columns=feat_cols, dtype=float)

    # numeric & boolean
    numeric_like = [
        "mag", "depth", "latitude", "longitude", "eq", "elevation", "vei",
        "distance_to_coast_km", "is_subduction_zone",
        # engineered
        "abs_lat", "is_tropic", "lat_lon_prod",
        "depth_log1p", "mag_sq", "mag_over_depth1p",
        "elev_log1p", "vei_sq", "eq_log1p", "vei_x_elev",
    ]
    for c in numeric_like:
        if c in df.columns and c in X.columns:
            X[c] = _bool01(df[c]) if c == "is_subduction_zone" else _coerce_float(df[c])

    # categorical one-hot ringan — hanya aktifkan kolom yang memang ada di model
    base_cat = "country_norm" if "country_norm" in df.columns else ("country" if "country" in df.columns else None)
    if base_cat is not None:
        _inject_onehot_rowwise(X, df, base_cat, "country")
    if "zone" in df.columns:
        _inject_onehot_rowwise(X, df, "zone", "zone")
    if "type" in df.columns:
        _inject_onehot_rowwise(X, df, "type", "type")

    return X, X.columns


# ================================ Predict ================================
def _predict_with_artifact(
    df_new: pd.DataFrame,
    dataset: str,
    artifact_path: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(artifact_path) if artifact_path else _find_artifact_path(dataset)
    art = _load_artifact(path)

    X, _ = _align_input_to_features(df_new, art, dataset)
    model = art["model"]

    # probabilitas/skor
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        y_prob = proba[:, 1] if proba.ndim == 2 else proba
    elif hasattr(model, "decision_function"):
        z = np.clip(model.decision_function(X), -20, 20)
        y_prob = 1.0 / (1.0 + np.exp(-z))
    else:
        y_prob = model.predict(X).astype(float)

    # threshold (bila disimpan di artifact saat training)
    thr = float(art.get("decision_threshold", 0.5))
    y_pred = (np.asarray(y_prob) >= thr).astype(int)

    out = df_new.copy()
    out["prediction"] = y_pred
    out["probability"] = np.asarray(y_prob, dtype=float)
    return out


# ============================== Public API ==============================
def predict_tectonic_stacking(
    df_new: pd.DataFrame,
    artifact_path: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(df_new, pd.DataFrame) or df_new.empty:
        raise ValueError("df_new must be a non-empty DataFrame.")
    return _predict_with_artifact(df_new, dataset="tectonic", artifact_path=artifact_path)


def predict_volcanic_stacking(
    df_new: pd.DataFrame,
    artifact_path: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(df_new, pd.DataFrame) or df_new.empty:
        raise ValueError("df_new must be a non-empty DataFrame.")
    return _predict_with_artifact(df_new, dataset="volcanic", artifact_path=artifact_path)