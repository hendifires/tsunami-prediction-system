import pandas as pd
import joblib

# 14.1.1 MAIN FUNCTION FOR PREDICTING TECTONIC TSUNAMI POTENTIAL
def predict_tectonic_stacking(
    df_new,
    model_path="models/stacking_tectonic.joblib",
    scaler_path="models/scaler_tectonic.joblib",
    encoder_path="models/encoder_tectonic.joblib"
):
    """
    Predict tsunami potential for tectonic earthquakes using a Stacking Ensemble model.
    Input MUST include all required features used in model training.
    Returns DataFrame with prediction label and probability.
    """
    # --- Standardize column names to lowercase for safety
    df_new = df_new.copy()
    df_new.columns = [c.lower() for c in df_new.columns]

    # --- List all required columns (SYNC with training pipeline!)
    required_cols = [
        'mag',
        'depth',
        'latitude',
        'longitude',
        'zone',                  # categorical: OHE or OrdinalEncoder
        'distance_to_coast_km',  # numeric
        'is_subduction_zone'     # binary (0/1)
    ]

    # --- Check missing features
    missing_cols = [col for col in required_cols if col not in df_new.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")

    # --- Select and order columns
    X_new = df_new[required_cols]

    # --- Load pipeline components
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    model   = joblib.load(model_path)

    # --- Apply encoder & scaler
    X_encoded = encoder.transform(X_new)
    X_scaled  = scaler.transform(X_encoded)

    # --- Predict
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    # --- Output result
    result = df_new.copy()
    result['prediction'] = preds
    result['probability'] = probs
    return result

# 14.1.2 MAIN FUNCTION FOR PREDICTING VOLCANIC TSUNAMI POTENTIAL
def predict_volcanic_stacking(
    df_new,
    model_path="models/stacking_volcanic.joblib",
    scaler_path="models/scaler_volcanic.joblib",
    encoder_path="models/encoder_volcanic.joblib"
):
    """
    Predict tsunami potential for volcanic eruptions using a Stacking Ensemble model.
    Input MUST include all required features used in model training.
    Returns DataFrame with prediction label and probability.
    """
    df_new = df_new.copy()
    df_new.columns = [c.lower() for c in df_new.columns]

    required_cols = [
        'eq',
        'elevation',
        'latitude',
        'longitude',
        'type',                  # categorical: OHE or OrdinalEncoder
        'vei',
        'distance_to_coast_km',  # numeric
        'is_subduction_zone'     # binary (0/1)
    ]

    missing_cols = [col for col in required_cols if col not in df_new.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")

    X_new   = df_new[required_cols]
    encoder = joblib.load(encoder_path)
    scaler  = joblib.load(scaler_path)
    model   = joblib.load(model_path)

    X_encoded = encoder.transform(X_new)
    X_scaled  = scaler.transform(X_encoded)
    preds     = model.predict(X_scaled)
    probs     = model.predict_proba(X_scaled)[:, 1]

    result = df_new.copy()
    result['prediction'] = preds
    result['probability'] = probs
    return result
