"""
smote_pipeline.py

Tanggung jawab:
- Semua transformasi/fit yang *hanya* boleh dilakukan pada training-split.
- Menyediakan fungsi aplikasi SMOTE yang mengembalikan X_res, y_res, dan preprocessor (encoder+scaler)
- Menyediakan factory untuk mendapatkan imblearn Pipeline jika diinginkan

Important:
- Jangan panggil apply_smote_train() pada seluruh dataset sebelum split.
- Preprocessor yang dikembalikan dapat digunakan untuk mentransform data validasi/test (transform saja).
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
import joblib


def get_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scaler: Optional[object]=None, encoder: Optional[object]=None):
    """Return ColumnTransformer fitted later on train set.

    Caller should fit the returned transformer on X_train only.
    """
    if scaler is None:
        scaler = StandardScaler()
    if encoder is None:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    transformers = []
    if numeric_cols:
        transformers.append(("num", scaler, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", encoder, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    return preprocessor


def apply_smote_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    random_state: int = 42,
    sampling_strategy='auto'
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Fit preprocessor on X_train (train-fold) and apply SMOTE.

    Returns:
        X_res (pd.DataFrame): DataFrame setelah preprocessor transform & SMOTE (features as columns)
        y_res (pd.Series): target array after resampling
        preprocessor (ColumnTransformer): fitted transformer to be used on validation/test

    Important: Preprocessor is FIT ON TRAIN ONLY. Use preprocessor.transform(X_val) to transform val/test.
    """
    # Defensive copy
    X_train = X_train.copy()
    y_train = y_train.copy()

    # Build preprocessor and fit on train
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)

    # Transform train to NumPy array
    X_train_trans = preprocessor.transform(X_train)

    # Build feature names after transformation
    feature_names = []
    # numeric names
    if numeric_cols:
        feature_names.extend(numeric_cols)
    # categorical get_feature_names_out (if exists)
    if categorical_cols:
        # We need to extract encoder object
        # the ColumnTransformer stores transformers_ after fit
        # find the cat transformer
        for name, trans, cols in preprocessor.transformers_:
            if name == 'cat':
                try:
                    cat_names = trans.get_feature_names_out(cols)
                except Exception:
                    # fallback
                    cat_names = [f"{c}_{i}" for c in cols for i in range(1)]
                feature_names.extend(list(cat_names))

    # Apply SMOTE
    smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
    X_res_np, y_res = smote.fit_resample(X_train_trans, y_train.values)

    # Convert back to DataFrame with feature names
    X_res = pd.DataFrame(X_res_np, columns=feature_names)
    y_res = pd.Series(y_res, name=y_train.name)

    return X_res, y_res, preprocessor


def get_smote_pipeline(numeric_cols: List[str], categorical_cols: List[str], random_state: int=42, scaler=None, encoder=None):
    """
    Returns an imblearn Pipeline that performs transformation then SMOTE.
    Useful when you want a single pipeline object to fit on training fold.

    NOTE: This pipeline must only be fit on training data.
    """
    preprocessor = get_preprocessor(numeric_cols, categorical_cols, scaler=scaler, encoder=encoder)
    smote = SMOTE(random_state=random_state)
    pipe = ImbPipeline(steps=[('preproc', preprocessor), ('smote', smote)])
    return pipe


# Small utility function to persist preprocessor
def save_preprocessor(preprocessor, out_path: str):
    joblib.dump(preprocessor, out_path)


if __name__ == "__main__":
    print("Module smote_pipeline.py - utilities to apply SMOTE on train fold only.")