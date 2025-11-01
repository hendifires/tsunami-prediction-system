# tests/test_pipeline.py
import os
import sys
sys.path.insert(0, 'src')
import joblib
import pandas as pd
import numpy as np
from tsunami_prediction.smote_pipeline import apply_smote_train, get_preprocessor
from tsunami_prediction.stacking_pipeline import train_stacking, get_default_base_models
from tsunami_prediction.run_experiment import run_experiment

def sample_data(n=300):
    df = pd.read_csv('dataset/processed/tectonic_cleaned.csv').dropna().sample(n, random_state=42).reset_index(drop=True)
    return df

def test_smote_only_on_train():
    df = sample_data(200)
    X = df[['mag','depth','latitude','longitude']]
    y = df['tsu'].astype(int)
    # split
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    tr_idx, val_idx = next(iter(skf.split(X, y)))
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_tr_res, y_tr_res, preproc = apply_smote_train(X_tr, y_tr, numeric_cols=list(X_tr.columns), categorical_cols=[])
    # assert balanced
    vals = y_tr_res.value_counts().to_list()
    assert len(vals) == 2 and vals[0] == vals[1]
    # ensure val unchanged
    assert y_val.value_counts().to_dict() == y.iloc[val_idx].value_counts().to_dict()

def test_oof_shape():
    df = sample_data(300)
    X = df[['mag','depth','latitude','longitude']]
    y = df['tsu'].astype(int)
    base_models = get_default_base_models()
    model_dict = train_stacking(X, y, base_models=base_models, cv=3)
    oof = model_dict['oof_features']
    assert oof.shape[1] == len(base_models)
    assert oof.shape[0] == X.shape[0]

def test_run_experiment_creates_metrics(tmp_path):
    out = tmp_path / "test_run"
    out_str = str(out)
    res = run_experiment('dataset/processed/tectonic_cleaned.csv', target='tsu', use_smote=False, kfold=3, out_dir=out_str)
    assert os.path.exists(res['metrics_per_fold'])
    df = pd.read_csv(res['metrics_per_fold'])
    assert 'model' in df.columns
