from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

# ================== PATHS ==================
ROOT = Path(__file__).resolve().parents[2]  # .../tsunami-prediction
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def _load_preprocessed(domain: str) -> pd.DataFrame:
    """
    domain: 'tectonic' atau 'volcanic'
    Baca file preprocessed hasil preprocessing.py
    """
    if domain not in {"tectonic", "volcanic"}:
        raise ValueError("domain harus 'tectonic' atau 'volcanic'")

    path = PROCESSED / f"{domain}_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"File {path} tidak ditemukan. "
            "Pastikan sudah menjalankan preprocessing.py terlebih dahulu."
        )
    return pd.read_csv(path)


def _train_baseline(
    df: pd.DataFrame,
    domain: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, float | int]:
    """
    Latih Logistic Regression sebagai baseline.
    Asumsi:
      - semua kolom kecuali 'tsu' = fitur numerik siap pakai (sudah di-encode & scale)
      - 'tsu' = target biner 0/1
    """

    if "tsu" not in df.columns:
        raise KeyError(f"[{domain}] kolom 'tsu' tidak ditemukan di dataset preprocessed.")

    # buang baris yang masih punya NaN di fitur/target
    df = df.dropna().reset_index(drop=True)

    X = df.drop(columns=["tsu"])
    y = df["tsu"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Logistic Regression dengan class_weight='balanced' supaya aware class imbalance
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # handle kasus probabilitas/ROC yang bisa error kalau hanya satu kelas di y_test
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = np.nan

    metrics: Dict[str, float | int] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "n_samples": len(df),
        "n_test": len(y_test),
        "tsu_rate_overall": y.mean(),
        "tsu_rate_test": y_test.mean(),
    }

    print(f"\n=== {domain.upper()} BASELINE (LogisticRegression) ===")
    print(f"Samples total: {len(df)} | Test: {len(y_test)}")
    print(f"Class balance (overall) tsu=1: {y.mean():.3f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    if np.isnan(roc_auc):
        print("ROC AUC: NaN")
    else:
        print(f"ROC AUC: {roc_auc:.3f}")

    return metrics


def main(domain: str, test_size: float = 0.2, random_state: int = 42) -> None:
    df = _load_preprocessed(domain)
    metrics = _train_baseline(df, domain, test_size=test_size, random_state=random_state)

    # simpan metrics ke CSV (append / update sederhana)
    out_path = TAB / "baseline_results.csv"
    row = {"domain": domain, **metrics}

    if out_path.exists():
        df_old = pd.read_csv(out_path)
        # buang baris lama untuk domain yang sama lalu append baru
        df_old = df_old[df_old["domain"] != domain]
        df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])

    df_new.to_csv(out_path, index=False)
    print(f"\n[INFO] Baseline metrics tersimpan di: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train baseline Logistic Regression untuk tsunami prediction."
    )
    ap.add_argument(
        "--domain",
        choices=["tectonic", "volcanic"],
        required=True,
        help="Pilih domain dataset: 'tectonic' atau 'volcanic'.",
    )
    ap.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data untuk test set (default=0.2).",
    )
    ap.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed untuk split train/test.",
    )
    args = ap.parse_args()

    main(
        domain=args.domain,
        test_size=args.test_size,
        random_state=args.random_state,
    )