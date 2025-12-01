from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# === FIXED PATH ===
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"


def analyze_model(prefix: str) -> None:
    """
    Analisis detail hasil prediksi untuk satu model.

    Parameter
    ---------
    prefix : str
        Prefix nama file prediksi (tanpa akhiran '_preds.csv'), misalnya:
          - 'events_smote_stacking'    -> file: events_smote_stacking_preds.csv
          - 'events_nosmote_stacking'  -> file: events_nosmote_stacking_preds.csv
          - 'events_smote_rf'          -> file: events_smote_rf_preds.csv (jika ada)

    File yang dibaca harus berada di:
        reports/tables/{prefix}_preds.csv
    dan memiliki kolom:
        - y_true
        - y_pred
    """
    p = TAB / f"{prefix}_preds.csv"

    if not p.exists():
        raise FileNotFoundError(f"File prediksi tidak ditemukan: {p}")

    df = pd.read_csv(p)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise KeyError(
            f"File {p} tidak memiliki kolom 'y_true' dan 'y_pred'. "
            f"Kolom yang tersedia: {list(df.columns)}"
        )

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    print(f"\n========== {prefix} ==========")
    print("Classification report (0=non,1=tektonik,2=vulkanik):")
    print(classification_report(y_true, y_pred, digits=3))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    # Analisis default: model stacking dengan SMOTE (model utama tesis)
    analyze_model("events_smote_stacking")

    # Jika ingin menganalisis baseline tanpa SMOTE, bisa jalankan:
    # analyze_model("events_nosmote_stacking")