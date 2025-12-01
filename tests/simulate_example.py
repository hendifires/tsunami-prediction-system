from pathlib import Path
import pandas as pd
from joblib import load

# Path dasar
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
ART  = ROOT / "artifacts"

# ... dst (logika lama tetap)

# 1) Load model stacking terbaik (SMOTE, window terakhir yang kamu run_all)
model_path = ART / "events_smote_stacking.joblib"
model = load(model_path)
print(f"Loaded model from: {model_path}")

# 2) Ambil data test yang sudah dipreproses
test_path = DATA / "events_test.csv"
df_test = pd.read_csv(test_path)
print(f"Loaded test data from: {test_path}, shape={df_test.shape}")

# Pisahkan fitur & label
X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

# 3) Pilih satu contoh event untuk simulasi
idx = 0  # ganti ke indeks lain kalau mau
x_sample = X_test.iloc[[idx]]   # tetap DataFrame 1 baris
y_true = int(y_test.iloc[idx])

# 4) Prediksi kelas & probabilitas
y_pred = int(model.predict(x_sample)[0])
proba = model.predict_proba(x_sample)[0]  # array [p0, p1, p2]

label_map = {
    0: "Non-tsunami",
    1: "Tsunami tektonik",
    2: "Tsunami vulkanik",
}

print("\n=== Simulasi 1 event ===")
print("Fitur event (ringkas):")
print(x_sample.head())

print(f"\nLabel sebenarnya (y_true) : {y_true} -> {label_map[y_true]}")
print(f"Label prediksi (y_pred)   : {y_pred} -> {label_map[y_pred]}")
print(f"Probabilitas kelas        :")
print(f"  P(non-tsu)   = {proba[0]:.3f}")
print(f"  P(tektonik)  = {proba[1]:.3f}")
print(f"  P(vulkanik)  = {proba[2]:.3f}")