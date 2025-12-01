from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"

# 1) Ringkasan semua eksperimen
df_all = pd.read_csv(TAB / "stacking_experiments_all_metrics.csv")
print("\n=== All stacking experiments ===")
print(df_all[["dataset","exp_tag","scenario","model","accuracy","f1_macro","prec_macro","rec_macro","role"]])

# 2) Fokus ke window 2000–2024
df_2000 = df_all[df_all["exp_tag"] == "y2000_2024"]
print("\n=== Window 2000–2024 ===")
print(df_2000[["scenario","model","accuracy","f1_macro","prec_macro","rec_macro","role"]])

# 3) Model utama (role=main)
print("\n=== Main model (role=='main') ===")
print(df_all[df_all["role"].str.contains("main", na=False)])