# scripts/make_processed.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]      # .. (folder proyek)
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

def copy_dataset(name: str):
    src = RAW / f"{name}.csv"
    dst = PROCESSED / f"{name}.csv"
    if not src.exists():
        raise FileNotFoundError(f"tidak ditemukan: {src}")
    df = pd.read_csv(src)
    # di sini nanti boleh tambah cleaning ringan
    df.to_csv(dst, index=False)
    print(f"[OK] {name}: {src.name} -> {dst.relative_to(ROOT)} (rows={len(df)})")

def main():
    for ds in ["tectonic", "volcanic"]:
        copy_dataset(ds)

if __name__ == "__main__":
    main()