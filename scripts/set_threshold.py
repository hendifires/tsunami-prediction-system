from __future__ import annotations

import argparse
from pathlib import Path

import joblib

# Pakai helper yang sama dengan serve_api (agar unpickle aman)
from tsunami_prediction.serve_api import (
    _ensure_unpickle_compat,
    _preferred_artifacts,
)

# ROOT & ART konsisten dengan modul lain (serve_api, stacking_pipeline, dsb.)
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"


def find_artifact(dataset: str) -> Path:
    """
    Cari artifact model stacking untuk dataset tertentu.

    Prioritas:
    - gunakan urutan preferensi di _preferred_artifacts()
      (mis. volcanic: smote -> smote_tomek -> smote_enn -> nosmote)
    - fallback: file pertama yang cocok pola '<dataset>_stack_*.joblib'
    """
    # 1) pakai preferensi sama seperti serve_api
    for p in _preferred_artifacts(dataset):
        if p.exists():
            return p

    # 2) fallback: ambil apa pun yang cocok pola
    if cands := sorted(ART.glob(f"{dataset}_stack_*.joblib")):
        return cands[0]

    raise SystemExit(f"Artifact for dataset='{dataset}' not found in {ART}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Set custom decision threshold for a trained stacking model.\n"
            "Threshold ini akan dipakai API saat mengubah probabilitas → label 0/1."
        )
    )
    ap.add_argument(
        "--dataset",
        choices=["tectonic", "volcanic"],
        required=True,
        help="pilih domain model yang akan diubah threshold-nya",
    )
    ap.add_argument(
        "--thr",
        type=float,
        required=True,
        help="nilai threshold baru (0–1), misal 0.5 atau 0.88",
    )
    ap.add_argument(
        "--path",
        type=str,
        help="opsional: path artifact eksplisit (override auto-discovery)",
    )
    args = ap.parse_args()

    if not (0.0 <= args.thr <= 1.0):
        raise SystemExit("ERROR: --thr harus di antara 0.0 dan 1.0")

    # pastikan helper to_np_writable dll tersedia saat unpickle
    _ensure_unpickle_compat()

    path = Path(args.path) if args.path else find_artifact(args.dataset)

    model_dict = joblib.load(path)
    if not isinstance(model_dict, dict) or "model" not in model_dict:
        raise SystemExit(f"File {path} bukan artifact stacking yang valid.")

    old_thr = float(model_dict.get("decision_threshold", 0.5))
    model_dict["decision_threshold"] = float(args.thr)

    # simpan kembali dengan kompresi ringan
    joblib.dump(model_dict, path, compress=3)

    meta = model_dict.get("meta", {})
    variant = meta.get("smote_variant") or meta.get("variant") or ""
    print(
        f"OK: {args.dataset} "
        f"(variant='{variant or 'auto'}') "
        f"threshold {old_thr:.3f} -> {args.thr:.3f}  |  file: {path.name}"
    )


if __name__ == "__main__":
    main()