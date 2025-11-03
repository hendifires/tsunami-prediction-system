from __future__ import annotations

from pathlib import Path
import argparse
import joblib

from tsunami_prediction.serve_api import _ensure_unpickle_compat

ART = Path(__file__).resolve().parents[1] / "artifacts"


def find_artifact(dataset: str) -> Path:
    """
    Cari artifact model stacking untuk dataset tertentu.
    Ambil file pertama yang cocok pola '<dataset>_stack_*.joblib'.
    """
    if cands := sorted(ART.glob(f"{dataset}_stack_*.joblib")):
        return cands[0]

    raise SystemExit(f"Artifact for {dataset} not found in {ART}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set custom decision threshold for a stacking model.")
    ap.add_argument("--dataset", choices=["tectonic", "volcanic"], required=True)
    ap.add_argument("--thr", type=float, required=True, help="new decision threshold (0–1)")
    ap.add_argument("--path", type=str, help="optional explicit artifact path")
    args = ap.parse_args()

    _ensure_unpickle_compat()

    path = Path(args.path) if args.path else find_artifact(args.dataset)

    model_dict = joblib.load(path)
    model_dict["decision_threshold"] = float(args.thr)
    joblib.dump(model_dict, path, compress=3)

    print(f"OK set threshold={args.thr} -> {path}")


if __name__ == "__main__":
    main()