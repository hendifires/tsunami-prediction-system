# experiments/run_experiment.py
from __future__ import annotations
# cSpell:ignore yaml ohe

"""
Runner serbaguna untuk eksperimen:
- FE (feature_engineering)  -> tambah distance_to_coast_km (opsional) + OHE diagnostics
- SMOTE (smote_pipeline)    -> buat ulang split train/test + varian SMOTE
- STACK (stacking_pipeline) -> train Stacking + simpan threshold & artefak

Konfigurasi dibaca dari experiments/configs.yaml
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any

try:
    import yaml  # type: ignore
except Exception as e:
    raise SystemExit(
        "PyYAML belum terpasang. Install dulu: pip install pyyaml"
    ) from e

ROOT = Path(__file__).resolve().parents[1]  # repo root


# ---------- Utilities ----------
def _log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: List[str]) -> None:
    """Jalankan proses child, tampilkan log real-time, dan fail jika exit!=0."""
    _log(f"[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def _str_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- Command builders ----------
def build_fe_cmd(cfg: Dict[str, Any]) -> List[str]:
    fe = cfg.get("fe", {})
    coast = (cfg.get("paths", {}) or {}).get("coastline") or fe.get("coastline")
    cmd = [
        sys.executable, "-m", "tsunami_prediction.feature_engineering",
        "--overwrite" if _str_bool(fe.get("overwrite", True)) else "",
        "--materialize-ohe" if _str_bool(fe.get("materialize_ohe", True)) else "",
    ]
    cmd = [c for c in cmd if c]  # buang string kosong
    if coast:
        cmd += ["--coast", str(coast)]
    return cmd


def build_smote_cmd(cfg: Dict[str, Any]) -> List[str]:
    sm = cfg.get("smote", {})
    cmd = [
        sys.executable, "-m", "tsunami_prediction.smote_pipeline",
        "--overwrite" if _str_bool(sm.get("overwrite", True)) else "",
    ]
    return [c for c in cmd if c]


def build_stack_cmds(cfg: Dict[str, Any]) -> List[List[str]]:
    st = cfg.get("stack", {})
    datasets = cfg.get("datasets", ["tectonic", "volcanic"])

    base_cmd = [
        sys.executable, "-m", "tsunami_prediction.stacking_pipeline",
        "--cv", str(st.get("cv", 3)),
        "--top-n", str(st.get("top_n", 30)),
        "--feature-select", str(st.get("feature_select", "none")),
    ]

    # toggles
    base_cmd += ["--with-xgb"] if _str_bool(st.get("with_xgb", True)) else ["--no-xgb"]
    base_cmd += ["--meta-grid"] if _str_bool(st.get("meta_grid", True)) else ["--no-meta-grid"]
    if _str_bool(st.get("with_mlp", False)):
        base_cmd += ["--with-mlp"]
    if _str_bool(st.get("fast", False)):
        base_cmd += ["--fast", "--fast-n", str(st.get("fast_n", 5000))]

    use_smote = str(st.get("use_smote", "auto")).lower()
    base_cmd += ["--use-smote", use_smote]

    ablation = _str_bool(st.get("ablation", False))

    cmds: List[List[str]] = []
    for ds in datasets:
        # ← inilah perbaikan yang menghapus warning:
        cmd = base_cmd + ["--datasets", ds] + (["--ablation"] if ablation else [])
        cmds.append(cmd)
    return cmds


# ---------- Main ----------
def main() -> None:
    cfg_path = ROOT / "experiments" / "configs.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Config tidak ditemukan: {cfg_path}")

    cfg = load_cfg(cfg_path)

    steps = cfg.get("steps", {})
    do_fe = _str_bool(steps.get("fe", True))
    do_sm = _str_bool(steps.get("smote", True))
    do_st = _str_bool(steps.get("stack", True))

    _log("[Runner] Mulai eksperimen…")

    if do_fe:
        _log("[Runner] 1) Feature Engineering")
        run_cmd(build_fe_cmd(cfg))
    else:
        _log("[Runner] 1) Feature Engineering — dilewati")

    if do_sm:
        _log("[Runner] 2) SMOTE pipeline")
        run_cmd(build_smote_cmd(cfg))
    else:
        _log("[Runner] 2) SMOTE pipeline — dilewati")

    if do_st:
        _log("[Runner] 3) Stacking pipeline")
        for cmd in build_stack_cmds(cfg):
            run_cmd(cmd)
    else:
        _log("[Runner] 3) Stacking pipeline — dilewati")

    _log("[Runner] Selesai ✅")


if __name__ == "__main__":
    main()