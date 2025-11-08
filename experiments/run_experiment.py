# experiments/run_experiment.py
from __future__ import annotations
# cSpell:ignore yaml ohe

"""
Runner serbaguna untuk eksperimen (berbasis configs.yaml):

- FE (feature_engineering)
    -> tambah distance_to_coast_km (opsional) + OHE diagnostics
- SMOTE (smote_pipeline)
    -> buat ulang split train/test + varian SMOTE
- STACK (stacking_pipeline)
    -> train Stacking + simpan threshold & artefak

Struktur konfigurasi (experiments/configs.yaml):

global:
  cv: 3
  random_state: 42
  with_xgb: false
  with_mlp: true
  feature_select: "none"
  top_n: 30
  meta_grid: true
  fast: false
  fast_n: 5000
  collect_outputs: true

seeds: [42]   # optional, override global.random_state per run

runs:
  - name: auto_best
    datasets: ["tectonic", "volcanic"]
    use_smote: "auto"
    ablation: false

  - name: ablation_full
    datasets: ["tectonic", "volcanic"]
    ablation: true

dst.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover - dependency error path
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
    """
    Bangun perintah untuk feature_engineering.

    Konfigurasi opsional:

    fe:
      overwrite: true
      materialize_ohe: true

    paths:
      coastline: "data/coastline/ne_10m_coastline.shp"
    """
    fe = cfg.get("fe", {})
    coast = (cfg.get("paths", {}) or {}).get("coastline") or fe.get("coastline")
    cmd = [
        sys.executable,
        "-m",
        "tsunami_prediction.feature_engineering",
        "--overwrite" if _str_bool(fe.get("overwrite", True)) else "",
        "--materialize-ohe" if _str_bool(fe.get("materialize_ohe", True)) else "",
    ]
    cmd = [c for c in cmd if c]  # buang string kosong
    if coast:
        cmd += ["--coast", str(coast)]
    return cmd


def build_smote_cmd(cfg: Dict[str, Any]) -> List[str]:
    """
    Bangun perintah untuk smote_pipeline.

    Konfigurasi opsional:

    smote:
      overwrite: true
    """
    sm = cfg.get("smote", {})
    cmd = [
        sys.executable,
        "-m",
        "tsunami_prediction.smote_pipeline",
        "--overwrite" if _str_bool(sm.get("overwrite", True)) else "",
    ]
    return [c for c in cmd if c]


def build_stack_cmds(
    global_cfg: Dict[str, Any],
    run_cfg: Dict[str, Any],
    seed: int,
) -> List[List[str]]:
    """
    Bangun list command untuk stacking_pipeline berdasarkan:
    - global config (bagian 'global')
    - konfigurasi run spesifik (entry di 'runs')
    - nilai seed (override random_state)
    """

    # helper: ambil nilai dari run_cfg, kalau tidak ada pakai global_cfg
    def g(key: str, default: Any) -> Any:
        return run_cfg[key] if key in run_cfg else global_cfg.get(key, default)

    datasets = run_cfg.get("datasets", ["tectonic", "volcanic"])

    cv = int(g("cv", 3))
    top_n = int(g("top_n", 30))
    feature_select = str(g("feature_select", "none")).lower()

    with_xgb = _str_bool(g("with_xgb", False))  # default: XGB dimatikan
    with_mlp = _str_bool(g("with_mlp", False))

    meta_grid = _str_bool(g("meta_grid", True))

    fast = _str_bool(g("fast", False))
    fast_n = int(g("fast_n", 5000))

    use_smote = str(g("use_smote", "auto")).lower()  # auto|yes|no
    ablation = _str_bool(g("ablation", False))

    base_cmd: List[str] = [
        sys.executable,
        "-m",
        "tsunami_prediction.stacking_pipeline",
        "--cv",
        str(cv),
        "--top-n",
        str(top_n),
        "--feature-select",
        feature_select,
        "--random-state",
        str(seed),
        "--use-smote",
        use_smote,
    ]

    # toggles XGB (dipertahankan cuma untuk kompatibilitas; default False)
    base_cmd += ["--with-xgb"] if with_xgb else ["--no-xgb"]

    # meta-grid
    base_cmd += ["--meta-grid"] if meta_grid else ["--no-meta-grid"]

    # MLP sebagai base learner
    if with_mlp:
        base_cmd += ["--with-mlp"]

    # mode cepat
    if fast:
        base_cmd += ["--fast", "--fast-n", str(fast_n)]

    cmds: List[List[str]] = []
    for ds in datasets:
        cmd = base_cmd + ["--datasets", ds]
        if ablation:
            cmd += ["--ablation"]
        cmds.append(cmd)

    return cmds


# ---------- Main ----------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Runner eksperimen stacking + SMOTE")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "experiments" / "configs.yaml"),
        help="Path ke file konfigurasi YAML (default: experiments/configs.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config tidak ditemukan: {cfg_path}")

    cfg = load_cfg(cfg_path)
    global_cfg = cfg.get("global", {}) or {}
    runs_cfg = cfg.get("runs", []) or []

    # steps opsional (kalau tidak ada, semua dianggap True)
    steps = cfg.get("steps", {}) or {}
    do_fe = _str_bool(steps.get("fe", True))
    do_sm = _str_bool(steps.get("smote", True))
    do_st = _str_bool(steps.get("stack", True))

    # seeds untuk random_state stacking
    seeds = cfg.get("seeds") or [int(global_cfg.get("random_state", 42))]

    # kalau tidak ada runs didefinisikan, buat satu run default
    if not runs_cfg:
        runs_cfg = [
            {
                "name": "default",
                "datasets": ["tectonic", "volcanic"],
                "use_smote": "auto",
                "ablation": False,
            }
        ]

    _log(f"[Runner] Mulai eksperimen (config={cfg_path}) …")

    # 1) Feature Engineering
    if do_fe:
        _log("[Runner] 1) Feature Engineering")
        run_cmd(build_fe_cmd(cfg))
    else:
        _log("[Runner] 1) Feature Engineering — dilewati")

    # 2) SMOTE pipeline
    if do_sm:
        _log("[Runner] 2) SMOTE pipeline")
        run_cmd(build_smote_cmd(cfg))
    else:
        _log("[Runner] 2) SMOTE pipeline — dilewati")

    # 3) Stacking runs (loop over runs x seeds)
    if do_st:
        _log("[Runner] 3) Stacking pipeline (multi-run / multi-seed)")
        for run_cfg in runs_cfg:
            name = str(run_cfg.get("name", "run"))
            for seed in seeds:
                _log(f"[Runner]   -> Run='{name}' | seed={seed}")
                cmds = build_stack_cmds(global_cfg, run_cfg, seed)
                for cmd in cmds:
                    run_cmd(cmd)
    else:
        _log("[Runner] 3) Stacking pipeline — dilewati")

    _log("[Runner] Selesai ✅")


if __name__ == "__main__":
    main()