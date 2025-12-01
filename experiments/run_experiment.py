from __future__ import annotations
# cSpell:ignore yaml ohe

"""
Runner serbaguna untuk eksperimen (berbasis configs.yaml):

- FE (feature_engineering)
    -> feature engineering ringan + events_fe.csv
- PREPROC (preprocessing)
    -> cleaning & split awal (events_train/test)
- SMOTE (smote_pipeline)
    -> buat ulang split train/test + varian SMOTE
- STACK (stacking_pipeline)
    -> train Stacking + simpan threshold & artefak

Versi tesis:
- Tidak ada XGBoost sama sekali.
- Base learners: DT, RF, MLP (opsional), SVM, NB, KNN.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "PyYAML belum terpasang. Install dulu: pip install pyyaml"
    ) from e

# File ini berada di folder experiments/, jadi root project = parent-nya
ROOT = Path(__file__).resolve().parents[1]


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
        data = yaml.safe_load(f)
    return data or {}


# ---------- Command builders ----------
def build_fe_cmd(cfg: Dict[str, Any]) -> List[str]:
    """
    Bangun perintah untuk feature_engineering.

    Konfigurasi opsional di YAML:

    fe:
      overwrite: true
      materialize_ohe: true
    """
    fe = cfg.get("fe", {}) or {}
    cmd = [
        sys.executable,
        "-m",
        "tsunami_prediction.feature_engineering",
    ]
    if _str_bool(fe.get("overwrite", True)):
        cmd.append("--overwrite")
    if _str_bool(fe.get("materialize_ohe", True)):
        cmd.append("--materialize-ohe")
    return cmd


def build_preproc_cmd(cfg: Dict[str, Any]) -> List[str]:
    """
    Bangun perintah untuk preprocessing.

    Saat ini diasumsikan modul preprocessing bisa dijalankan
    dengan default CLI: python -m tsunami_prediction.preprocessing
    (input default: data/processed/events_fe.csv).
    """
    pre = cfg.get("preproc", {}) or {}
    cmd = [
        sys.executable,
        "-m",
        "tsunami_prediction.preprocessing",
    ]
    # Kalau suatu saat mau override year_min/test_size lewat YAML, bisa ditambah di sini.
    if "year_min" in pre:
        cmd += ["--year-min", str(pre["year_min"])]
    if "test_size" in pre:
        cmd += ["--test-size", str(pre["test_size"])]
    return cmd


def build_smote_cmd(cfg: Dict[str, Any]) -> List[str]:
    """
    Bangun perintah untuk smote_pipeline.

    Konfigurasi opsional di YAML:

    smote:
      overwrite: true
    """
    sm = cfg.get("smote", {}) or {}
    cmd = [
        sys.executable,
        "-m",
        "tsunami_prediction.smote_pipeline",
    ]
    if _str_bool(sm.get("overwrite", True)):
        cmd.append("--overwrite")
    return cmd


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

    def g(key: str, default: Any) -> Any:
        """Ambil nilai dari run_cfg, lalu global_cfg, lalu default."""
        return run_cfg.get(key, global_cfg.get(key, default))

    datasets = run_cfg.get("datasets", ["tectonic", "volcanic"])

    cv = int(g("cv", 3))
    top_n = int(g("top_n", 30))
    feature_select = str(g("feature_select", "none")).lower()

    with_mlp = _str_bool(g("with_mlp", True))
    meta_grid = _str_bool(g("meta_grid", False))
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

    base_cmd += ["--meta-grid"] if meta_grid else ["--no-meta-grid"]

    if with_mlp:
        base_cmd += ["--with-mlp"]

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

    parser = argparse.ArgumentParser(
        description="Runner eksperimen FE + preprocessing + SMOTE + stacking"
    )
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

    steps = cfg.get("steps", {}) or {}
    do_pre = _str_bool(steps.get("preproc", True))
    do_fe = _str_bool(steps.get("fe", True))
    do_sm = _str_bool(steps.get("smote", True))
    do_st = _str_bool(steps.get("stack", True))

    seeds_cfg = cfg.get("seeds")
    if seeds_cfg:
        seeds = list(seeds_cfg)
    else:
        seeds = [int(global_cfg.get("random_state", 42))]

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

    # 0) Feature Engineering (harus duluan supaya events_fe.csv ada)
    if do_fe:
        _log("[Runner] 0) Feature Engineering")
        run_cmd(build_fe_cmd(cfg))
    else:
        _log("[Runner] 0) Feature Engineering — dilewati")

    # 1) Preprocessing (multiclass events)
    if do_pre:
        _log("[Runner] 1) Preprocessing (events)")
        run_cmd(build_preproc_cmd(cfg))
    else:
        _log("[Runner] 1) Preprocessing — dilewati")

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