from __future__ import annotations
# cSpell:ignore nosmote smote xgb

"""
Export tabel ringkasan resmi untuk tesis:

- Menggabungkan semua eksperimen stacking dari
  reports/tables/stacking_experiments_all_metrics.csv

- Fokus pada:
    * dataset   = 'events'
    * scenario  = ['nosmote', 'smote']
    * model     = ['rf', 'stacking_lr', 'stacking_xgb', 'extra_trees' (opsional)]

- Menambahkan:
    * year_window       : '1900-2024' atau '2000-2024' (dari exp_tag)
    * is_main_model     : True untuk model utama tesis
    * is_baseline_model : True untuk model pembanding/baseline

Output:
    - reports/tables/stacking_thesis_summary.csv
    - reports/tables/stacking_thesis_summary.md
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"

TAB.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_all_metrics() -> pd.DataFrame:
    """
    Load tabel gabungan semua eksperimen stacking.
    Diharapkan sudah dibuat oleh compare_stacking_runs.py
    sebagai stacking_experiments_all_metrics.csv
    """
    path = TAB / "stacking_experiments_all_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"[Export] File tidak ditemukan: {path}\n"
            "Pastikan python -m tsunami_prediction.compare_stacking_runs sudah dijalankan."
        )
    df = pd.read_csv(path)
    _log(f"[Export] Loaded all metrics: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def _infer_year_window(exp_tag: str) -> str:
    """
    Mapping sederhana dari exp_tag ke label window tahun
    yang mudah dibaca di tesis.
    """
    if isinstance(exp_tag, str):
        if "y1900_2024" in exp_tag:
            return "1900-2024"
        if "y2000_2024" in exp_tag:
            return "2000-2024"
    return "unknown"


def _prepare_thesis_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dan rapikan tabel metrik agar siap dipakai di tesis.
    """
    # --- Pastikan kolom kunci ada ---
    required_cols = ["dataset", "scenario", "model", "accuracy", "f1_macro"]
    for c in required_cols:
        if c not in df_all.columns:
            raise KeyError(
                f"[Export] Kolom wajib '{c}' tidak ditemukan di stacking_experiments_all_metrics.csv. "
                f"Kolom yang ada: {list(df_all.columns)}"
            )

    # exp_tag bisa saja belum ada jika versi lama pipeline,
    # jadi kita toleran: kalau tidak ada, isi 'unknown'.
    if "exp_tag" not in df_all.columns:
        _log("[Export] Kolom 'exp_tag' tidak ditemukan, semua baris akan diberi year_window='unknown'.")
        df_all["exp_tag"] = "unknown"

    # --- Filter hanya dataset & model yang relevan ---
    df = df_all.copy()

    df = df[df["dataset"] == "events"]

    df = df[df["scenario"].isin(["nosmote", "smote"])]

    # Model inti untuk tesis
    core_models = {"rf", "stacking_lr", "stacking_xgb", "extra_trees"}
    df = df[df["model"].isin(core_models)]

    # Tambah year_window dari exp_tag
    df["year_window"] = df["exp_tag"].apply(_infer_year_window)

    # Sortir supaya rapi:
    #   - year_window
    #   - scenario (nosmote dulu, lalu smote)
    #   - model (rf, extra_trees, stacking_lr, stacking_xgb)
    scenario_order = {"nosmote": 0, "smote": 1}
    model_order = {"rf": 0, "extra_trees": 1, "stacking_lr": 2, "stacking_xgb": 3}

    df["scenario_order"] = df["scenario"].map(scenario_order).fillna(99)
    df["model_order"] = df["model"].map(model_order).fillna(99)

    df = df.sort_values(
        by=["year_window", "scenario_order", "model_order"],
        ascending=[True, True, True],
    )

    # --- Pilih subset kolom yang akan ditampilkan ---
    keep_cols = [
        "dataset",
        "year_window",
        "scenario",
        "model",
        "accuracy",
        "f1_macro",
    ]
    # kalau prec_macro & rec_macro ada, ikutkan
    if "prec_macro" in df.columns:
        keep_cols.append("prec_macro")
    if "rec_macro" in df.columns:
        keep_cols.append("rec_macro")
    keep_cols.append("exp_tag")

    df = df[keep_cols].reset_index(drop=True)

    return df


def _pick_main_and_baseline(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Tandai:
      - is_main_model=True : model stacking utama tesis
      - is_baseline_model=True : model pembanding utama

    Aturan (disesuaikan dengan hasil eksperimen yang terlihat di log kamu):

    - Main model   : events, year_window='2000-2024',
                     scenario='smote', model='stacking_lr'
                     (punya f1_macro tertinggi di konfigurasi resmi)

    - Baseline     : events, year_window='1900-2024',
                     scenario='nosmote', model='rf'
                     (RF tanpa SMOTE sebagai pembanding klasik)
    """
    df = df.copy()
    df["is_main_model"] = False
    df["is_baseline_model"] = False

    # --- Main model (2000–2024, smote, stacking_lr) ---
    main_mask = (
        (df["year_window"] == "2000-2024")
        & (df["scenario"] == "smote")
        & (df["model"] == "stacking_lr")
    )
    if main_mask.any():
        df.loc[main_mask, "is_main_model"] = True
        main_desc = "events, 2000-2024, smote, stacking_lr"
    else:
        main_desc = "NONE_FOUND"

    # --- Baseline (1900–2024, nosmote, rf) ---
    base_mask = (
        (df["year_window"] == "1900-2024")
        & (df["scenario"] == "nosmote")
        & (df["model"] == "rf")
    )
    if base_mask.any():
        df.loc[base_mask, "is_baseline_model"] = True
        base_desc = "events, 1900-2024, nosmote, rf"
    else:
        base_desc = "NONE_FOUND"

    _log(f"[Export] Main model      : {main_desc}")
    _log(f"[Export] Baseline model  : {base_desc}")

    return df, main_desc, base_desc


def _save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """
    Simpan tabel ke Markdown dengan format pipe table.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Untuk Markdown yang rapi, kita buang beberapa kolom teknis jika tidak perlu.
    df_md = df.copy()

    # Atur urutan kolom untuk readability tesis
    cols_order = [
        "dataset",
        "year_window",
        "scenario",
        "model",
        "accuracy",
        "f1_macro",
    ]
    if "prec_macro" in df_md.columns:
        cols_order.append("prec_macro")
    if "rec_macro" in df_md.columns:
        cols_order.append("rec_macro")
    cols_order.extend(["exp_tag", "is_main_model", "is_baseline_model"])

    df_md = df_md[cols_order]

    md = df_md.to_markdown(index=False)

    with path.open("w", encoding="utf-8") as f:
        f.write(md)

    _log(f"[Export] Saved Markdown table to {path}")


def main() -> None:
    _log("[Export] Start exporting thesis summary tables …")

    df_all = _load_all_metrics()
    df_thesis = _prepare_thesis_summary(df_all)
    df_thesis, main_desc, base_desc = _pick_main_and_baseline(df_thesis)

    # Simpan CSV
    out_csv = TAB / "stacking_thesis_summary.csv"
    df_thesis.to_csv(out_csv, index=False)
    _log(f"[Export] Saved thesis summary CSV to {out_csv}")

    # Simpan Markdown
    out_md = TAB / "stacking_thesis_summary.md"
    _save_markdown_table(df_thesis, out_md)

    _log("[Export] Done.")


if __name__ == "__main__":
    main()