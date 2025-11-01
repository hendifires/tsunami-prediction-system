# src/tsunami_prediction/smote_pipeline.py
from __future__ import annotations

"""
SMOTE pipeline (tectonic & volcanic) dengan anti-data-leakage:
- Load *_fe_ohe.csv jika ada; jika tidak, load *_fe.csv lalu OHE otomatis (opsional).
- Split train/test terstratifikasi.
- Cleaning: ganti inf -> NaN, laporkan missing values.
- Imputation (fit di train, transform ke test) + buang zero-variance.
- SMOTE/SMOTETomek/SMOTEENN hanya di training set.
- Simpan train.csv, test.csv, train_smote.csv + ringkasan & plot distribusi kelas.
- Simpan artifacts (imputer, varselector, daftar fitur, config).
"""

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple

# Matplotlib non-GUI
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# ---------------- PATHS ----------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
ART = ROOT / "artifacts"

for p in [PROCESSED, TAB, FIG, ART]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- Utils ----------
def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)

def _savetab(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def _one_hot_encoder() -> OneHotEncoder:
    # Kompatibel untuk sklearn lama/baru
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _format_axis_plain(ax):
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))

def _nan_report(X: pd.DataFrame, name: str, dataset: str):
    s = X.isna().sum()
    s = s[s > 0].sort_values(ascending=False)
    rep = s.rename("n_missing").reset_index().rename(columns={"index": "feature"})
    if rep.empty:
        rep = pd.DataFrame(columns=["feature", "n_missing"])
    _savetab(rep, TAB / f"{dataset}_nan_{name}.csv")

def _replace_non_finite(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan)

# ---------- Data loading (FE / FE_OHE) ----------
def load_fe_dataset(dataset: str, auto_ohe: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load *_fe_ohe.csv bila ada; jika tidak ada dan auto_ohe=True, lakukan OHE pada *_fe.csv.
    Return: (dataframe, feature_columns)
    """
    target = "tsu"
    p_ohe = PROCESSED / f"{dataset}_fe_ohe.csv"
    p_fe  = PROCESSED / f"{dataset}_fe.csv"

    # 1) Prefer FE_OHE jika tersedia
    if p_ohe.exists():
        df = _read_csv(p_ohe)
        if target not in df.columns:
            raise KeyError(f"[SMOTE] '{dataset}_fe_ohe.csv' tidak memiliki kolom '{target}'.")
        feat_cols = [c for c in df.columns if c != target]
        return df, feat_cols

    # 2) FE wajib ada
    if not p_fe.exists():
        raise FileNotFoundError(
            f"[SMOTE] {p_ohe.name} & {p_fe.name} tidak ditemukan. Jalankan feature_engineering dulu."
        )

    df = _read_csv(p_fe)
    if target not in df.columns:
        raise KeyError(f"[SMOTE] '{dataset}_fe.csv' tidak memiliki kolom '{target}'.")
    feat_cols = [c for c in df.columns if c != target]

    # 3) Kalau auto_ohe=False, kembalikan apa adanya
    if not auto_ohe:
        return df, feat_cols

    # 4) OHE otomatis (pakai daftar kategorikal dari artifacts bila ada; kalau tidak, infer dtype)
    cat_job = ART / f"{dataset}_fe_cat_cols.joblib"
    if cat_job.exists():
        cat_cols: List[str] = [c for c in load(cat_job) if c in df.columns]
    else:
        cat_cols = [
            c for c in df.columns
            if c != target and (df[c].dtype == "object" or str(df[c].dtype) == "boolean")
        ]

    # <- Perbaikan Sourcery (hindari len(cat_cols) == 0)
    if not cat_cols:
        return df, feat_cols

    enc = _one_hot_encoder()
    arr = enc.fit_transform(df[cat_cols].astype("object"))
    ohe_cols = enc.get_feature_names_out(cat_cols).tolist()
    base = df.drop(columns=cat_cols).reset_index(drop=True)
    df_ohe = pd.concat([base, pd.DataFrame(arr, columns=ohe_cols, index=base.index)], axis=1)

    dump(enc, ART / f"{dataset}_smote_ohe_encoder.joblib")
    dump(ohe_cols, ART / f"{dataset}_smote_ohe_feature_names.joblib")

    feat_cols = [c for c in df_ohe.columns if c != target]
    return df_ohe, feat_cols

# ---------- Split ----------
def stratified_split(
    df: pd.DataFrame,
    target: str = "tsu",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = df[df[target].notna()].copy()
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# ---------- Cleaning & Imputation (train-only fit) ----------
def clean_and_impute(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    dataset: str,
    strategy: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    - Laporan NaN sebelum & sesudah.
    - Ganti inf -> NaN.
    - SimpleImputer (fit di train, apply ke test).
    - VarianceThreshold(0.0) di train (fit) lalu transform ke test.
    """
    # Report sebelum
    _nan_report(X_train, "train_before", dataset)
    _nan_report(X_test,  "test_before",  dataset)

    # Ganti inf → NaN
    X_train = _replace_non_finite(X_train)
    X_test  = _replace_non_finite(X_test)

    # Imputer (fit di train, transform ke test)
    imp = SimpleImputer(strategy=strategy)
    Xtr = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(imp.transform(X_test),  columns=X_test.columns,  index=X_test.index)

    # Drop zero-variance (fit di train)
    vt = VarianceThreshold(threshold=0.0)
    Xtr_v = vt.fit_transform(Xtr)
    kept = Xtr.columns[vt.get_support()].tolist()
    Xtr_v = pd.DataFrame(Xtr_v, columns=kept, index=Xtr.index)
    Xte_v = pd.DataFrame(vt.transform(Xte), columns=kept, index=Xte.index)

    # Simpan artifacts
    dump(imp, ART / f"{dataset}_smote_imputer.joblib")
    dump(vt,  ART / f"{dataset}_smote_varselector.joblib")
    dump(kept, ART / f"{dataset}_smote_kept_features.joblib")

    # Report sesudah
    _nan_report(Xtr_v, "train_after", dataset)
    _nan_report(Xte_v, "test_after",  dataset)

    return Xtr_v, Xte_v, kept

# ---------- SMOTE variants ----------
def make_smote(variant: str, random_state: int, k_neighbors: int, sampling_strategy):
    variant = variant.lower()
    if variant == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    elif variant == "smote_tomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        return SMOTETomek(random_state=random_state, smote=sm)
    elif variant == "smoteenn":
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        return SMOTEENN(random_state=random_state, smote=sm)
    else:
        raise ValueError("variant harus salah satu dari: smote | smote_tomek | smoteenn")

def adjust_k_neighbors(y_train: pd.Series, k_neighbors: int) -> int:
    """
    k_neighbors maksimum harus < jumlah sampel kelas minoritas.
    Jika data sangat kecil, turunkan otomatis agar valid.
    """
    vc = y_train.value_counts()
    if vc.empty:
        return max(1, min(k_neighbors, 5))
    minority = int(vc.idxmin())
    n_min = int(vc.loc[minority])
    return max(1, min(k_neighbors, max(1, n_min - 1)))

# ---------- Plots ----------
def plot_pie(before: Dict[int, int], after: Dict[int, int], title: str, out_png: Path):
    labels = ["Non-tsunami (0)", "Tsunami (1)"]
    b_vals = [before.get(0, 0), before.get(1, 0)]
    a_vals = [after.get(0, 0), after.get(1, 0)]

    plt.figure(figsize=(10, 4.2))
    plt.subplot(1, 2, 1)
    plt.pie(b_vals, labels=labels, autopct="%.1f%%", startangle=90)
    plt.title(f"Before SMOTE\n({title})")

    plt.subplot(1, 2, 2)
    plt.pie(a_vals, labels=labels, autopct="%.1f%%", startangle=90)
    plt.title(f"After SMOTE\n({title})")
    _savefig(out_png)

def plot_bar(before: Dict[int, int], after: Dict[int, int], title: str, out_png: Path):
    labels = ["Non-tsunami (0)", "Tsunami (1)"]
    b_vals = [before.get(0, 0), before.get(1, 0)]
    a_vals = [after.get(0, 0), after.get(1, 0)]

    plt.figure(figsize=(10, 4.2))

    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(labels, b_vals)
    _format_axis_plain(ax1)
    for i, v in enumerate(b_vals):
        ax1.text(i, v + max(1, 0.02 * max(b_vals)), str(int(v)), ha="center", fontweight="bold")
    ax1.set_title(f"Before SMOTE\n({title})")
    ax1.set_ylabel("Jumlah Sampel")

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(labels, a_vals)
    _format_axis_plain(ax2)
    for i, v in enumerate(a_vals):
        ax2.text(i, v + max(1, 0.02 * max(a_vals)), str(int(v)), ha="center", fontweight="bold")
    ax2.set_title(f"After SMOTE\n({title})")
    ax2.set_ylabel("Jumlah Sampel")

    _savefig(out_png)

# ---------- Main runner per dataset ----------
def run_one(
    dataset: str,
    overwrite: bool,
    use_ohe_auto: bool,
    test_size: float,
    random_state: int,
    k_neighbors: int,
    sampling_strategy,
    variant: str,
):
    print(f"[SMOTE] Start -> {dataset}")

    p_train = PROCESSED / f"{dataset}_train.csv"
    p_test  = PROCESSED / f"{dataset}_test.csv"
    p_train_smote = PROCESSED / f"{dataset}_train_smote.csv"
    p_sum   = TAB / f"{dataset}_smote_summary.csv"
    p_pie   = FIG / f"{dataset}_smote_pie.png"
    p_bar   = FIG / f"{dataset}_smote_bar.png"
    p_cfg   = ART / f"{dataset}_smote_config.joblib"

    if all(p.exists() for p in [p_train, p_test, p_train_smote]) and not overwrite:
        print(f"[INFO] Reusing existing SMOTE outputs for '{dataset}' (use --overwrite untuk regenerasi).")
        return

    # 1) Load FE / FE_OHE
    df, feat_cols = load_fe_dataset(dataset, auto_ohe=use_ohe_auto)
    if "tsu" not in df.columns:
        raise KeyError(f"[SMOTE] '{dataset}' tidak memiliki kolom 'tsu'.")

    # 2) Split stratified
    X_train, X_test, y_train, y_test = stratified_split(df, target="tsu", test_size=test_size, random_state=random_state)

    # 3) CLEAN + IMPUTE (fit di train → transform test)
    X_train, X_test, kept_cols = clean_and_impute(X_train, X_test, dataset, strategy="median")

    # 4) Adjust k_neighbors (berdasarkan jumlah minoritas di train)
    k_adj = adjust_k_neighbors(y_train, k_neighbors)

    # 5) Fit-resample di training
    smote_obj = make_smote(
        variant=variant,
        random_state=random_state,
        k_neighbors=k_adj,
        sampling_strategy=sampling_strategy,
    )
    X_train_res, y_train_res = smote_obj.fit_resample(X_train, y_train)

    # 6) Simpan dataset
    df_train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    df_test  = pd.concat([X_test.reset_index(drop=True),  y_test.reset_index(drop=True)],  axis=1)
    df_train_res = pd.concat(
        [pd.DataFrame(X_train_res, columns=X_train.columns), pd.Series(y_train_res, name="tsu")],
        axis=1,
    )

    df_train.to_csv(p_train, index=False)
    df_test.to_csv(p_test, index=False)
    df_train_res.to_csv(p_train_smote, index=False)

    # 7) Ringkasan + plots
    before = y_train.value_counts().sort_index().to_dict()
    after  = pd.Series(y_train_res).value_counts().sort_index().to_dict()

    summary = pd.DataFrame({
        "class": [0, 1],
        "train_before": [before.get(0, 0), before.get(1, 0)],
        "train_after":  [after.get(0, 0),  after.get(1, 0)],
    })
    _savetab(summary, p_sum)

    plot_pie(before, after, title=dataset.title(), out_png=p_pie)
    plot_bar(before, after, title=dataset.title(), out_png=p_bar)

    # 8) Simpan config/artifacts
    dump(
        {
            "variant": variant,
            "random_state": random_state,
            "k_neighbors_requested": k_neighbors,
            "k_neighbors_used": k_adj,
            "sampling_strategy": sampling_strategy,
            "test_size": test_size,
            "feature_columns_before": feat_cols,
            "feature_columns_after_clean": kept_cols,
        },
        p_cfg,
    )

    print(f"[SMOTE] {dataset}: done | train={p_train.name}, test={p_test.name}, smote={p_train_smote.name}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="SMOTE pipeline (tectonic/volcanic) with anti-leakage + imputation."
    )
    ap.add_argument("--datasets", nargs="+", default=["tectonic", "volcanic"],
                    help="dataset list: tectonic volcanic")
    ap.add_argument("--overwrite", action="store_true", help="regenerate outputs")
    ap.add_argument("--no-auto-ohe", action="store_true",
                    help="jangan OHE otomatis jika *_fe_ohe.csv tidak ada")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--k-neighbors", type=int, default=5)
    ap.add_argument("--sampling-strategy", type=str, default="auto",
                    help="auto | float (0<r<=1) | dict JSON-like, contoh: '{\"1\": 2000}'")
    ap.add_argument("--variant", type=str, default="smote",
                    choices=["smote", "smote_tomek", "smoteenn"])
    args = ap.parse_args()

    # Parse sampling_strategy aman (hindari eval tak aman)
    sampling_strategy = args.sampling_strategy
    if sampling_strategy != "auto":
        try:
            s = sampling_strategy.strip()
            sampling_strategy = ast.literal_eval(s) if s.startswith("{") else float(s)
        except Exception:
            print("[WARN] sampling_strategy tidak valid, fallback ke 'auto'.")
            sampling_strategy = "auto"

    print("[SMOTE] Pipeline start …")
    for ds in args.datasets:
        run_one(
            dataset=ds,
            overwrite=args.overwrite,
            use_ohe_auto=(not args.no_auto_ohe),
            test_size=args.test_size,
            random_state=args.random_state,
            k_neighbors=args.k_neighbors,
            sampling_strategy=sampling_strategy,
            variant=args.variant,
        )
    print(f"[DONE] SMOTE pipeline selesai.\n - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}\n - DATA: {PROCESSED}")

if __name__ == "__main__":
    main()