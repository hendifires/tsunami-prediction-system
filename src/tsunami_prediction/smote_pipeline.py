# src/tsunami_prediction/smote_pipeline.py
from __future__ import annotations
# cSpell:ignore dengan Tomek ENN oversampling imbalanced

"""
SMOTE pipeline (tectonic & volcanic) dengan anti-data-leakage:

- Load *_fe_ohe.csv jika ada; jika tidak, load *_fe.csv lalu OHE otomatis (opsional).
- Split train/test terstratifikasi berdasarkan tsu.
- Cleaning: ganti inf -> NaN, laporkan missing values, drop kolom all-NaN.
- Imputation (fit di train, transform ke test) + buang zero-variance.
- SMOTE/SMOTE+Tomek/SMOTEENN hanya di training set.

Output utama per dataset:

- data/processed/{dataset}_train.csv                -> train asli (tanpa SMOTE)
- data/processed/{dataset}_test.csv                 -> test asli (tanpa SMOTE)
- data/processed/{dataset}_train_{tag}.csv          -> train hasil resampling:
      tag = smote | smote_tomek | smote_enn
- reports/tables/{dataset}_{tag}_summary.csv
- reports/figures/{dataset}_{tag}_pie.png
- reports/figures/{dataset}_{tag}_bar.png
- artifacts/{dataset}_{tag}_config.joblib

Catatan penting (sinkron dengan stacking_pipeline):

- Stacking memakai:
    * {dataset}_train.csv           sebagai varian "no-SMOTE"
    * {dataset}_train_smote.csv     (tag: "smote")
    * {dataset}_train_smote_tomek.csv (tag: "smote_tomek")
    * {dataset}_train_smote_enn.csv (tag: "smote_enn")

Tambahan visual konseptual (simulasi 2D, bukan data real):

- reports/figures/smote_mechanism.png
    Ilustrasi garis interpolasi SMOTE pada ruang fitur 2D.
- reports/figures/minority_regions.png
    Sebaran minority: safe / borderline / abnormal (rare+outlier).
- reports/figures/simulated_oversampling_grid.png
    Grid Baseline vs SMOTE vs SMOTE+Tomek vs SMOTEENN.

- reports/tables/smote_mechanism_points.csv
- reports/tables/smote_minority_region_counts.csv
- reports/tables/smote_simulated_summary.csv
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

for p in (PROCESSED, TAB, FIG, ART):
    p.mkdir(parents=True, exist_ok=True)


# ---------- Utils ----------
def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _savetab(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _one_hot_encoder() -> OneHotEncoder:
    """Helper kecil, kompatibel sklearn lama/baru."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _format_axis_plain(ax) -> None:
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))


def _nan_report(X: pd.DataFrame, name: str, dataset: str) -> None:
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
    Load {dataset}_fe_ohe.csv bila ada; jika tidak ada dan auto_ohe=True,
    lakukan OHE pada {dataset}_fe.csv.

    Return:
        df         : DataFrame (fitur + tsu)
        feat_cols  : daftar kolom fitur (tanpa 'tsu')
    """
    target = "tsu"
    p_ohe = PROCESSED / f"{dataset}_fe_ohe.csv"
    p_fe = PROCESSED / f"{dataset}_fe.csv"

    # 1) Prefer FE_OHE jika tersedia
    if p_ohe.exists():
        df = _read_csv(p_ohe)
        if target not in df.columns:
            raise KeyError(f"[SMOTE] '{p_ohe.name}' tidak memiliki kolom '{target}'.")
        feat_cols = [c for c in df.columns if c != target]
        return df, feat_cols

    # 2) FE wajib ada
    if not p_fe.exists():
        raise FileNotFoundError(
            f"[SMOTE] {p_ohe.name} & {p_fe.name} tidak ditemukan. Jalankan feature_engineering dulu."
        )

    df = _read_csv(p_fe)
    if target not in df.columns:
        raise KeyError(f"[SMOTE] '{p_fe.name}' tidak memiliki kolom '{target}'.")
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
            c
            for c in df.columns
            if c != target and (df[c].dtype == "object" or str(df[c].dtype) == "boolean")
        ]

    if not cat_cols:
        return df, feat_cols

    enc = _one_hot_encoder()
    arr = enc.fit_transform(df[cat_cols].astype("object"))
    ohe_cols = enc.get_feature_names_out(cat_cols).tolist()
    base = df.drop(columns=cat_cols).reset_index(drop=True)
    df_ohe = pd.concat(
        [base, pd.DataFrame(arr, columns=ohe_cols, index=base.index)],
        axis=1,
    )

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
    """
    Split stratified tsu → anti-data-leakage untuk SMOTE & imputer.
    """
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
    - Drop kolom yang all-NaN di training (tidak bisa di-impute).
    - SimpleImputer (fit di train, apply ke test).
    - VarianceThreshold(0.0) di train (fit) lalu transform ke test.
    """
    # Report sebelum
    _nan_report(X_train, "train_before", dataset)
    _nan_report(X_test, "test_before", dataset)

    # Ganti inf → NaN
    X_train = _replace_non_finite(X_train)
    X_test = _replace_non_finite(X_test)

    # Drop kolom yang full NaN di TRAIN (sklearn akan "skip" kolom2 ini)
    if all_nan_cols := X_train.columns[X_train.isna().all()].tolist():
        print(f"[{dataset}] Drop all-NaN features (no value for imputer): {all_nan_cols}")
        X_train = X_train.drop(columns=all_nan_cols)
        X_test = X_test.drop(columns=[c for c in all_nan_cols if c in X_test.columns])
        dump(all_nan_cols, ART / f"{dataset}_smote_allnan_features.joblib")

    # Imputer (fit di train, transform ke test)
    imp = SimpleImputer(strategy=strategy)
    Xtr_arr = imp.fit_transform(X_train)
    Xte_arr = imp.transform(X_test)

    Xtr = pd.DataFrame(Xtr_arr, columns=X_train.columns, index=X_train.index)
    Xte = pd.DataFrame(Xte_arr, columns=X_train.columns, index=X_test.index)

    # Drop zero-variance (fit di train)
    vt = VarianceThreshold(threshold=0.0)
    Xtr_v = vt.fit_transform(Xtr)
    kept = Xtr.columns[vt.get_support()].tolist()
    Xtr_v = pd.DataFrame(Xtr_v, columns=kept, index=Xtr.index)
    Xte_v = pd.DataFrame(vt.transform(Xte), columns=kept, index=Xte.index)

    # Simpan artifacts
    dump(imp, ART / f"{dataset}_smote_imputer.joblib")
    dump(vt, ART / f"{dataset}_smote_varselector.joblib")
    dump(kept, ART / f"{dataset}_smote_kept_features.joblib")

    # Report sesudah
    _nan_report(Xtr_v, "train_after", dataset)
    _nan_report(Xte_v, "test_after", dataset)

    return Xtr_v, Xte_v, kept


# ---------- SMOTE variants ----------
def make_smote(variant: str, random_state: int, k_neighbors: int, sampling_strategy):
    """
    Factory untuk varian SMOTE:
    - 'smote'
    - 'smote_tomek'
    - 'smoteenn'
    """
    variant = variant.lower()
    if variant == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
    if variant == "smote_tomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        return SMOTETomek(random_state=random_state, smote=sm)
    if variant == "smoteenn":
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE

        sm = SMOTE(
            random_state=random_state,
            k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
        )
        return SMOTEENN(random_state=random_state, smote=sm)
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
def plot_pie(before: Dict[int, int], after: Dict[int, int], title: str, out_png: Path) -> None:
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


def plot_bar(before: Dict[int, int], after: Dict[int, int], title: str, out_png: Path) -> None:
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


# ---------- Helpers ----------
def _variant_tag(variant: str) -> str:
    """
    Nama suffix aman untuk file output per varian.
    Konsisten dengan stacking_pipeline:

    smote       -> 'smote'
    smote_tomek -> 'smote_tomek'
    smoteenn    -> 'smote_enn'
    """
    v = variant.lower()
    if v == "smote":
        return "smote"
    if v == "smote_tomek":
        return "smote_tomek"
    if v == "smoteenn":
        return "smote_enn"
    raise ValueError("Unknown variant")


# ---------- Visualisasi konseptual SMOTE (simulasi 2D) ----------
def _make_simulated_2d(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Buat dataset 2D imbalanced untuk ilustrasi SMOTE."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=600,
        n_features=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


def visualize_smote_mechanism(random_state: int = 42) -> None:
    """
    Gambar garis interpolasi SMOTE antara satu titik minority dan tetangganya.
    Output:
      - figures/smote_mechanism.png
      - tables/smote_mechanism_points.csv
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        print("[WARN] sklearn.neighbors tidak tersedia; skip smote_mechanism.png")
        return

    X, y = _make_simulated_2d(random_state)
    X_min = X[y == 1]
    if len(X_min) < 2:
        return

    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(X_min)
    dists, idxs = nn.kneighbors(X_min)
    base_idx = 0
    neigh_idx = idxs[base_idx, 1]

    x_i = X_min[base_idx]
    x_n = X_min[neigh_idx]
    alpha = 0.4
    x_new = x_i + alpha * (x_n - x_i)

    df_pts = pd.DataFrame(
        [
            {"role": "minority_base", "x1": x_i[0], "x2": x_i[1]},
            {"role": "minority_neighbor", "x1": x_n[0], "x2": x_n[1]},
            {"role": "synthetic_point", "x1": x_new[0], "x2": x_new[1]},
        ]
    )
    _savetab(df_pts, TAB / "smote_mechanism_points.csv")

    plt.figure(figsize=(6, 5))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=15, alpha=0.5, label="Majority")
    plt.scatter(X_min[:, 0], X_min[:, 1], s=25, alpha=0.8, label="Minority", color="orange")

    plt.scatter(x_i[0], x_i[1], s=80, marker="*", color="red", label="Base minority")
    plt.scatter(x_n[0], x_n[1], s=80, marker="X", color="green", label="Neighbor")
    plt.scatter(x_new[0], x_new[1], s=80, marker="^", color="purple", label="Synthetic")

    plt.plot([x_i[0], x_n[0]], [x_i[1], x_n[1]], "k--", alpha=0.7)
    plt.plot([x_i[0], x_new[0]], [x_i[1], x_new[1]], "k-.", alpha=0.7)

    plt.title("Illustration of SMOTE Data Generation (2D Simulation)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best")
    _savefig(FIG / "smote_mechanism.png")


def visualize_minority_regions(random_state: int = 42) -> None:
    """
    Kategorisasi minority menjadi Safe / Borderline / Abnormal (rare+outlier)
    berdasarkan tetangga terdekat (seperti konsep Borderline-SMOTE).
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        print("[WARN] sklearn.neighbors tidak tersedia; skip minority_regions.png")
        return

    X, y = _make_simulated_2d(random_state)
    k = 6
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    dists, idxs = nn.kneighbors(X)

    cats: List[str] = []
    for i, lab in enumerate(y):
        if lab != 1:
            cats.append("majority")
            continue
        neigh_idx = idxs[i, 1:]
        neigh_labels = y[neigh_idx]
        n_majority = int((neigh_labels == 0).sum())
        if n_majority == 0:
            cats.append("safe")
        elif n_majority < k // 2:
            cats.append("borderline")
        else:
            cats.append("abnormal")

    cats_arr = np.array(cats)
    counts = (
        pd.Series(cats_arr)
        .value_counts()
        .rename_axis("region_type")
        .reset_index(name="count")
    )
    _savetab(counts, TAB / "smote_minority_region_counts.csv")

    plt.figure(figsize=(6.5, 5))
    plt.scatter(
        X[cats_arr == "majority", 0],
        X[cats_arr == "majority", 1],
        s=15,
        alpha=0.5,
        label="Majority class",
        color="steelblue",
    )
    plt.scatter(
        X[cats_arr == "safe", 0],
        X[cats_arr == "safe", 1],
        s=35,
        alpha=0.9,
        label="Minority - Safe",
        color="green",
    )
    plt.scatter(
        X[cats_arr == "borderline", 0],
        X[cats_arr == "borderline", 1],
        s=35,
        alpha=0.9,
        label="Minority - Borderline",
        color="orange",
    )
    plt.scatter(
        X[cats_arr == "abnormal", 0],
        X[cats_arr == "abnormal", 1],
        s=40,
        alpha=0.9,
        label="Minority - Abnormal",
        color="red",
    )

    plt.title("Minority Regions: Safe / Borderline / Abnormal (2D Simulation)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc="best", fontsize=8)
    _savefig(FIG / "minority_regions.png")


def visualize_simulated_oversampling_grid(random_state: int = 42) -> None:
    """
    Grid perbandingan sebelum & sesudah oversampling:
    Baseline, SMOTE, SMOTE+Tomek, SMOTEENN.
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.combine import SMOTETomek, SMOTEENN
    except Exception:
        print("[WARN] imblearn tidak tersedia; skip simulated_oversampling_grid.png")
        return

    X, y = _make_simulated_2d(random_state)

    variants: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"Baseline": (X, y)}

    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_sm, y_sm = sm.fit_resample(X, y)
    variants["SMOTE"] = (X_sm, y_sm)

    smt = SMOTETomek(
        random_state=random_state,
        smote=SMOTE(random_state=random_state, k_neighbors=5),
    )
    X_smt, y_smt = smt.fit_resample(X, y)
    variants["SMOTE+Tomek"] = (X_smt, y_smt)

    sme = SMOTEENN(
        random_state=random_state,
        smote=SMOTE(random_state=random_state, k_neighbors=5),
    )
    X_sme, y_sme = sme.fit_resample(X, y)
    variants["SMOTEENN"] = (X_sme, y_sme)

    rows = []
    for name, (Xv, yv) in variants.items():
        vc = pd.Series(yv).value_counts().sort_index()
        rows.append(
            {
                "variant": name,
                "n_majority": int(vc.get(0, 0)),
                "n_minority": int(vc.get(1, 0)),
            }
        )
    _savetab(pd.DataFrame(rows), TAB / "smote_simulated_summary.csv")

    plt.figure(figsize=(10, 8))
    titles = ["Baseline", "SMOTE", "SMOTE+Tomek", "SMOTEENN"]
    for i, t in enumerate(titles, 1):
        Xv, yv = variants[t]
        plt.subplot(2, 2, i)
        plt.scatter(
            Xv[yv == 0, 0],
            Xv[yv == 0, 1],
            s=10,
            alpha=0.5,
            label="Majority",
            color="steelblue",
        )
        plt.scatter(
            Xv[yv == 1, 0],
            Xv[yv == 1, 1],
            s=15,
            alpha=0.8,
            label="Minority",
            color="orange",
        )
        plt.title(t)
        if i in (1, 3):
            plt.ylabel("Feature 2")
        if i in (3, 4):
            plt.xlabel("Feature 1")
        if i == 1:
            plt.legend(loc="best", fontsize=8)

    plt.suptitle("Simulated Oversampling: Baseline vs SMOTE Variants", y=0.95)
    _savefig(FIG / "simulated_oversampling_grid.png")


def make_smote_conceptual_visuals(random_state: int = 42) -> None:
    """Wrapper untuk memanggil semua visual konseptual SMOTE (simulasi 2D)."""
    print("[SMOTE] Generating conceptual SMOTE visuals (simulation)…")
    try:
        visualize_smote_mechanism(random_state=random_state)
        visualize_minority_regions(random_state=random_state)
        visualize_simulated_oversampling_grid(random_state=random_state)
    except Exception as e:
        print(f"[WARN] Gagal membuat visual SMOTE simulasi: {e!r}")
    else:
        print("[SMOTE] Conceptual visuals saved to:", FIG)


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
) -> None:
    tag = _variant_tag(variant)
    print(f"[SMOTE] Start -> {dataset} [{variant}]")

    # Output paths (umum & per-varian)
    p_train = PROCESSED / f"{dataset}_train.csv"
    p_test = PROCESSED / f"{dataset}_test.csv"

    p_train_smote = PROCESSED / f"{dataset}_train_{tag}.csv"
    p_sum = TAB / f"{dataset}_{tag}_summary.csv"
    p_pie = FIG / f"{dataset}_{tag}_pie.png"
    p_bar = FIG / f"{dataset}_{tag}_bar.png"
    p_cfg = ART / f"{dataset}_{tag}_config.joblib"

    # Jika semua output utama sudah ada dan tidak overwrite → reuse
    if (
        all(p.exists() for p in (p_train, p_test, p_train_smote, p_sum, p_pie, p_bar, p_cfg))
        and not overwrite
    ):
        print(
            f"[INFO] Reusing existing outputs for '{dataset}' [{variant}] "
            "(use --overwrite untuk regenerasi)."
        )
        return

    # 1) Load FE / FE_OHE
    df, feat_cols = load_fe_dataset(dataset, auto_ohe=use_ohe_auto)
    if "tsu" not in df.columns:
        raise KeyError(f"[SMOTE] '{dataset}' tidak memiliki kolom 'tsu'.")

    # 2) Split stratified
    X_train, X_test, y_train, y_test = stratified_split(
        df,
        target="tsu",
        test_size=test_size,
        random_state=random_state,
    )

    # 3) CLEAN + IMPUTE (fit di train → transform test)
    X_train_clean, X_test_clean, kept_cols = clean_and_impute(
        X_train,
        X_test,
        dataset,
        strategy="median",
    )

    # 4) Adjust k_neighbors (berdasarkan jumlah minoritas di train)
    k_adj = adjust_k_neighbors(y_train, k_neighbors)

    # 5) Fit-resample di training
    smote_obj = make_smote(
        variant=variant,
        random_state=random_state,
        k_neighbors=k_adj,
        sampling_strategy=sampling_strategy,
    )
    X_train_res, y_train_res = smote_obj.fit_resample(X_train_clean, y_train)

    # 6) Simpan dataset (umum & hasil resampling per varian)
    df_train = pd.concat(
        [X_train_clean.reset_index(drop=True), y_train.reset_index(drop=True)],
        axis=1,
    )
    df_test = pd.concat(
        [X_test_clean.reset_index(drop=True), y_test.reset_index(drop=True)],
        axis=1,
    )
    df_train_res = pd.concat(
        [pd.DataFrame(X_train_res, columns=X_train_clean.columns), pd.Series(y_train_res, name="tsu")],
        axis=1,
    )

    df_train.to_csv(p_train, index=False)
    df_test.to_csv(p_test, index=False)
    df_train_res.to_csv(p_train_smote, index=False)

    # 7) Ringkasan + plots
    before = y_train.value_counts().sort_index().to_dict()
    after = pd.Series(y_train_res).value_counts().sort_index().to_dict()

    summary = pd.DataFrame(
        {
            "class": [0, 1],
            "train_before": [before.get(0, 0), before.get(1, 0)],
            "train_after": [after.get(0, 0), after.get(1, 0)],
        }
    )
    _savetab(summary, p_sum)

    plot_pie(before, after, title=f"{dataset.title()} - {variant}", out_png=p_pie)
    plot_bar(before, after, title=f"{dataset.title()} - {variant}", out_png=p_bar)

    # 8) Simpan config/artifacts
    dump(
        {
            "variant": variant,
            "variant_tag": tag,
            "random_state": random_state,
            "k_neighbors_requested": k_neighbors,
            "k_neighbors_used": k_adj,
            "sampling_strategy": sampling_strategy,
            "test_size": test_size,
            "feature_columns_before": feat_cols,
            "feature_columns_after_clean": kept_cols,
            "y_train_before_counts": before,
            "y_train_after_counts": after,
            "outputs": {
                "train": str(p_train),
                "test": str(p_test),
                "train_resampled": str(p_train_smote),
                "summary_csv": str(p_sum),
                "pie_png": str(p_pie),
                "bar_png": str(p_bar),
            },
        },
        p_cfg,
    )

    print(
        f"[SMOTE] {dataset} [{variant}]: done | "
        f"train={p_train.name}, test={p_test.name}, smote={p_train_smote.name}"
    )


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="SMOTE pipeline (tectonic/volcanic) with anti-leakage + imputation."
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["tectonic", "volcanic"],
        help="dataset list: tectonic volcanic",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["smote", "smote_tomek", "smoteenn"],
        choices=["smote", "smote_tomek", "smoteenn"],
        help="daftar varian SMOTE yang akan dijalankan (default: semua).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate outputs",
    )
    ap.add_argument(
        "--no-auto-ohe",
        action="store_true",
        help="jangan OHE otomatis jika *_fe_ohe.csv tidak ada",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--k-neighbors", type=int, default=5)
    ap.add_argument(
        "--sampling-strategy",
        type=str,
        default="auto",
        help="auto | float (0<r<=1) | dict JSON-like, contoh: '{\"1\": 2000}'",
    )
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
        for var in args.variants:
            run_one(
                dataset=ds,
                overwrite=args.overwrite,
                use_ohe_auto=(not args.no_auto_ohe),
                test_size=args.test_size,
                random_state=args.random_state,
                k_neighbors=args.k_neighbors,
                sampling_strategy=sampling_strategy,
                variant=var,
            )

    # Visualisasi konsep SMOTE (simulasi 2D) sekali di akhir
    make_smote_conceptual_visuals(random_state=args.random_state)

    print(
        f"[DONE] SMOTE pipeline selesai.\n"
        f" - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}\n - DATA: {PROCESSED}"
    )


if __name__ == "__main__":
    main()