from __future__ import annotations
# cSpell:ignore tsunami tectonic volcanic

"""
Preprocessing pipeline untuk dataset gabungan tsunami (multiclass 0/1/2).

Tugas utama:
- Membaca dataset gabungan (default: data/processed/events_fe.csv).
- Filter tahun (>= 1900).
- Membuat label multi-class `label`:
      0 = non-tsunami
      1 = tsunami tektonik
      2 = tsunami vulkanik
  menggunakan:
      (a) jika tersedia: kombinasi kolom tsunami_flag & cause, atau
      (b) fallback: kombinasi `tsu` + `source_domain` (tectonic / volcanic).
- Memilih fitur fisik yang sederhana dan umum di literatur:
      * magnitude / mag
      * depth
      * latitude, longitude
      * VEI
      * elevation
      * alert (kategorikal → OHE)
      * sig (significance)
- Cleaning & imputasi:
      * ganti inf/-inf → NaN
      * imputasi median untuk numerik
      * imputasi modus + OHE untuk kategorikal
      * drop fitur zero-variance
- Split train/test (stratified 0/1/2).
- Menyimpan:
      * data/processed/events_train.csv
      * data/processed/events_test.csv

Catatan:
- SMOTE TIDAK dilakukan di sini (lihat smote_pipeline.py).
- Scaling SVM dilakukan di dalam model (pipeline di stacking_pipeline.py),
  bukan di level dataset global.
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from joblib import dump
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


# --------- Helper untuk OHE (kompatibel sklearn lama/baru) ---------
def _one_hot_encoder() -> OneHotEncoder:
    """OneHotEncoder yang aman untuk berbagai versi scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # versi lama
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ---------- Logging & util kecil ----------
def _log(msg: str) -> None:
    print(msg, flush=True)


def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _savetab(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _replace_non_finite(df: pd.DataFrame) -> pd.DataFrame:
    """Ganti inf/-inf dengan NaN (imputasi dilakukan terpisah)."""
    return df.replace([np.inf, -np.inf], np.nan)


# ---------- Deteksi nama kolom fleksibel ----------
def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Cari kolom pertama yang muncul dari daftar kandidat.
    Return None kalau tidak ada.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------- Labeling 0/1/2 ----------
def _build_tsunami_label(df: pd.DataFrame) -> Tuple[pd.Series, str, str]:
    """
    Bangun label multi-class 0/1/2 untuk prediksi tsunami.

    Skema utama (jika kolom tersedia eksplisit):
      - gunakan kombinasi:
            tsunami_flag ∈ {0/1, yes/no, dsb.}
            cause        ∈ {earthquake, volcanic eruption, ...}

        Aturan:
          * tsunami_flag = False/0                 -> 0 (non-tsunami)
          * tsunami_flag = True & cause ~ earthquake   -> 1 (tsunami tektonik)
          * tsunami_flag = True & cause ~ volcanic     -> 2 (tsunami vulkanik)
          * lainnya                                     -> 0

    Fallback (jika tidak ada tsunami_flag/cause):
      - gunakan kombinasi:
            tsu            : flag tsunami 0/1 dari sumber resmi (NOAA/GVP)
            source_domain  : 'tectonic' / 'volcanic' (asal event)

        Aturan:
          * tsu == 0                                  -> 0
          * tsu == 1 & source_domain ~ 'tect'         -> 1
          * tsu == 1 & source_domain ~ 'volc'         -> 2
          * selain itu                                 -> 0
    """

    # -----------------------------
    # 1) Coba pakai tsunami_flag + cause (jika ada)
    # -----------------------------
    tsunami_flag_candidates = ["tsunami_flag", "tsunami", "tsu_flag", "Tsu"]
    cause_candidates = ["cause", "cause_name", "origin", "source", "trigger"]

    flag_col = _find_column(df, tsunami_flag_candidates)
    cause_col = _find_column(df, cause_candidates)

    if flag_col is not None and cause_col is not None:
        _log(f"[Pre] Menggunakan kolom tsunami_flag='{flag_col}', cause='{cause_col}'.")

        flag_raw = df[flag_col]
        cause_raw = df[cause_col]

        flag_norm = flag_raw.astype(str).str.lower().str.strip()
        cause_norm = cause_raw.astype(str).str.lower().str.strip()

        yes_values = {"1", "y", "yes", "true", "tsunami", "t"}
        flag_yes = flag_norm.isin(yes_values) | (
            pd.to_numeric(flag_norm, errors="coerce") > 0
        )

        quake_terms = {"earthquake", "eq", "tectonic", "seismic", "quake"}
        volc_terms = {"volcano", "volcanic", "eruption", "volcanic eruption"}

        cause_quake = cause_norm.apply(
            lambda s: any(t in s for t in quake_terms) if isinstance(s, str) else False
        )
        cause_volc = cause_norm.apply(
            lambda s: any(t in s for t in volc_terms) if isinstance(s, str) else False
        )

        label = np.zeros(len(df), dtype=int)
        label[flag_yes & cause_quake] = 1
        label[flag_yes & cause_volc] = 2

        label_series = pd.Series(label, index=df.index, name="label")
        return label_series, flag_col, cause_col

        # -----------------------------
    # 2) Fallback: tsu + source_domain (desain events_fe)
    # -----------------------------
    tsu_col = _find_column(df, ["tsu", "Tsu", "TSU"])
    dom_col = _find_column(df, ["source_domain", "Source_Domain", "sourceDomain"])

    if tsu_col is not None and dom_col is not None:
        _log(
            f"[Pre] Fallback label: menggunakan kolom '{tsu_col}' + '{dom_col}'."
        )

        tsu_raw = df[tsu_col].fillna(0)
        tsu_bin = pd.to_numeric(tsu_raw, errors="coerce").fillna(0).astype(int)
        dom = df[dom_col].astype(str).str.lower()

        label = np.zeros(len(df), dtype=int)
        # tsunami tektonik
        label[(tsu_bin == 1) & dom.str.contains("tect")] = 1
        # tsunami vulkanik
        label[(tsu_bin == 1) & dom.str.contains("volc")] = 2

        label_series = pd.Series(label, index=df.index, name="label")
        return label_series, tsu_col, dom_col

    # -----------------------------
    # 3) Jika dua-duanya tidak tersedia → error eksplisit
    # -----------------------------
    raise KeyError(
        "Kolom tsunami_flag/cause maupun fallback (tsu + source_domain) tidak ditemukan.\n"
        f"  Kolom tersedia: {list(df.columns)}"
    )


# ---------- Seleksi fitur ----------

# Fitur yang ingin dipakai (kandidat nama kolom yang lazim muncul)
FEATURE_CANDIDATES: List[str] = [
    # --- fitur utama tektonik ---
    "magnitude", "mag", "magMw", "mag_ml",
    "depth", "depth_km", "focal_depth",
    # transformasi dasar (kalau sudah dibuat di FE)
    "depth_log1p", "mag_sq",

    # --- koordinat & posisi ---
    "latitude", "lat",
    "longitude", "lon",
    "abs_lat",          # |latitude|
    "is_tropic",        # boolean zona tropis
    "lat_lon_prod",     # kombinasi posisi

    # --- fitur utama vulkanik ---
    "vei", "VEI",
    "elevation", "elev", "elevation_m",

    # --- kombinasi & transformasi vulkanik (kalau ada) ---
    "elev_log1p", "vei_sq",

    # --- fitur jarak ke pantai (butuh FE dengan --coast) ---
    "distance_to_coast_km",
    "dist_coast_log1p",
    "near_coast_50km",
    "near_coast_100km",

    # --- zona subduksi ---
    "is_subduction_zone",

    # --- fitur tambahan dari katalog jika tersedia ---
    "alert",
    "sig",
]

# Kolom yang jelas bukan fitur model
NON_FEATURE_CANDIDATES: List[str] = [
    "tsunami_flag", "tsu_flag", "tsunami",
    "cause", "source_cause",
    "tsu", "source_domain",      # dari events_fe
    "label", "tsunami_label",
    "year", "Year",
    "event_id", "id",
]



def _select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Pilih subset fitur yang relevan untuk model:

    - Langkah utama:
        * Ambil semua kolom yang cocok dengan FEATURE_CANDIDATES.
    - Jika hasilnya kosong/terlalu sedikit (misal dataset lain):
        * Fallback ke semua kolom numerik, kecuali target & NON_FEATURE_CANDIDATES.

    Return:
        X_features : DataFrame fitur
        num_cols   : daftar kolom numerik
        cat_cols   : daftar kolom kategorikal
    """
    non_feature_set = set(NON_FEATURE_CANDIDATES + [target_col])

    # 1) kandidat fitur yang tersedia di df
    candidates_set = set(FEATURE_CANDIDATES)
    feature_cols = [c for c in df.columns if c in candidates_set and c != target_col]
    feature_cols = [c for c in feature_cols if c not in non_feature_set]

    # minimal 3 fitur; kalau kurang → fallback ke semua numerik
    if len(feature_cols) < 3:
        _log(
            "[Pre] WARNING: jumlah fitur dari FEATURE_CANDIDATES terlalu sedikit "
            f"({len(feature_cols)}). Fallback ke semua kolom numerik."
        )
        num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in num_cols_all if c not in non_feature_set]

    X = df[feature_cols].copy()

    # deteksi numerik vs kategorikal
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    _log(f"[Pre] Fitur terpilih ({len(feature_cols)} kolom): {feature_cols}")
    _log(f"[Pre]  - numerik    : {num_cols}")
    _log(f"[Pre]  - kategorikal: {cat_cols}")

    return X, num_cols, cat_cols


# ---------- Imputasi + OHE + VarianceThreshold ----------
def _impute_and_encode(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    dataset_name: str,
    num_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Imputasi & encoding:
    - Numerik   : SimpleImputer(median)
    - Kategorik : SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    - Gabungkan numerik + OHE, lalu VarianceThreshold(0.0) untuk drop zero-variance.

    Artifacts (untuk serving nanti) disimpan di artifacts/:
        - {dataset}_num_imputer.joblib
        - {dataset}_cat_imputer.joblib  (jika ada kategorikal)
        - {dataset}_ohe_encoder.joblib  (jika ada kategorikal)
        - {dataset}_var_selector.joblib
        - {dataset}_feature_names.joblib
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Ganti inf/-inf → NaN (jaga-jaga)
    X_train = _replace_non_finite(X_train)
    X_test = _replace_non_finite(X_test)

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent") if cat_cols else None
    ohe = _one_hot_encoder() if cat_cols else None

    # --- Numerik ---
    if num_cols:
        Xtr_num = num_imputer.fit_transform(X_train[num_cols])
        Xte_num = num_imputer.transform(X_test[num_cols])
        num_feature_names = num_cols
    else:
        Xtr_num = np.empty((len(X_train), 0))
        Xte_num = np.empty((len(X_test), 0))
        num_feature_names = []

    # --- Kategorikal + OHE ---
    if cat_cols and cat_imputer is not None and ohe is not None:
        Xtr_cat_imp = cat_imputer.fit_transform(X_train[cat_cols].astype("object"))
        Xte_cat_imp = cat_imputer.transform(X_test[cat_cols].astype("object"))

        Xtr_cat_ohe = ohe.fit_transform(Xtr_cat_imp)
        Xte_cat_ohe = ohe.transform(Xte_cat_imp)

        ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    else:
        Xtr_cat_ohe = np.empty((len(X_train), 0))
        Xte_cat_ohe = np.empty((len(X_test), 0))
        ohe_feature_names: List[str] = []

    # --- Gabungkan numerik + kategorikal ---
    Xtr_all = np.hstack([Xtr_num, Xtr_cat_ohe])
    Xte_all = np.hstack([Xte_num, Xte_cat_ohe])
    all_feature_names = num_feature_names + ohe_feature_names

    # --- Drop zero-variance features ---
    if Xtr_all.shape[1] > 0:
        vt = VarianceThreshold(threshold=0.0)
        Xtr_sel = vt.fit_transform(Xtr_all)
        Xte_sel = vt.transform(Xte_all)

        kept_mask = vt.get_support()
        kept_names = [
            name for flag, name in zip(kept_mask, all_feature_names) if flag
        ]
    else:
        vt = VarianceThreshold(threshold=0.0)
        Xtr_sel = Xtr_all
        Xte_sel = Xte_all
        kept_names: List[str] = []

    X_train_proc = pd.DataFrame(Xtr_sel, columns=kept_names, index=X_train.index)
    X_test_proc = pd.DataFrame(Xte_sel, columns=kept_names, index=X_test.index)

    # Simpan artifacts untuk serving/API
    dump(num_imputer, ART / f"{dataset_name}_num_imputer.joblib")
    if cat_imputer is not None:
        dump(cat_imputer, ART / f"{dataset_name}_cat_imputer.joblib")
    if ohe is not None:
        dump(ohe, ART / f"{dataset_name}_ohe_encoder.joblib")
    dump(vt, ART / f"{dataset_name}_var_selector.joblib")
    dump(kept_names, ART / f"{dataset_name}_feature_names.joblib")

    _log(f"[Pre] Fitur akhir setelah VarianceThreshold: {len(kept_names)} kolom.")
    return X_train_proc, X_test_proc


# ---------- Pipeline utama ----------
def preprocess_events(
    input_path: Path,
    dataset_name: str = "events",
    year_min: int = 1900,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Pipeline utama:
        - load events_fe.csv (atau file lain yang ditentukan)
        - filter tahun >= year_min
        - labeling 0/1/2 → kolom 'label'
        - pilih fitur sederhana
        - imputasi + OHE + drop zero-variance
        - split train/test (stratified)
        - simpan events_train.csv & events_test.csv
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"[Pre] File input '{input_path}' tidak ditemukan.\n"
            "Pastikan feature_engineering.py sudah menghasilkan events_fe.csv."
        )

    _log(f"[Pre] Load: {input_path}")
    df = _read_csv(input_path)

    # 1) deteksi kolom tahun & filter >= year_min
    year_col = _find_column(df, ["year", "Year", "YEAR"])
    if year_col is None:
        _log("[Pre] WARNING: kolom 'year' tidak ditemukan, skip filter tahun.")
    else:
        _log(f"[Pre] Filter tahun: kolom='{year_col}' >= {year_min}")
        df = df[df[year_col] >= year_min].copy()

    # 2) buat label 0/1/2
    label_series, flag_col, cause_col = _build_tsunami_label(df)
    df["label"] = label_series

    _log(
        "[Pre] Distribusi label (0=non,1=tektonik,2=vulkanik): "
        + repr(df["label"].value_counts().sort_index().to_dict())
    )

    # 3) pilih fitur (pakai df dengan semua kolom, target 'label')
    X_all, num_cols, cat_cols = _select_feature_columns(df=df, target_col="label")
    y = df["label"].astype("int64")

    # 4) split stratified train/test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    _log(
        f"[Pre] Split: train={X_train_raw.shape}, test={X_test_raw.shape}, "
        f"test_size={test_size}"
    )

    # 5) imputasi + OHE + drop zero-variance
    X_train_proc, X_test_proc = _impute_and_encode(
        X_train=X_train_raw,
        X_test=X_test_raw,
        dataset_name=dataset_name,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    # 6) gabungkan kembali dengan label, simpan ke CSV
    train_df = X_train_proc.copy()
    train_df["label"] = y_train.values

    test_df = X_test_proc.copy()
    test_df["label"] = y_test.values

    train_path = PROCESSED / f"{dataset_name}_train.csv"
    test_path = PROCESSED / f"{dataset_name}_test.csv"

    _savetab(train_df, train_path)
    _savetab(test_df, test_path)

    # 7) simpan ringkasan untuk laporan
    label_counts = (
        y.value_counts()
        .sort_index()
        .rename("count")
        .reset_index()
        .rename(columns={"index": "label"})
    )
    _savetab(label_counts, TAB / f"{dataset_name}_label_distribution.csv")

    _log(f"[Pre] Saved train  : {train_path}")
    _log(f"[Pre] Saved test   : {test_path}")
    _log(f"[Pre] Label summary: {TAB / f'{dataset_name}_label_distribution.csv'}")


# ---------- CLI ----------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocessing events (multiclass 0/1/2) untuk prediksi tsunami."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED / "events_fe.csv"),
        help="Path input CSV (default: data/processed/events_fe.csv).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="events",
        help="Nama dataset/prefix output (default: 'events').",
    )
    parser.add_argument(
        "--year-min",
        type=int,
        default=1900,
        help="Tahun minimum data yang disertakan (default=1900).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi test set (default=0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed random untuk split & imputasi (default=42).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    _log("[Pre] Preprocessing pipeline start …")
    preprocess_events(
        input_path=input_path,
        dataset_name=args.dataset_name,
        year_min=args.year_min,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    _log(
        f"[DONE] Preprocessing selesai.\n"
        f" - INPUT : {input_path}\n"
        f" - DATA  : {PROCESSED}\n"
        f" - TAB   : {TAB}\n"
        f" - ART   : {ART}"
    )


if __name__ == "__main__":
    main()