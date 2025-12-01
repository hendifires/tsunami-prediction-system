from __future__ import annotations
# cSpell:ignore tectonic volcanic tsunami

"""
Feature Engineering sederhana untuk domain tektonik & vulkanik.

Fokus desain baru (selaras dengan metodologi tesis):

1) Menjaga FE tetap:
   - ringkas,
   - mudah dijelaskan,
   - dan tidak “mengintervensi” pipeline utama yang berbasis fitur fisik sederhana.

2) Fitur inti yang dipertahankan (sejalan dengan penelitian sebelumnya):
   - Tektonik:
       * magnitude (mag / magnitude / Mw)
       * depth
       * latitude, longitude
       * sig (significance, numerik jika ada)
       * alert / region / country / area / location (kategorikal jika ada)
   - Vulkanik:
       * VEI
       * elevation
       * latitude, longitude
       * (opsional) eq (jumlah gempa sekitar gunung)

3) Feature engineering tambahan hanya untuk EDA:
   - Transformasi sederhana: log1p, kuadrat, rasio.
   - TIDAK lagi membuat fitur kompleks seperti:
       * jarak ke pantai,
       * subduction flag,
       * time-gap antar kejadian,
       * encoding siklis waktu (sin/cos),
       * fitur-fitur geospasial berat.
   - Ini menjaga pipeline utama tetap bersih, sementara tabel/plot Pearson & RFE
     tetap ada sebagai dokumentasi analisis fitur.

4) Output:
   - data/processed/tectonic_fe.csv
   - data/processed/volcanic_fe.csv
   - data/processed/events_fe.csv              (gabungan tektonik+vulkanik, untuk preprocessing.py)
   - (opsional) *_fe_ohe.csv jika --materialize-ohe
   - artifacts/*_fe_num_cols.joblib
   - artifacts/*_fe_cat_cols.joblib
   - artifacts/*_ohe_encoder.joblib (opsional)
   - reports/tables/*.csv & reports/figures/*.png untuk EDA:
       * sebelum/sesudah OHE
       * Pearson top-k
       * RFE top-k
"""

import argparse
from contextlib import suppress
from pathlib import Path
from typing import Any, List, Optional, Tuple

# ========== MATPLOTLIB NON-GUI ==========
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import dump  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

# (Opsional) seaborn untuk tampilan plot (tidak wajib)
with suppress(Exception):
    import seaborn as sns  # type: ignore  # noqa: E401

    sns.set_theme(style="whitegrid")

# ====================================================
# PATH SETUP
# ====================================================
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
ART = ROOT / "artifacts"
for p in (PROCESSED, TAB, FIG, ART):
    p.mkdir(parents=True, exist_ok=True)


# ====================================================
# UTILITIES
# ====================================================
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


def _num(s: Any) -> pd.Series:
    """
    Konversi ke numerik yang tahan terhadap:
    - Series biasa,
    - DataFrame dengan kolom duplikat (ambil kolom pertama),
    - list/ndarray/Index.

    Output selalu Series 1D dengan panjang = jumlah baris.
    """
    # Jika DataFrame (mis. karena nama kolom duplikat), pakai kolom pertama saja
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return pd.Series([np.nan] * len(s), index=s.index)
        s = s.iloc[:, 0]

    # Jika array-like, buat Series
    if isinstance(s, (pd.Index, np.ndarray, list, tuple)):
        s = pd.Series(s)

    return pd.to_numeric(s, errors="coerce")


def _one_hot_encoder() -> OneHotEncoder:
    """
    Helper kecil agar kompatibel dengan berbagai versi scikit-learn:
    - sparse_output (versi baru)
    - sparse (versi lama)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _clip_physical_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamp nilai fisik ke rentang wajar agar stabil (mengurangi outlier ekstrem).
    FE di sini hanya untuk EDA; pipeline utama nanti memilih subset fitur inti.
    """
    out = df.copy()
    if "mag" in out.columns:
        out["mag"] = _num(out["mag"]).clip(lower=0, upper=10)
    if "magnitude" in out.columns:
        out["magnitude"] = _num(out["magnitude"]).clip(lower=0, upper=10)
    if "depth" in out.columns:
        out["depth"] = _num(out["depth"]).clip(lower=0)
    if "vei" in out.columns:
        out["vei"] = _num(out["vei"]).clip(lower=0, upper=8)
    if "elevation" in out.columns:
        out["elevation"] = _num(out["elevation"]).clip(
            lower=-500,
            upper=9000,
        )
    if "eq" in out.columns:
        out["eq"] = _num(out["eq"]).clip(lower=0)
    if "sig" in out.columns:
        out["sig"] = _num(out["sig"]).clip(lower=0)
    return out


def _normalize_mag_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Satukan berbagai nama kolom magnitudo menjadi 'mag' jika memungkinkan.
    """
    out = df.copy()
    mag_candidates = ["mag", "magnitude", "magMw", "mag_ml", "Mw", "Mag"]
    mag_col = next((c for c in mag_candidates if c in out.columns), None)
    if mag_col and mag_col != "mag":
        out["mag"] = _num(out[mag_col])
    return out


def _normalize_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lat" in out.columns and "latitude" not in out.columns:
        out["latitude"] = _num(out["lat"])
    if "lon" in out.columns and "longitude" not in out.columns:
        out["longitude"] = _num(out["lon"])
    if "Latitude" in out.columns and "latitude" not in out.columns:
        out["latitude"] = _num(out["Latitude"])
    if "Longitude" in out.columns and "longitude" not in out.columns:
        out["longitude"] = _num(out["Longitude"])
    return out


def _standardize_tectonic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarisasi nama kolom penting untuk domain tektonik
    (sesuai struktur katalog NOAA yang kamu kirim).
    """
    out = df.copy()
    rename: dict[str, str] = {}

    # magnitudo -> mag
    if "Mag" in out.columns and "mag" not in out.columns:
        rename["Mag"] = "mag"
    if "Mw" in out.columns and "mag" not in out.columns:
        rename["Mw"] = "mag"

    # depth
    if "Focal Depth (km)" in out.columns and "depth" not in out.columns:
        rename["Focal Depth (km)"] = "depth"

    # koordinat
    if "Latitude" in out.columns and "latitude" not in out.columns:
        rename["Latitude"] = "latitude"
    if "Longitude" in out.columns and "longitude" not in out.columns:
        rename["Longitude"] = "longitude"

    # flag tsunami
    if "Tsu" in out.columns and "tsu" not in out.columns:
        rename["Tsu"] = "tsu"

    # tahun
    if "Year" in out.columns and "year" not in out.columns:
        rename["Year"] = "year"

    if rename:
        out = out.rename(columns=rename)

    # buang kolom duplikat jika ada
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _standardize_volcanic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarisasi nama kolom penting untuk domain vulkanik
    (sesuai struktur katalog GVP yang kamu kirim).
    """
    out = df.copy()
    rename: dict[str, str] = {}

    # VEI, elevasi, eq
    if "VEI" in out.columns and "vei" not in out.columns:
        rename["VEI"] = "vei"
    if "Elevation (m)" in out.columns and "elevation" not in out.columns:
        rename["Elevation (m)"] = "elevation"
    if "Eq" in out.columns and "eq" not in out.columns:
        rename["Eq"] = "eq"

    # koordinat
    if "Latitude" in out.columns and "latitude" not in out.columns:
        rename["Latitude"] = "latitude"
    if "Longitude" in out.columns and "longitude" not in out.columns:
        rename["Longitude"] = "longitude"

    # flag tsunami
    if "Tsu" in out.columns and "tsu" not in out.columns:
        rename["Tsu"] = "tsu"

    # tahun
    if "Year" in out.columns and "year" not in out.columns:
        rename["Year"] = "year"

    if rename:
        out = out.rename(columns=rename)

    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


# ====================================================
# FEATURE ENGINEERING CORE (RINGAN)
# ====================================================
def engineer_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur dasar yang berlaku untuk kedua domain (tektonik & vulkanik):

    - normalisasi nama kolom magnitudo dan lat/lon
    - clipping nilai fisik
    - lintang absolut + indikator daerah tropis (untuk EDA)
    """
    out = df.copy()
    out = _normalize_mag_cols(out)
    out = _normalize_lat_lon(out)
    out = _clip_physical_ranges(out)

    if "latitude" in out.columns:
        out["abs_lat"] = _num(out["latitude"]).abs()
        out["is_tropic"] = (out["abs_lat"] <= 23.5).astype("Int64")

    return out


def engineer_tectonic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Feature engineering khusus domain tektonik (ringan):

    - Standarisasi kolom dari katalog NOAA:
        Mag -> mag, Focal Depth (km) -> depth, Latitude/Longitude -> latitude/longitude,
        Tsu -> tsu, Year -> year.
    - Fitur inti:
        mag, depth, latitude, longitude, sig (jika ada), year
    - Turunan sederhana untuk EDA:
        depth_log1p, mag_sq, mag_over_depth1p
    - Kolom kategorikal utama:
        alert (jika ada), region/country/location (untuk OHE di EDA)
    """
    df_std = _standardize_tectonic_columns(df)
    out = engineer_common(df_std)

    # transformasi sederhana (untuk EDA)
    if "depth" in out.columns:
        out["depth_log1p"] = np.log1p(_num(out["depth"]).clip(lower=0))
    if "mag" in out.columns:
        out["mag_sq"] = _num(out["mag"]) ** 2
    if {"mag", "depth"}.issubset(out.columns):
        out["mag_over_depth1p"] = _num(out["mag"]) / (_num(out["depth"]).abs() + 1.0)

    num_cols = [
        c
        for c in [
            "mag",
            "depth",
            "latitude",
            "longitude",
            "year",
            "sig",
            "abs_lat",
            "is_tropic",
            "depth_log1p",
            "mag_sq",
            "mag_over_depth1p",
        ]
        if c in out.columns
    ]

    # kolom kategorikal utama (untuk OHE EDA)
    cat_cols = [
        c
        for c in [
            "alert",
            "Country",
            "Area",
            "Region",
            "Location",
            "Location Name",
            "country",
            "area",
            "region",
            "location",
        ]
        if c in out.columns
    ]

    # pastikan 'tsu' tetap ada jika tersedia
    cols = [c for c in out.columns if c != "tsu"] + (
        ["tsu"] if "tsu" in out.columns else []
    )
    return out[cols], num_cols, cat_cols


def engineer_volcanic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Feature engineering khusus domain vulkanik (ringan):

    - Standarisasi kolom dari katalog GVP:
        VEI -> vei, Elevation (m) -> elevation, Eq -> eq,
        Latitude/Longitude -> latitude/longitude, Tsu -> tsu, Year -> year.
    - Fitur inti:
        vei, elevation, eq, latitude, longitude, year
    - Turunan sederhana untuk EDA:
        elev_log1p, vei_sq, eq_log1p
    - Kolom kategorikal utama:
        type, status, name, agent, country (untuk OHE di EDA)
    """
    df_std = _standardize_volcanic_columns(df)
    out = engineer_common(df_std)

    if "elevation" in out.columns:
        out["elev_log1p"] = np.log1p(_num(out["elevation"]).clip(lower=0))
    if "vei" in out.columns:
        out["vei_sq"] = _num(out["vei"]) ** 2
    if "eq" in out.columns:
        out["eq_log1p"] = np.log1p(_num(out["eq"]).clip(lower=0))

    num_cols = [
        c
        for c in [
            "vei",
            "elevation",
            "eq",
            "latitude",
            "longitude",
            "year",
            "abs_lat",
            "is_tropic",
            "elev_log1p",
            "vei_sq",
            "eq_log1p",
        ]
        if c in out.columns
    ]

    cat_cols = [
        c
        for c in [
            "Type",
            "Status",
            "Name",
            "Agent",
            "Country",
            "Location",
            "type",
            "status",
            "name",
            "agent",
            "country",
            "location",
        ]
        if c in out.columns
    ]

    cols = [c for c in out.columns if c != "tsu"] + (
        ["tsu"] if "tsu" in out.columns else []
    )
    return out[cols], num_cols, cat_cols


# ====================================================
# OHE + DIAGNOSTICS (Pearson & RFE)
# ====================================================
def do_ohe(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, OneHotEncoder, List[str]]:
    """
    One-Hot Encoding untuk kolom kategorikal (untuk keperluan EDA).
    """
    enc = _one_hot_encoder()
    if not cat_cols:
        return df.copy(), enc, []
    arr = enc.fit_transform(df[cat_cols].astype("object"))
    ohe_cols = enc.get_feature_names_out(cat_cols).tolist()
    base = df.drop(columns=cat_cols).reset_index(drop=True)
    ohe_df = pd.concat(
        [base, pd.DataFrame(arr, columns=ohe_cols, index=base.index)],
        axis=1,
    )
    return ohe_df, enc, ohe_cols


def plot_feature_count(before: int, after: int, title: str, out_png: Path) -> None:
    """
    Visualisasi sederhana: jumlah fitur sebelum & sesudah OHE.
    """
    plt.figure(figsize=(4.8, 4))
    xs = np.arange(2)
    ys = [before, after]
    plt.bar(xs, ys)
    plt.xticks(xs, ["Before OHE", "After OHE"])
    for x, y in zip(xs, ys, strict=False):
        plt.text(
            x,
            y + max(1, 0.02 * max(ys)),
            str(int(y)),
            ha="center",
            fontweight="bold",
        )
    plt.title(title)
    plt.ylabel("Jumlah Fitur")
    _savefig(out_png)


def pearson_topn(
    df: pd.DataFrame,
    target: str,
    top_n: int,
    title: str,
    out_png: Path,
    out_csv: Path,
) -> None:
    """
    Top-k fitur berdasarkan |korelasi Pearson| dengan target (tsu).
    Dipakai sebagai dokumentasi EDA saja, bukan seleksi fitur keras.
    """
    num_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in {target, "id"}
    ]
    if not num_cols or target not in df.columns:
        print(f"[Pearson] {title}: no numeric columns or target missing, skip.")
        return

    corr = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    _savetab(
        corr.rename("abs_corr")
        .reset_index()
        .rename(columns={"index": "feature"}),
        out_csv,
    )
    topcorr = corr.head(top_n)
    plt.figure(figsize=(7, 4))
    y_pos = np.arange(len(topcorr))
    plt.barh(y_pos, topcorr.values)
    plt.yticks(y_pos, topcorr.index)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Abs Pearson Corr")
    _savefig(out_png)


def _plot_rfe_features(selected: List[str], title: str, out_png: Path) -> None:
    if not selected:
        return
    plt.figure(figsize=(7, 4))
    y_pos = np.arange(len(selected))
    plt.barh(y_pos, list(range(len(selected), 0, -1)))
    plt.yticks(y_pos, selected)
    plt.gca().invert_yaxis()
    plt.title(f"{title}: Top {len(selected)} Features (RFE)")
    plt.xlabel("Relative Rank (1 = best)")
    _savefig(out_png)


def rfe_select(
    df: pd.DataFrame,
    target: str,
    n_features: int,
    title: str,
    out_csv: Path,
    out_png: Optional[Path] = None,
    max_rows: int = 2000,
    max_cols: int = 120,
) -> List[str]:
    """
    RFE (Recursive Feature Elimination) versi ringan dengan RandomForest.

    Di sini kita pastikan target (tsu) aman dari NaN sebelum cast ke int:
    - to_numeric(errors='coerce') → NaN jika aneh
    - fillna(0)                   → anggap 0 (non-tsunami)
    """
    if target not in df.columns:
        print(f"[RFE] {title}: target '{target}' tidak ada, skip.")
        return []

    # Kolom fitur = semua kolom kecuali target + id
    drop_cols = [target] + (["id"] if "id" in df.columns else [])
    X = df.drop(columns=drop_cols)

    # Label aman: coerce → NaN → isi 0 → int
    y_raw = pd.to_numeric(df[target], errors="coerce")
    y = y_raw.fillna(0).astype(int)

    # Batasi jumlah kolom berdasarkan varians (kalau terlalu banyak)
    if X.shape[1] > max_cols:
        var = X.var(numeric_only=True).sort_values(ascending=False)
        keep = var.index[:max_cols]
        X = X[keep]
        print(f"[RFE] {title}: limited to {len(keep)} cols by variance.")

    # Batasi jumlah baris (sample) untuk efisiensi
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
        print(f"[RFE] {title}: sampled {max_rows} rows.")

    est = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    try:
        n_sel = min(n_features, X.shape[1])
        rfe = RFE(est, n_features_to_select=n_sel)
        rfe.fit(X, y)

        selected = X.columns[rfe.support_].tolist()
        df_out = pd.DataFrame(
            {"rank": range(1, len(selected) + 1), "feature": selected},
        )
        df_out.to_csv(out_csv, index=False)
        print(f"[RFE] {title}: top {len(selected)} features -> {selected[:5]}...")

        if out_png is not None:
            _plot_rfe_features(selected, title, out_png)

        return selected

    except Exception as exc:  # pragma: no cover
        print(f"[RFE] {title}: skipped ({exc})")
        return []


# ====================================================
# MAIN DRIVER PER DOMAIN
# ====================================================
def run_one(
    dataset: str,
    overwrite: bool,
    materialize_ohe: bool,
    rfe_topn: int = 10,
    corr_topn: int = 10,
) -> None:
    """
    Jalankan pipeline FE untuk satu domain (tectonic / volcanic).

    Sumber data:
      1) data/processed/{dataset}.csv  (jika ada)
      2) data/raw/{dataset}.csv       (fallback; cocok dengan struktur sekarang)
    """
    # cari sumber data yang tersedia
    candidates = [
        PROCESSED / f"{dataset}.csv",
        RAW / f"{dataset}.csv",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"[FE] Missing CSV for dataset '{dataset}'. "
            f"Tried: {candidates}"
        )

    dst_fe = PROCESSED / f"{dataset}_fe.csv"
    dst_fe_ohe = PROCESSED / f"{dataset}_fe_ohe.csv"

    # ---- Step 1: FE per-domain (ringan) ----
    if dst_fe.exists() and not overwrite:
        print(f"[INFO] Reusing existing FE: {dst_fe.name}")
        fe_df = _read_csv(dst_fe)
        num_cols: List[str] = [
            c for c in fe_df.select_dtypes(include="number").columns if c != "tsu"
        ]
        cat_cols: List[str] = [
            c for c in fe_df.columns if fe_df[c].dtype == "object" and c != "tsu"
        ]
    else:
        df = _read_csv(src)
        if dataset == "tectonic":
            fe_df, num_cols, cat_cols = engineer_tectonic(df)
        else:
            fe_df, num_cols, cat_cols = engineer_volcanic(df)

        fe_df.to_csv(dst_fe, index=False)
        dump(num_cols, ART / f"{dataset}_fe_num_cols.joblib")
        dump(cat_cols, ART / f"{dataset}_fe_cat_cols.joblib")
        print(
            f"[FE] {dataset}: saved {dst_fe.name} | "
            f"num={len(num_cols)} cat={len(cat_cols)}"
        )

    # ---- Step 2: Diagnostics (OHE + Pearson + RFE) ----
    fe_df = _read_csv(dst_fe)
    target = "tsu"  # label biner per-domain untuk analisis FE

    if target not in fe_df.columns:
        print(f"[FE] {dataset}: kolom 'tsu' tidak ada, skip Pearson & RFE.")
        cat_for_ohe: List[str] = []
    else:
        if dataset == "tectonic":
            default_cat = ["alert", "Country", "Area", "Region",
                           "Location", "Location Name",
                           "country", "region", "area", "location"]
        else:
            default_cat = ["Type", "Status", "Name", "Agent", "Country", "Location",
                           "type", "status", "name", "agent", "country", "location"]
        cat_for_ohe = [c for c in default_cat if c in fe_df.columns]

    before = fe_df.shape[1]
    ohe_df, enc, ohe_cols = do_ohe(fe_df, cat_for_ohe)
    after = ohe_df.shape[1]

    plot_feature_count(
        before,
        after,
        f"{dataset.title()}: Before vs After OHE",
        FIG / f"{dataset}_ohe_count.png",
    )

    # Pearson hanya jika target ada & corr_topn > 0
    if target in ohe_df.columns and corr_topn > 0:
        pearson_topn(
            ohe_df,
            target,
            corr_topn,
            f"{dataset.title()}: Top {corr_topn} |Corr| with tsu",
            FIG / f"{dataset}_pearson.png",
            TAB / f"{dataset}_pearson.csv",
        )

    # RFE hanya jika target ada & rfe_topn > 0
    if target in ohe_df.columns and rfe_topn > 0:
        rfe_select(
            ohe_df,
            target,
            rfe_topn,
            f"{dataset.title()}",
            TAB / f"{dataset}_rfe.csv",
            out_png=FIG / f"{dataset}_rfe_top{rfe_topn}.png",
        )
    else:
        if rfe_topn <= 0:
            print(f"[RFE] {dataset}: rfe_topn <= 0, skip RFE.")

    # ---- Step 3: (opsional) materialize FE-OHE ----
    if materialize_ohe:
        ohe_df.to_csv(dst_fe_ohe, index=False)
        dump(enc, ART / f"{dataset}_ohe_encoder.joblib")
        dump(ohe_cols, ART / f"{dataset}_ohe_feature_names.joblib")
        print(f"[FE] {dataset}: OHE materialized.")

# ====================================================
# BUILD GABUNGAN: events_fe.csv
# ====================================================
def build_events_fe(
    tectonic_name: str = "tectonic_fe.csv",
    volcanic_name: str = "volcanic_fe.csv",
    output_name: str = "events_fe.csv",
) -> None:
    """
    Gabungkan tectonic_fe.csv dan volcanic_fe.csv menjadi events_fe.csv
    yang dipakai oleh preprocessing.py.

    - Tambah kolom 'source_domain' = 'tectonic' / 'volcanic'
    - Kolom yang tidak ada di salah satu domain akan berisi NaN (wajar).
    - Label multi-class 0/1/2 akan dibuat di tahap preprocessing
      berdasarkan tsu + source_domain (fallback).
    """
    tect_path = PROCESSED / tectonic_name
    volc_path = PROCESSED / volcanic_name

    if not tect_path.exists() or not volc_path.exists():
        print("[FE] build_events_fe: salah satu dari tectonic_fe/volcanic_fe belum ada, skip.")
        return

    tect_df = _read_csv(tect_path).copy()
    volc_df = _read_csv(volc_path).copy()

    tect_df["source_domain"] = "tectonic"
    volc_df["source_domain"] = "volcanic"

    all_cols = sorted(set(tect_df.columns) | set(volc_df.columns))
    tect_df = tect_df.reindex(columns=all_cols)
    volc_df = volc_df.reindex(columns=all_cols)

    events_df = pd.concat([tect_df, volc_df], ignore_index=True)
    out_path = PROCESSED / output_name
    events_df.to_csv(out_path, index=False)

    print(
        f"[FE] events_fe.csv built: {out_path.name} "
        f"(rows={len(events_df)}, cols={len(events_df.columns)})"
    )


# ====================================================
# MAIN
# ====================================================
def main(
    datasets: Optional[List[str]] = None,
    overwrite: bool = False,
    materialize_ohe: bool = False,
    rfe_topn: int = 10,
    corr_topn: int = 10,
) -> None:
    """
    Wrapper main: bisa dipanggil dari kode Python atau CLI.

    datasets:
      - None  -> proses ["tectonic", "volcanic"]
      - list  -> contoh ["tectonic"], ["volcanic"], atau keduanya.
    """
    if datasets is None:
        datasets = ["tectonic", "volcanic"]

    print("[FE] Start feature engineering (ringan)...")
    for ds in datasets:
        run_one(
            ds,
            overwrite=overwrite,
            materialize_ohe=materialize_ohe,
            rfe_topn=rfe_topn,
            corr_topn=corr_topn,
        )

    if set(datasets) >= {"tectonic", "volcanic"}:
        build_events_fe()

    print(
        f"[DONE] Feature engineering selesai.\n"
        f" - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}\n - DATA: {PROCESSED}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering ringan dengan OHE, Pearson, dan RFE untuk EDA."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["tectonic", "volcanic"],
        default=["tectonic", "volcanic"],
        help="dataset yang akan diproses (default: tectonic volcanic)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--materialize-ohe",
        action="store_true",
        help="simpan juga *_fe_ohe.csv dan encoder OHE ke artifacts/",
    )
    parser.add_argument("--rfe-topn", type=int, default=10)
    parser.add_argument("--corr-topn", type=int, default=10)
    args = parser.parse_args()

    main(
        datasets=args.datasets,
        overwrite=args.overwrite,
        materialize_ohe=args.materialize_ohe,
        rfe_topn=args.rfe_topn,
        corr_topn=args.corr_topn,
    )