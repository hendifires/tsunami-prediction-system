# src/tsunami_prediction/feature_engineering.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

# ========== MATPLOTLIB NON-GUI ==========
import matplotlib
matplotlib.use("Agg")  # hindari Tkinter / GUI backend di mode CLI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

sns.set_theme(style="whitegrid")

# ====================================================
# PATH SETUP
# ====================================================
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"
ART = ROOT / "artifacts"

for p in [PROCESSED, TAB, FIG, ART]:
    p.mkdir(parents=True, exist_ok=True)


# ====================================================
# UTILITY FUNCTIONS
# ====================================================
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


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """Encode kolom siklikal (bulan/hari)."""
    if col in df.columns:
        x = _safe_num(df[col]).clip(lower=0)
        df[f"{col}_sin"] = np.sin(2 * np.pi * x / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * x / period)
    return df


def _one_hot_encoder():
    # kompatibel untuk sklearn lama/baru
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ====================================================
# DOMAIN FEATURES
# ====================================================
_SUBDUCTION = {
    "Indonesia", "Japan", "Philippines", "Taiwan", "Papua New Guinea", "Solomon Islands",
    "Vanuatu", "Tonga", "New Zealand", "Fiji",
    "United States", "Canada", "Mexico", "Guatemala", "El Salvador", "Honduras", "Nicaragua",
    "Costa Rica", "Panama", "Colombia", "Ecuador", "Peru", "Chile",
    "Russia", "Greece", "Argentina", "Bolivia",
}
_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "myanmar (burma)": "Myanmar",
    "burma": "Myanmar",
    "russian federation": "Russia",
}


def _normalize_country(x: str) -> str:
    """Normalisasi nama negara ke bentuk konsisten."""
    if pd.isna(x):
        return x
    # x sudah bertipe str, jadi tidak perlu str(x)
    s = x.strip()
    low = s.lower()
    return _ALIASES.get(low, s.title())


def add_subduction_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan flag negara subduksi."""
    if "country" not in df.columns:
        return df
    out = df.copy()
    out["country_norm"] = out["country"].astype("object").map(_normalize_country)
    out["is_subduction_zone"] = out["country_norm"].isin(_SUBDUCTION).astype("Int64")
    return out


def add_distance_to_coast(df: pd.DataFrame, coast_path: str | None) -> pd.DataFrame:
    """Tambahkan 'distance_to_coast_km' bila shapefile tersedia."""
    if coast_path is None or not Path(coast_path).exists():
        print("[FE] Coastline file not found or not provided — skip distance_to_coast_km.")
        return df
    if not {"latitude", "longitude"}.issubset(df.columns):
        return df
    try:
        import geopandas as gpd  # type: ignore
        from shapely.geometry import Point  # type: ignore
    except Exception:
        print("[FE] geopandas/shapely not installed — skip distance_to_coast_km.")
        return df

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[
            Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
            for lon, lat in zip(df["longitude"], df["latitude"])
        ],
        crs="EPSG:4326",
    )
    coast = gpd.read_file(coast_path)
    if coast.crs is None:
        coast.set_crs("EPSG:4326", inplace=True)
    coast = coast.to_crs("EPSG:3857")
    gdf_merc = gdf.to_crs("EPSG:3857")

    coast_union = coast.unary_union
    gdf_merc["distance_to_coast_km"] = gdf_merc.geometry.apply(
        lambda geom: geom.distance(coast_union) / 1000.0 if geom is not None else np.nan
    )
    return pd.DataFrame(gdf_merc.drop(columns="geometry"))


def add_time_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan days_since_prev antar event (berdasarkan tanggal)."""
    if "year" not in df.columns:
        return df
    out = df.copy()
    out = out.rename(columns={"mo": "month", "dy": "day"})
    for c in ["month", "day"]:
        if c in out.columns:
            out[c] = _safe_num(out[c])
    out["event_date"] = pd.to_datetime(out[["year", "month", "day"]], errors="coerce")
    out = out.sort_values(["event_date"], kind="mergesort")
    out["days_since_prev"] = out["event_date"].diff().dt.days.fillna(0).clip(lower=0)
    return out.drop(columns=["event_date"])


# ====================================================
# FEATURE ENGINEERING CORE
# ====================================================
def engineer_common(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={"mo": "month", "dy": "day"})
    out = _cyclical(out, "month", 12)
    out = _cyclical(out, "day", 31)

    if "latitude" in out.columns:
        out["abs_lat"] = _safe_num(out["latitude"]).abs()
        out["is_tropic"] = (out["abs_lat"] <= 23.5).astype("Int64")
    if {"latitude", "longitude"}.issubset(out.columns):
        out["lat_lon_prod"] = _safe_num(out["latitude"]) * _safe_num(out["longitude"])
    return out


def engineer_tectonic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = engineer_common(df)
    out = add_subduction_flag(out)
    out = add_time_gap(out)

    if "depth" in out.columns:
        out["depth_log1p"] = np.log1p(_safe_num(out["depth"]).clip(lower=0))
    if "mag" in out.columns:
        out["mag_sq"] = _safe_num(out["mag"]) ** 2
    if {"mag", "depth"}.issubset(out.columns):
        out["mag_over_depth1p"] = _safe_num(out["mag"]) / (_safe_num(out["depth"]).abs() + 1.0)

    num_cols = [
        c
        for c in [
            "mag",
            "depth",
            "latitude",
            "longitude",
            "year",
            "month",
            "day",
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
            "abs_lat",
            "is_tropic",
            "lat_lon_prod",
            "depth_log1p",
            "mag_sq",
            "mag_over_depth1p",
            "days_since_prev",
            "is_subduction_zone",
        ]
        if c in out.columns
    ]
    cat_cols = [c for c in ["country_norm", "country", "region", "area", "location"] if c in out.columns]

    cols = [c for c in out.columns if c != "tsu"] + (["tsu"] if "tsu" in out.columns else [])
    return out[cols], num_cols, cat_cols


def engineer_volcanic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    out = engineer_common(df)
    out = add_subduction_flag(out)
    out = add_time_gap(out)

    if "elevation" in out.columns:
        out["elev_log1p"] = np.log1p(_safe_num(out["elevation"]).clip(lower=0))
    if "vei" in out.columns:
        out["vei_sq"] = _safe_num(out["vei"]) ** 2
    if "eq" in out.columns:
        out["eq_log1p"] = np.log1p(_safe_num(out["eq"]).clip(lower=0))
    if {"vei", "elevation"}.issubset(out.columns):
        out["vei_x_elev"] = _safe_num(out["vei"]) * _safe_num(out["elevation"])

    num_cols = [
        c
        for c in [
            "vei",
            "elevation",
            "eq",
            "latitude",
            "longitude",
            "year",
            "month",
            "day",
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
            "abs_lat",
            "is_tropic",
            "lat_lon_prod",
            "elev_log1p",
            "vei_sq",
            "eq_log1p",
            "vei_x_elev",
            "days_since_prev",
            "is_subduction_zone",
        ]
        if c in out.columns
    ]
    cat_cols = [c for c in ["country_norm", "country", "type", "status", "location", "name", "agent"] if c in out.columns]

    cols = [c for c in out.columns if c != "tsu"] + (["tsu"] if "tsu" in out.columns else [])
    return out[cols], num_cols, cat_cols


# ====================================================
# OHE + DIAGNOSTICS
# ====================================================
def do_ohe(df: pd.DataFrame, cat_cols: List[str]) -> Tuple[pd.DataFrame, OneHotEncoder, List[str]]:
    enc = _one_hot_encoder()
    # ⚠️ peringatan “Simplify sequence length comparison” diperbaiki di sini
    if not cat_cols:
        return df.copy(), enc, []
    arr = enc.fit_transform(df[cat_cols].astype("object"))
    ohe_cols = enc.get_feature_names_out(cat_cols).tolist()
    base = df.drop(columns=cat_cols).reset_index(drop=True)
    ohe_df = pd.concat([base, pd.DataFrame(arr, columns=ohe_cols, index=base.index)], axis=1)
    return ohe_df, enc, ohe_cols


def plot_feature_count(before: int, after: int, title: str, out_png: Path):
    """Pakai matplotlib murni supaya gak kena FutureWarning seaborn."""
    plt.figure(figsize=(4.8, 4))
    xs = np.arange(2)
    ys = [before, after]
    plt.bar(xs, ys)
    plt.xticks(xs, ["Before OHE", "After OHE"])
    for x, y in zip(xs, ys):
        plt.text(x, y + max(1, 0.02 * max(ys)), str(int(y)), ha="center", fontweight="bold")
    plt.title(title)
    plt.ylabel("Jumlah Fitur")
    _savefig(out_png)


def pearson_topn(df: pd.DataFrame, target: str, top_n: int, title: str, out_png: Path, out_csv: Path):
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    corr = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    _savetab(corr.rename("abs_corr").reset_index().rename(columns={"index": "feature"}), out_csv)

    # pakai matplotlib supaya gak muncul warning seaborn
    topcorr = corr.head(top_n)
    plt.figure(figsize=(7, 4))
    y_pos = np.arange(len(topcorr))
    plt.barh(y_pos, topcorr.values)
    plt.yticks(y_pos, topcorr.index)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Abs Pearson Corr")
    _savefig(out_png)

    return corr


def rfe_select(
    df: pd.DataFrame,
    target: str,
    n_features: int,
    title: str,
    out_csv: Path,
    max_rows: int = 2000,
    max_cols: int = 120,
):
    """RFE versi ringan agar tidak macet."""
    if target not in df.columns:
        print(f"[RFE] {title}: target '{target}' tidak ada, skip.")
        return []

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Batasi jumlah kolom (biar gak kebanyakan setelah OHE)
    if X.shape[1] > max_cols:
        var = X.var(numeric_only=True).sort_values(ascending=False)
        keep = var.index[:max_cols]
        X = X[keep]
        print(f"[RFE] {title}: limited to {len(keep)} cols by variance.")

    # Sampling baris
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]
        print(f"[RFE] {title}: sampled {max_rows} rows.")

    est = RandomForestClassifier(
        n_estimators=120, random_state=42, class_weight="balanced", n_jobs=-1
    )
    try:
        rfe = RFE(est, n_features_to_select=min(n_features, X.shape[1]))
        rfe.fit(X, y)
        selected = X.columns[rfe.support_].tolist()
        pd.DataFrame(
            {"rank": range(1, len(selected) + 1), "feature": selected}
        ).to_csv(out_csv, index=False)
        print(f"[RFE] {title}: top {len(selected)} features -> {selected[:5]}...")
        return selected
    except Exception as e:
        print(f"[RFE] {title}: skipped ({e})")
        return []


# ====================================================
# MAIN DRIVER
# ====================================================
def run_one(
    dataset: str,
    overwrite: bool,
    materialize_ohe: bool,
    coast_path: str | None,
    rfe_topn: int = 10,
    corr_topn: int = 10,
) -> None:
    src = PROCESSED / f"{dataset}.csv"
    if not src.exists():
        raise FileNotFoundError(f"[FE] Missing clean CSV: {src}")

    dst_fe = PROCESSED / f"{dataset}_fe.csv"
    dst_fe_ohe = PROCESSED / f"{dataset}_fe_ohe.csv"

    if dst_fe.exists() and not overwrite:
        print(f"[INFO] Reusing existing FE: {dst_fe.name}")
    else:
        df = _read_csv(src)
        if dataset == "tectonic":
            fe_df, num_cols, cat_cols = engineer_tectonic(df)
        else:
            fe_df, num_cols, cat_cols = engineer_volcanic(df)
        fe_df = add_distance_to_coast(fe_df, coast_path)
        fe_df.to_csv(dst_fe, index=False)
        dump(num_cols, ART / f"{dataset}_fe_num_cols.joblib")
        dump(cat_cols, ART / f"{dataset}_fe_cat_cols.joblib")
        print(f"[FE] {dataset}: saved {dst_fe.name} | num={len(num_cols)} cat={len(cat_cols)}")

    # --- diagnostics ---
    fe_df = _read_csv(dst_fe)
    target = "tsu"
    if dataset == "tectonic":
        cat_for_ohe = ["country_norm", "country", "region", "area", "location"]
    else:
        cat_for_ohe = ["country_norm", "country", "type", "status", "location", "name", "agent"]
    cat_for_ohe = [c for c in cat_for_ohe if c in fe_df.columns]

    before = fe_df.shape[1]
    ohe_df, enc, ohe_cols = do_ohe(fe_df, cat_for_ohe)
    after = ohe_df.shape[1]

    plot_feature_count(
        before,
        after,
        f"{dataset.title()}: Before vs After OHE",
        FIG / f"{dataset}_ohe_count.png",
    )

    pearson_topn(
        ohe_df,
        target,
        corr_topn,
        f"{dataset.title()}: Top {corr_topn} Corr",
        FIG / f"{dataset}_pearson.png",
        TAB / f"{dataset}_pearson.csv",
    )

    rfe_select(
        ohe_df,
        target,
        rfe_topn,
        f"{dataset.title()}",
        TAB / f"{dataset}_rfe.csv",
    )

    if materialize_ohe:
        ohe_df.to_csv(dst_fe_ohe, index=False)
        dump(enc, ART / f"{dataset}_ohe_encoder.joblib")
        dump(ohe_cols, ART / f"{dataset}_ohe_feature_names.joblib")
        print(f"[FE] {dataset}: OHE materialized.")


def main(overwrite: bool = False, materialize_ohe: bool = False, coast_path: str | None = None):
    print("[FE] Start feature engineering...")
    run_one("tectonic", overwrite, materialize_ohe, coast_path)
    run_one("volcanic", overwrite, materialize_ohe, coast_path)
    print(f"[DONE] Feature engineering selesai.\n - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Feature engineering with OHE diagnostics.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--materialize-ohe", action="store_true")
    ap.add_argument("--coast", type=str, default=None)
    ap.add_argument("--rfe-topn", type=int, default=10)
    ap.add_argument("--corr-topn", type=int, default=10)
    args = ap.parse_args()
    main(
        overwrite=args.overwrite,
        materialize_ohe=args.materialize_ohe,
        coast_path=args.coast,
    )