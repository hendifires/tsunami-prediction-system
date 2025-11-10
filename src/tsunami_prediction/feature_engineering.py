from __future__ import annotations
# cSpell:ignore writeable subduction Vanuatu Tonga Nicaragua Rica seaborn geopandas shapely Opsional opsional untuk

"""
Feature Engineering untuk domain tektonik & vulkanik.

Fokus:
- Menambahkan fitur domain-aware yang relevan untuk prediksi tsunami:
  * Tektonik:
      - magnitudo, depth, MMI (mmi_int) + transformasi (log / kuadrat / rasio)
      - lokasi (lat/lon, abs_lat, tropis atau tidak)
      - zona subduksi (is_subduction_zone)
      - jarak ke garis pantai (distance_to_coast_km + turunannya, bila tersedia)
      - jeda waktu antar kejadian (days_since_prev)
      - encoding siklik kalender (month_sin/cos, day_sin/cos; opsional)
  * Vulkanik:
      - VEI, elevasi, jumlah gempa (eq) + transformasinya
      - lokasi (lat/lon, abs_lat, tropis atau tidak)
      - zona subduksi
      - jarak ke pantai (jika shapefile ada)
      - jeda waktu antar kejadian + fitur kalender
- Menyediakan tabel & visual diagnostik untuk FE:
  * Jumlah fitur sebelum vs sesudah One-Hot Encoding (OHE).
  * Top-k fitur berdasar |Pearson corr| terhadap tsu.
  * Top-k fitur hasil RFE dengan RandomForest (tabel + plot bar).

Output utama per domain:
- data/processed/<dataset>_fe.csv           : dataset dengan fitur rekayasa (belum OHE)
- data/processed/<dataset>_fe_ohe.csv       : (opsional) versi OHE
- artifacts/<dataset>_fe_num_cols.joblib   : daftar fitur numerik FE
- artifacts/<dataset>_fe_cat_cols.joblib   : daftar fitur kategorikal FE
- artifacts/<dataset>_ohe_encoder.joblib   : encoder OHE (opsional)
- reports/tables/*.csv & reports/figures/*.png untuk dokumentasi tesis
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Any
from contextlib import suppress  # hanya untuk seaborn opsional

# ========== MATPLOTLIB NON-GUI ==========
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# (Opsional) tema; semua plot utama tetap pakai matplotlib murni
with suppress(Exception):
    import seaborn as sns  # type: ignore

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
for p in (PROCESSED, TAB, FIG, ART):
    p.mkdir(parents=True, exist_ok=True)

# default shapefile coastline (opsional)
DEFAULT_COAST = DATA / "coastline" / "ne_10m_coastline.shp"

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


def _num(s: pd.Series) -> pd.Series:
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
    """Clamp nilai fisik ke rentang wajar agar stabil (mengurangi outlier ekstrem)."""
    out = df.copy()
    if "mag" in out.columns:
        out["mag"] = _num(out["mag"]).clip(lower=0, upper=10)
    if "depth" in out.columns:
        out["depth"] = _num(out["depth"]).clip(lower=0)
    if "mmi_int" in out.columns:
        # Modified Mercalli Intensity biasanya 0–12
        out["mmi_int"] = _num(out["mmi_int"]).clip(lower=0, upper=12)
    if "vei" in out.columns:
        out["vei"] = _num(out["vei"]).clip(lower=0, upper=8)
    if "elevation" in out.columns:
        out["elevation"] = _num(out["elevation"]).clip(lower=-500, upper=9000)
    if "eq" in out.columns:
        out["eq"] = _num(out["eq"]).clip(lower=0)
    return out


# ====================================================
# DOMAIN RULES
# ====================================================
_SUBDUCTION = {
    "Indonesia",
    "Japan",
    "Philippines",
    "Taiwan",
    "Papua New Guinea",
    "Solomon Islands",
    "Vanuatu",
    "Tonga",
    "New Zealand",
    "Fiji",
    "United States",
    "Canada",
    "Mexico",
    "Guatemala",
    "El Salvador",
    "Honduras",
    "Nicaragua",
    "Costa Rica",
    "Panama",
    "Colombia",
    "Ecuador",
    "Peru",
    "Chile",
    "Russia",
    "Greece",
    "Argentina",
    "Bolivia",
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


def _normalize_country(x: Any) -> Any:
    if pd.isna(x):
        return x
    s = str(x).strip()
    return _ALIASES.get(s.lower(), s.title())


def add_subduction_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan indikator apakah negara berada di zona subduksi utama.
    Feature ini sangat penting secara fisik, karena banyak tsunami besar
    terjadi di zona subduksi.
    """
    if "country" not in df.columns:
        return df
    out = df.copy()
    out["country_norm"] = out["country"].astype("object").map(_normalize_country)
    out["is_subduction_zone"] = out["country_norm"].isin(_SUBDUCTION).astype("Int64")
    return out


# ---- helper terpisah untuk distance-to-coast ----
def _distance_to_coast_with_gpd(df: pd.DataFrame, coast_path: str) -> pd.DataFrame:
    """Hitung jarak ke garis pantai dengan GeoPandas/Shapely (EPSG:4326->3857)."""
    import geopandas as gpd  # type: ignore
    from shapely.geometry import Point  # type: ignore

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[
            Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
            for lon, lat in zip(df["longitude"], df["latitude"], strict=False)
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


def add_distance_to_coast(df: pd.DataFrame, coast_path: Optional[str]) -> pd.DataFrame:
    """
    Tambahkan 'distance_to_coast_km' bila shapefile tersedia.

    Dirancang aman:
    - Jika file garis pantai tidak ada → hanya log dan return df apa adanya.
    - Jika geopandas/shapely tidak tersedia → juga di-skip dengan pesan info.
    """
    if coast_path is None or not Path(coast_path).exists():
        print("[FE] Coastline file not found or not provided — skip distance_to_coast_km.")
        return df
    if not {"latitude", "longitude"}.issubset(df.columns):
        return df
    try:
        return _distance_to_coast_with_gpd(df, coast_path)
    except Exception:
        print("[FE] geopandas/shapely not installed — skip distance_to_coast_km.")
        return df


def add_distance_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fitur turunan dari distance_to_coast_km:
    - dist_coast_log1p    : skala log (mengurangi efek outlier).
    - near_coast_50km     : indikator dekat pantai <= 50 km.
    - near_coast_100km    : indikator dekat pantai <= 100 km.
    """
    out = df.copy()
    if "distance_to_coast_km" in out.columns:
        d = _num(out["distance_to_coast_km"])
        out["dist_coast_log1p"] = np.log1p(d.clip(lower=0))
        out["near_coast_50km"] = (d <= 50).astype("Int64")
        out["near_coast_100km"] = (d <= 100).astype("Int64")
    return out


def add_time_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan days_since_prev antar event (berdasarkan tanggal).

    Interpretasi:
    - Menggambarkan 'kepadatan' kejadian di waktu berdekatan.
    - Bisa relevan untuk memodelkan clustering temporal aktivitas seismik/vulkanik.
    """
    if "year" not in df.columns:
        return df
    out = df.copy()
    out = out.rename(columns={"mo": "month", "dy": "day"})
    for c in ("month", "day"):
        if c in out.columns:
            out[c] = _num(out[c])
    out["event_date"] = pd.to_datetime(out[["year", "month", "day"]], errors="coerce")
    out = out.sort_values(["event_date"], kind="mergesort")
    out["days_since_prev"] = out["event_date"].diff().dt.days.fillna(0).clip(lower=0)
    return out.drop(columns=["event_date"])


def _cyclical(df: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
    """
    Encode kolom waktu (mis. bulan/hari) sebagai fitur siklik:
    - col_sin, col_cos
    """
    if col in df.columns:
        x = _num(df[col]).clip(lower=0)
        df[f"{col}_sin"] = np.sin(2 * np.pi * x / period)
        df[f"{col}_cos"] = np.cos(2 * np.pi * x / period)
    return df


# ====================================================
# FEATURE ENGINEERING CORE
# ====================================================
def engineer_common(df: pd.DataFrame, use_temporal: bool = False) -> pd.DataFrame:
    """
    Fitur dasar yang berlaku untuk kedua domain (tektonik & vulkanik):
    - Clipping nilai fisik.
    - Normalisasi nama bulan/hari.
    - Fitur lintang absolut + indikator daerah tropis.
    - Produk lat*lon sebagai sinyal lokasi kasar.
    - Encoding siklik bulan/hari (opsional).
    """
    out = _clip_physical_ranges(df.copy())
    out = out.rename(columns={"mo": "month", "dy": "day"})
    if use_temporal:
        out = _cyclical(out, "month", 12)
        out = _cyclical(out, "day", 31)
    if "latitude" in out.columns:
        out["abs_lat"] = _num(out["latitude"]).abs()
        out["is_tropic"] = (out["abs_lat"] <= 23.5).astype("Int64")
    if {"latitude", "longitude"}.issubset(out.columns):
        out["lat_lon_prod"] = _num(out["latitude"]) * _num(out["longitude"])
    return out


def engineer_tectonic(
    df: pd.DataFrame,
    use_temporal: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Feature engineering khusus domain tektonik.
    """
    out = engineer_common(df, use_temporal=use_temporal)
    out = add_subduction_flag(out)
    out = add_time_gap(out)

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
            "mmi_int",
            "latitude",
            "longitude",
            "year",
            "month",
            "day",
            "abs_lat",
            "is_tropic",
            "lat_lon_prod",
            "depth_log1p",
            "mag_sq",
            "mag_over_depth1p",
            "days_since_prev",
            "distance_to_coast_km",
            "dist_coast_log1p",
            "near_coast_50km",
            "near_coast_100km",
            "is_subduction_zone",
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
        ]
        if c in out.columns
    ]

    cat_cols = [
        c
        for c in ["country_norm", "country", "region", "area", "location"]
        if c in out.columns
    ]

    cols = [c for c in out.columns if c != "tsu"] + (
        ["tsu"] if "tsu" in out.columns else []
    )
    return out[cols], num_cols, cat_cols


def engineer_volcanic(
    df: pd.DataFrame,
    use_temporal: bool = False,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Feature engineering khusus domain vulkanik.
    """
    out = engineer_common(df, use_temporal=use_temporal)
    out = add_subduction_flag(out)
    out = add_time_gap(out)

    if "elevation" in out.columns:
        out["elev_log1p"] = np.log1p(_num(out["elevation"]).clip(lower=0))
    if "vei" in out.columns:
        out["vei_sq"] = _num(out["vei"]) ** 2
    if "eq" in out.columns:
        out["eq_log1p"] = np.log1p(_num(out["eq"]).clip(lower=0))
    if {"vei", "elevation"}.issubset(out.columns):
        out["vei_x_elev"] = _num(out["vei"]) * _num(out["elevation"])
    if {"vei", "eq"}.issubset(out.columns):
        out["vei_over_eq1p"] = _num(out["vei"]) / (_num(out["eq"]) + 1.0)

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
            "abs_lat",
            "is_tropic",
            "lat_lon_prod",
            "elev_log1p",
            "vei_sq",
            "eq_log1p",
            "vei_x_elev",
            "vei_over_eq1p",
            "days_since_prev",
            "distance_to_coast_km",
            "dist_coast_log1p",
            "near_coast_50km",
            "near_coast_100km",
            "is_subduction_zone",
            "month_sin",
            "month_cos",
            "day_sin",
            "day_cos",
        ]
        if c in out.columns
    ]

    cat_cols = [
        c
        for c in ["country_norm", "country", "type", "status", "location", "name", "agent"]
        if c in out.columns
    ]

    cols = [c for c in out.columns if c != "tsu"] + (
        ["tsu"] if "tsu" in out.columns else []
    )
    return out[cols], num_cols, cat_cols


# ====================================================
# OHE + DIAGNOSTICS (opsional)
# ====================================================
def do_ohe(
    df: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, OneHotEncoder, List[str]]:
    """
    One-Hot Encoding untuk kolom kategorikal.
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
):
    """
    Top-k fitur berdasarkan |korelasi Pearson| dengan target tsu.
    """
    num_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in {target, "id"}
    ]
    if not num_cols or target not in df.columns:
        print(f"[Pearson] {title}: no numeric columns or target missing, skip.")
        return pd.Series(dtype=float)

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
    return corr


def _plot_rfe_features(selected: List[str], title: str, out_png: Path) -> None:
    """
    Helper plotting untuk hasil RFE.
    """
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
    """
    if target not in df.columns:
        print(f"[RFE] {title}: target '{target}' tidak ada, skip.")
        return []

    drop_cols = [target] + (["id"] if "id" in df.columns else [])
    X = df.drop(columns=drop_cols)
    y = df[target].astype(int)

    if X.shape[1] > max_cols:
        var = X.var(numeric_only=True).sort_values(ascending=False)
        keep = var.index[:max_cols]
        X = X[keep]
        print(f"[RFE] {title}: limited to {len(keep)} cols by variance.")
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
            {"rank": range(1, len(selected) + 1), "feature": selected}
        )
        df_out.to_csv(out_csv, index=False)
        print(f"[RFE] {title}: top {len(selected)} features -> {selected[:5]}...")

        if out_png is not None:
            _plot_rfe_features(selected, title, out_png)

        return selected
    except Exception as exc:  # pragma: no cover (defensif)
        print(f"[RFE] {title}: skipped ({exc})")
        return []


# ====================================================
# MAIN DRIVER
# ====================================================
def run_one(
    dataset: str,
    overwrite: bool,
    materialize_ohe: bool,
    coast_path: Optional[str],
    rfe_topn: int = 10,
    corr_topn: int = 10,
    use_temporal: bool = False,
) -> None:
    """
    Jalankan pipeline FE untuk satu domain (tectonic / volcanic).
    """
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
            fe_df, num_cols, cat_cols = engineer_tectonic(
                df,
                use_temporal=use_temporal,
            )
        else:
            fe_df, num_cols, cat_cols = engineer_volcanic(
                df,
                use_temporal=use_temporal,
            )

        # ---- Coastline features dihitung di sini, lalu num_cols di-update ----
        fe_df = add_distance_to_coast(fe_df, coast_path)
        fe_df = add_distance_derivatives(fe_df)

        coast_features = [
            "distance_to_coast_km",
            "dist_coast_log1p",
            "near_coast_50km",
            "near_coast_100km",
        ]
        num_cols = sorted(
            {
                *num_cols,
                *(c for c in coast_features if c in fe_df.columns),
            }
        )

        fe_df.to_csv(dst_fe, index=False)
        dump(num_cols, ART / f"{dataset}_fe_num_cols.joblib")
        dump(cat_cols, ART / f"{dataset}_fe_cat_cols.joblib")
        print(
            f"[FE] {dataset}: saved {dst_fe.name} | "
            f"num={len(num_cols)} cat={len(cat_cols)}"
        )

    # ---------- Diagnostics (OHE + Pearson + RFE) ----------
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
        f"{dataset.title()}: Top {corr_topn} |Corr| with tsu",
        FIG / f"{dataset}_pearson.png",
        TAB / f"{dataset}_pearson.csv",
    )

    rfe_select(
        ohe_df,
        target,
        rfe_topn,
        f"{dataset.title()}",
        TAB / f"{dataset}_rfe.csv",
        out_png=FIG / f"{dataset}_rfe_top{rfe_topn}.png",
    )

    if materialize_ohe:
        ohe_df.to_csv(dst_fe_ohe, index=False)
        dump(enc, ART / f"{dataset}_ohe_encoder.joblib")
        dump(ohe_cols, ART / f"{dataset}_ohe_feature_names.joblib")
        print(f"[FE] {dataset}: OHE materialized.")


def main(
    datasets: Optional[List[str]] = None,
    overwrite: bool = False,
    materialize_ohe: bool = False,
    coast_path: Optional[str] = None,
    use_temporal: bool = False,
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

    print("[FE] Start feature engineering...")
    for ds in datasets:
        run_one(
            ds,
            overwrite,
            materialize_ohe,
            coast_path,
            rfe_topn=rfe_topn,
            corr_topn=corr_topn,
            use_temporal=use_temporal,
        )
    print(
        f"[DONE] Feature engineering selesai.\n"
        f" - FIG: {FIG}\n - TAB: {TAB}\n - ART: {ART}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering with OHE diagnostics."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["tectonic", "volcanic"],
        default=["tectonic", "volcanic"],
        help="dataset yang akan diproses (default: tectonic volcanic)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--materialize-ohe", action="store_true")
    parser.add_argument(
        "--coast",
        type=str,
        default=str(DEFAULT_COAST) if DEFAULT_COAST.exists() else None,
        help="path ke shapefile coastline (.shp); default ke data/coastline/ne_10m_coastline.shp bila ada",
    )
    parser.add_argument("--rfe-topn", type=int, default=10)
    parser.add_argument("--corr-topn", type=int, default=10)
    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="aktifkan sinus/cos bulan & hari (default off)",
    )
    args = parser.parse_args()
    main(
        datasets=args.datasets,
        overwrite=args.overwrite,
        materialize_ohe=args.materialize_ohe,
        coast_path=args.coast,
        use_temporal=args.use_temporal,
        rfe_topn=args.rfe_topn,
        corr_topn=args.corr_topn,
    )