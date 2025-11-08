from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ================== PATHS ==================
ROOT = Path(__file__).resolve().parents[2]  # .../tsunami-prediction
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"

ART = ROOT / "artifacts"  # encoder, scaler, daftar fitur

for p in (PROCESSED, TAB, ART):
    p.mkdir(parents=True, exist_ok=True)


# ================== I/O HELPERS ==================
def _read_csv_guess(path: Path) -> pd.DataFrame:
    """Baca CSV dengan autodetect delimiter. (engine='python' tidak boleh pakai low_memory)."""
    return pd.read_csv(path, sep=None, engine="python")


def _save_info_tables(df: pd.DataFrame, name: str) -> None:
    """Simpan dtypes & missing summary ke reports/tables."""
    dtypes = pd.DataFrame(
        {"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}
    )
    dtypes.to_csv(TAB / f"{name}_dtypes.csv", index=False)

    miss = df.isna().sum().rename("missing")
    missp = (df.isna().mean() * 100).round(2).rename("missing_percent")
    (
        pd.concat([miss, missp], axis=1)
        .reset_index()
        .rename(columns={"index": "column"})
        .to_csv(TAB / f"{name}_missing.csv", index=False)
    )


def _save_shape_summary(rows_cols: List[Tuple[str, int, int]]) -> None:
    pd.DataFrame(rows_cols, columns=["dataset", "rows", "cols"]).to_csv(
        TAB / "processed_shapes.csv", index=False
    )


def _save_schema_and_samples(df: pd.DataFrame, dataset: str) -> None:
    """
    Simpan:
    - Schema clean (Kolom, Tipe data, Contoh) -> untuk Tabel 6 & 7 di tesis.
    - Sample baris nyata -> contoh data real di lampiran / ilustrasi.
    """
    rows: List[Dict[str, object]] = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        non_na = series.dropna()
        example = "" if non_na.empty else non_na.iloc[0]
        rows.append({"Kolom": col, "Tipe data": dtype, "Contoh": example})

    schema_df = pd.DataFrame(rows)
    schema_df.to_csv(TAB / f"{dataset}_schema_clean.csv", index=False)

    # contoh beberapa baris data real
    df.head(20).to_csv(TAB / f"{dataset}_sample_clean.csv", index=False)


# ================== CLEANING & STANDARDIZATION ==================
def _standardize_columns(
    df: pd.DataFrame, col_map: Dict[str, str] | None = None
) -> pd.DataFrame:
    """Normalkan nama kolom dan terapkan peta rename yang diperlukan."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=True)
        .str.lower()
        .str.replace(" ", "_")
    )
    if col_map:
        df = df.rename(columns=col_map)
    # singkirkan kolom duplikat (bisa muncul setelah normalisasi nama)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace & normalisasi string NaN."""
    df = df.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace(
            {"nan": np.nan, "none": np.nan, "None": np.nan, "NONE": np.nan, "": np.nan}
        )
    return df


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Paksa kolom jadi numerik (errors->NaN)."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _normalize_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validasi kalender:
    - month di luar 1..12 -> NaN
    - day di luar 1..31 -> NaN
    - year boleh negatif (contoh: -4360) tapi dibatasi kisaran wajar [-5000, 2100]
    - jam:   0..23; menit & detik: 0..59
    """
    df = df.copy()
    if "year" in df.columns:
        df.loc[~df["year"].between(-5000, 2100), "year"] = np.nan
    if "month" in df.columns:
        df.loc[~df["month"].between(1, 12), "month"] = np.nan
    if "day" in df.columns:
        df.loc[~df["day"].between(1, 31), "day"] = np.nan
    if "hr" in df.columns:
        df.loc[~df["hr"].between(0, 23), "hr"] = np.nan
    if "mn" in df.columns:
        df.loc[~df["mn"].between(0, 59), "mn"] = np.nan
    if "sec" in df.columns:
        df.loc[~df["sec"].between(0, 59), "sec"] = np.nan
    return df


def _map_tsu_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan kolom target 'tsu' biner 0/1.
    - Jika angka: >0 -> 1, else 0
    - Jika string: yes/true/y/t/1 -> 1
    - Jika kolom target tidak ada, buat tsu=0 (agar pipeline tetap jalan)
    """
    df = df.copy()
    candidates = [
        "tsu",
        "tsunami",
        "tsu_flag",
        "tsunami_flag",
        "is_tsunami",
        "tsunami_event",
        "tsunami_(0/1)",
        "tsunami_(y/n)",
    ]
    target_col = next((c for c in candidates if c in df.columns), None)

    if target_col is None:
        df["tsu"] = 0
        return df

    s = df[target_col]
    if s.dtype == "O":
        s2 = s.astype(str).str.lower().str.strip()
        df["tsu"] = s2.isin(
            {"1", "y", "yes", "true", "tsunami", "t"}
        ).astype(int)
    else:
        df["tsu"] = pd.to_numeric(s, errors="coerce").fillna(0)
        df["tsu"] = (df["tsu"] > 0).astype(int)

    if target_col != "tsu":
        df = df.drop(columns=[target_col])
    return df


def _drop_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Filter nilai koordinat/fisik ekstrem yang tidak masuk akal."""
    df = df.copy()
    if "latitude" in df.columns:
        df = df[(df["latitude"] >= -90) & (df["latitude"] <= 90)]
    if "longitude" in df.columns:
        df = df[(df["longitude"] >= -180) & (df["longitude"] <= 180)]
    if "depth" in df.columns:
        df.loc[df["depth"] < 0, "depth"] = np.nan
        df.loc[df["depth"] > 700, "depth"] = np.nan
    if "mag" in df.columns:
        df.loc[(df["mag"] < 0) | (df["mag"] > 10), "mag"] = np.nan
    if "vei" in df.columns:
        df.loc[(df["vei"] < 0) | (df["vei"] > 8), "vei"] = np.nan
    if "elevation" in df.columns:
        df.loc[df["elevation"] < -500, "elevation"] = np.nan
    return df


# ---------- dedup utility ----------
def _drop_and_log(
    df: pd.DataFrame,
    subset,
    dataset: str,
    label: str,
) -> pd.DataFrame:
    """Drop duplicates on subset and print delta."""
    before = len(df)
    df2 = df.drop_duplicates(subset=subset) if subset else df.drop_duplicates()
    after = len(df2)
    if after != before:
        print(f"[{dataset}] drop_duplicates {label}: {before} -> {after}")
    return df2


def _drop_duplicates_smart(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Strategi deduplikasi yang stabil:
    1) Jika ada 'id' -> pakai 'id'
    2) Kalau tidak ada, gunakan kombinasi kunci yang tersedia dari prioritas berikut
       (ambil yang ada saja; minimal 3 kolom agar kunci cukup ketat):
       year, month, day, hr, mn, sec, latitude, longitude, name, location, country
    """
    df = df.copy()
    if "id" in df.columns:
        return _drop_and_log(df, ["id"], dataset, "by 'id'")

    priority = [
        "year",
        "month",
        "day",
        "hr",
        "mn",
        "sec",
        "latitude",
        "longitude",
        "name",
        "location",
        "country",
    ]
    keys = [c for c in priority if c in df.columns]
    if len(keys) >= 3:
        return _drop_and_log(df, keys, dataset, f"by {keys}")
    return _drop_and_log(df, None, dataset, "on all columns")


# ---------- time-erupt parser (volcanic) ----------
def _extract_time_cols(df: pd.DataFrame, time_col: str = "time_erupt") -> pd.DataFrame:
    """
    Parse kolom waktu letusan (mis. 'time_erupt') menjadi hr, mn, sec bila memungkinkan.
    Format yang ditangani:
      - 'HH:MM', 'HH:MM:SS'
      - 'HHMM' atau 'HHMMSS'
    Jika format tidak cocok -> diisi NaN.
    """
    if time_col not in df.columns:
        return df

    def _split_one(val: object) -> tuple[float, float, float]:
        if pd.isna(val):
            return (np.nan, np.nan, np.nan)
        txt = str(val).strip().lower()
        if not txt or txt in {"nan", "none"}:
            return (np.nan, np.nan, np.nan)

        txt = txt.replace(".", ":").replace("-", ":")
        if ":" in txt:
            parts = [p for p in txt.split(":") if p]
        else:
            digits = "".join(ch for ch in txt if ch.isdigit())
            if len(digits) == 4:
                parts = [digits[:2], digits[2:4]]
            elif len(digits) == 6:
                parts = [digits[:2], digits[2:4], digits[4:6]]
            else:
                return (np.nan, np.nan, np.nan)

        def _to_int(s: str) -> float:
            try:
                return float(int(s))
            except Exception:
                return float("nan")

        h = _to_int(parts[0]) if len(parts) >= 1 else np.nan
        m = _to_int(parts[1]) if len(parts) >= 2 else np.nan
        s_val = _to_int(parts[2]) if len(parts) >= 3 else np.nan
        return (h, m, s_val)

    tuples = df[time_col].apply(_split_one)
    hr = tuples.apply(lambda t: t[0])
    mn = tuples.apply(lambda t: t[1])
    sec = tuples.apply(lambda t: t[2])

    out = df.copy()
    out["hr"] = pd.to_numeric(hr, errors="coerce")
    out["mn"] = pd.to_numeric(mn, errors="coerce")
    out["sec"] = pd.to_numeric(sec, errors="coerce")
    return out


# ================== DATASET-SPECIFIC PREP ==================
def prepare_tectonic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning domain tektonik.
    Target akhir kolom & tipe data (Tabel 6):
    - id          : float64
    - year        : float64
    - month       : float64
    - day         : float64
    - hr          : float64
    - mn          : float64
    - sec         : float64
    - country     : object
    - area        : object
    - region      : float64
    - location    : object
    - latitude    : float64
    - longitude   : float64
    - depth       : float64
    - mag         : float64
    - tsu         : int64
    """
    # mapping khusus dari spesifikasi kolom raw
    df = _standardize_columns(
        df,
        col_map={
            "focal_depth_(km)": "depth",
            "location_name": "location",
        },
    )
    df = _clean_whitespace(df)
    df = _map_tsu_binary(df)

    # ke numerik (termasuk 'id' & 'region' agar sesuai spesifikasi)
    df = _to_numeric(
        df,
        [
            "id",
            "year",
            "mo",
            "dy",
            "hr",
            "mn",
            "sec",
            "latitude",
            "longitude",
            "depth",
            "mag",
            "mw",
            "ms",
            "mb",
            "ml",
            "region",
        ],
    )

    # Mo/Dy -> month/day
    df = df.rename(columns={"mo": "month", "dy": "day"})

    df = _normalize_calendar(df)
    df = _drop_out_of_bounds(df)

    wanted = [
        "id",
        "year",
        "month",
        "day",
        "hr",
        "mn",
        "sec",
        "country",
        "area",
        "region",
        "location",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "tsu",
    ]
    existing = [c for c in wanted if c in df.columns]
    df = df[existing].copy()

    # wajib ada koordinat + tahun; bulan/hari boleh kosong untuk data historis
    must_have = [c for c in ("latitude", "longitude", "year", "tsu") if c in df.columns]
    df = df.dropna(subset=must_have)

    # deduplikasi pintar
    df = _drop_duplicates_smart(df, dataset="tectonic")

    # pastikan kolom unik & index rapi
    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

    # ====== ENFORCE TIPE DATA SESUAI TABEL 6 ======
    num_cols_float = [
        "id",
        "year",
        "month",
        "day",
        "hr",
        "mn",
        "sec",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "region",
    ]
    for c in num_cols_float:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in ["country", "area", "location"]:
        if c in df.columns:
            df[c] = df[c].astype("object")

    if "tsu" in df.columns:
        df["tsu"] = df["tsu"].astype("int64")

    return df


def prepare_volcanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning domain vulkanik.
    Target akhir kolom & tipe data (Tabel 7):
    - id          : float64
    - year        : float64
    - month       : float64
    - day         : float64
    - name        : object
    - location    : object
    - country     : object
    - latitude    : float64
    - longitude   : float64
    - elevation   : float64
    - type        : object
    - status      : object
    - vei         : float64
    - eq          : float64
    - agent       : object
    - tsu         : int64
    """
    df = _standardize_columns(
        df,
        col_map={
            "elevation_(m)": "elevation",
            "time_erupt": "time_erupt",
        },
    )
    df = _clean_whitespace(df)
    df = _map_tsu_binary(df)
    df = _extract_time_cols(df, time_col="time_erupt")

    df = _to_numeric(
        df,
        [
            "id",
            "year",
            "mo",
            "dy",
            "hr",
            "mn",
            "sec",
            "latitude",
            "longitude",
            "elevation",
            "vei",
            "eq",
        ],
    )
    df = df.rename(columns={"mo": "month", "dy": "day"})

    df = _normalize_calendar(df)
    df = _drop_out_of_bounds(df)

    wanted = [
        "id",
        "year",
        "month",
        "day",
        "hr",
        "mn",
        "sec",
        "name",
        "location",
        "country",
        "latitude",
        "longitude",
        "elevation",
        "type",
        "status",
        "vei",
        "eq",
        "agent",
        "tsu",
    ]
    existing = [c for c in wanted if c in df.columns]
    df = df[existing].copy()

    must_have = [c for c in ("latitude", "longitude", "year", "tsu") if c in df.columns]
    df = df.dropna(subset=must_have)

    df = _drop_duplicates_smart(df, dataset="volcanic")

    df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)

    # ====== ENFORCE TIPE DATA SESUAI TABEL 7 ======
    num_cols_float = [
        "id",
        "year",
        "month",
        "day",
        "latitude",
        "longitude",
        "elevation",
        "vei",
        "eq",
    ]
    for c in num_cols_float:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in ["name", "location", "country", "type", "status", "agent"]:
        if c in df.columns:
            df[c] = df[c].astype("object")

    if "tsu" in df.columns:
        df["tsu"] = df["tsu"].astype("int64")

    return df


# ================== NORMALIZATION + ENCODING ==================
def encode_and_scale(
    df: pd.DataFrame,
    dataset: str,
    normalize: bool = True,
) -> Tuple[pd.DataFrame, Pipeline | None, List[str], List[str]]:
    """
    Buat dataset final fitur (scaled + one-hot) + kolom target 'tsu' di akhir.
    NOTE: Untuk training final, scaler/encoder akan berada di dalam CV Pipeline.
    """
    if "tsu" not in df.columns:
        raise KeyError(
            f"[{dataset}] kolom target 'tsu' tidak ditemukan setelah cleaning."
        )

    # pastikan tsu int64
    y = df["tsu"].astype("int64")

    if dataset == "tectonic":
        num_cols = [
            c
            for c in (
                "mag",
                "depth",
                "latitude",
                "longitude",
                "year",
                "month",
                "day",
            )
            if c in df.columns
        ]
        cat_cols = [
            c
            for c in ("country", "region", "area", "location")
            if c in df.columns
        ]
    else:
        num_cols = [
            c
            for c in (
                "vei",
                "elevation",
                "latitude",
                "longitude",
                "year",
                "month",
                "day",
                "eq",
            )
            if c in df.columns
        ]
        cat_cols = [
            c
            for c in ("country", "type", "status", "location", "name")
            if c in df.columns
        ]

    X = df[num_cols + cat_cols].copy()

    if not normalize:
        out = X.copy()
        out["tsu"] = y.values
        return out, None, num_cols, cat_cols

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=0.01,
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipe = Pipeline([("ct", ct)])

    X_t = pipe.fit_transform(X)
    out_cols = pipe.get_feature_names_out()
    X_df = pd.DataFrame(X_t, columns=out_cols, index=X.index)
    X_df["tsu"] = y.values

    dump(pipe, ART / f"{dataset}_preprocess_pipe.joblib")
    dump(num_cols, ART / f"{dataset}_num_cols.joblib")
    dump(cat_cols, ART / f"{dataset}_cat_cols.joblib")
    pd.DataFrame({"feature": out_cols}).to_csv(
        TAB / f"{dataset}_feature_names.csv", index=False
    )

    return X_df, pipe, num_cols, cat_cols


# ================== WRITE/REUSE HELPERS ==================
def _save_or_reuse_clean(
    df_t_clean: pd.DataFrame,
    df_v_clean: pd.DataFrame,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_t_clean = PROCESSED / "tectonic.csv"
    out_v_clean = PROCESSED / "volcanic.csv"

    if out_t_clean.exists() and out_v_clean.exists() and not overwrite:
        print(
            "[INFO] Clean CSVs already exist — reusing. "
            "Use --overwrite to regenerate."
        )
        # gunakan versi di disk agar konsisten dengan langkah berikutnya
        return pd.read_csv(out_t_clean), pd.read_csv(out_v_clean)

    # tulis ulang (overwrite True atau belum ada file)
    df_t_clean.to_csv(out_t_clean, index=False)
    df_v_clean.to_csv(out_v_clean, index=False)
    return df_t_clean, df_v_clean


def _maybe_write_csv(
    df: pd.DataFrame,
    path: Path,
    overwrite: bool,
    label: str,
) -> None:
    if path.exists() and not overwrite:
        print(f"[INFO] Reusing existing {label}: {path.name}")
    else:
        df.to_csv(path, index=False)


# ================== MAIN ==================
def main(overwrite: bool = False, normalize: bool = True) -> None:
    # cari file mentah (nama baru & fallback)
    raw_tec = next(
        (
            p
            for p in (
                RAW / "tectonic.csv",
                RAW / "raw_tectonic_global.csv",
                RAW / "raw_tectonic.csv",
            )
            if p.exists()
        ),
        None,
    )
    raw_vul = next(
        (
            p
            for p in (
                RAW / "volcanic.csv",
                RAW / "raw_volcanic_global.csv",
                RAW / "raw_volcanic.csv",
            )
            if p.exists()
        ),
        None,
    )
    if raw_tec is None or raw_vul is None:
        raise FileNotFoundError(
            "Raw files not found.\n"
            f"  Tectonic : {RAW / 'tectonic.csv'} (fallback: raw_tectonic_global.csv)\n"
            f"  Volcanic : {RAW / 'volcanic.csv'} (fallback: raw_volcanic_global.csv)"
        )

    # baca raw
    df_t_raw = _read_csv_guess(raw_tec)
    df_v_raw = _read_csv_guess(raw_vul)

    # ---------- FILTERING + CLEANING ----------
    df_t_clean0 = prepare_tectonic(df_t_raw)
    df_v_clean0 = prepare_volcanic(df_v_raw)

    # simpan/reuse clean
    df_t_clean, df_v_clean = _save_or_reuse_clean(
        df_t_clean0,
        df_v_clean0,
        overwrite=overwrite,
    )

    # QC tables untuk CLEAN (selalu update agar sesuai isi terkini)
    _save_info_tables(df_t_clean, "tectonic_clean")
    _save_info_tables(df_v_clean, "volcanic_clean")

    # Tabel skema + contoh data nyata (Tabel 6 & 7 + sample)
    _save_schema_and_samples(df_t_clean, "tectonic")
    _save_schema_and_samples(df_v_clean, "volcanic")

    # ---------- (OPSIONAL) NORMALIZATION + ENCODING ----------
    df_t_prep, _, _, _ = encode_and_scale(
        df_t_clean,
        dataset="tectonic",
        normalize=normalize,
    )
    df_v_prep, _, _, _ = encode_and_scale(
        df_v_clean,
        dataset="volcanic",
        normalize=normalize,
    )

    out_t_prep = PROCESSED / "tectonic_preprocessed.csv"
    out_v_prep = PROCESSED / "volcanic_preprocessed.csv"
    _maybe_write_csv(df_t_prep, out_t_prep, overwrite, "preprocessed tectonic")
    _maybe_write_csv(df_v_prep, out_v_prep, overwrite, "preprocessed volcanic")

    _save_shape_summary(
        [
            ("tectonic_clean", len(df_t_clean), df_t_clean.shape[1]),
            ("volcanic_clean", len(df_v_clean), df_v_clean.shape[1]),
            ("tectonic_preprocessed", len(df_t_prep), df_t_prep.shape[1]),
            ("volcanic_preprocessed", len(df_v_prep), df_v_prep.shape[1]),
        ]
    )

    print(
        "[DONE] Preprocessing selesai.\n"
        f"- Clean (tectonic)  : {PROCESSED / 'tectonic.csv'}\n"
        f"- Clean (volcanic)  : {PROCESSED / 'volcanic.csv'}\n"
        f"- Preprocessed (tec): {out_t_prep}\n"
        f"- Preprocessed (vul): {out_v_prep}\n"
        f"- Artifacts (pipes/cols): {ART}\n"
        f"- QC tables & schema: {TAB}\n"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Preprocessing = Filtering + Cleaning + "
            "(Optional) Normalization+Encoding"
        )
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace processed files if exist",
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip normalization & encoding, save only clean CSVs",
    )
    args = ap.parse_args()
    main(overwrite=args.overwrite, normalize=not args.no_normalize)