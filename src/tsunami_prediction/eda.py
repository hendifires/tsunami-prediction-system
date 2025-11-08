from __future__ import annotations
# cSpell:ignore whitegrid savetab

import argparse
from pathlib import Path
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"


# ---------- UTIL ----------
def _ensure_dirs() -> None:
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _savetab(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _info_to_txt(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def _load_dataset(name: str) -> pd.DataFrame:
    """
    Ambil dataset hasil preprocessing/cleaning (bukan hasil FE spesifik).
    Urutan prioritas:
    1) <name>.csv             (clean)
    2) <name>_cleaned.csv     (legacy)
    3) <name>_biner.csv       (legacy)
    4) <name>_preprocessed.csv
    """
    candidates = [
        PROCESSED / f"{name}.csv",               # hasil cleaning utama
        PROCESSED / f"{name}_cleaned.csv",       # fallback lama
        PROCESSED / f"{name}_biner.csv",         # fallback lama
        PROCESSED / f"{name}_preprocessed.csv",  # fallback lain
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            return df
    raise FileNotFoundError(
        f"[EDA] Processed dataset not found for '{name}'. Tried: {candidates}"
    )


# ---------- EDA PRIMITIVES ----------
def class_distribution(df: pd.DataFrame, target: str, name: str) -> None:
    """
    Tujuan:
    - Menunjukkan keseimbangan / ketidakseimbangan kelas target (tsu=0 vs tsu=1).
    - Dipakai untuk menjelaskan masalah class imbalance di tesis.

    Output:
    - CSV ringkasan jumlah & persentase per kelas.
    - Bar chart distribusi kelas (visual utama, tanpa pie chart agar tidak redundant).
    """
    if target not in df.columns:
        raise KeyError(f"[{name}] Target column '{target}' tidak ditemukan.")

    ct = (
        df[target]
        .value_counts(dropna=False)
        .rename_axis(target)
        .reset_index(name="count")
        .sort_values(target)
    )
    total = ct["count"].sum()
    ct["percent"] = (ct["count"] / total * 100).round(2)
    _savetab(ct, TAB / f"{name}_class_counts.csv")

    # bar (visual utama untuk imbalance)
    plt.figure(figsize=(4.6, 3.4))
    ax = sns.countplot(x=target, data=df)
    ax.bar_label(ax.containers[0])
    plt.title(f"Class Distribution — {name}")
    _savefig(FIG / f"{name}_class_bar.png")


def missing_value_fraction(df: pd.DataFrame, name: str) -> None:
    """
    Tujuan:
    - Menggambarkan kualitas data melalui proporsi missing value per kolom.
    - Mendukung penjelasan pembersihan data di bagian preprocessing.

    Output:
    - CSV missing_fraction per kolom.
    - Bar chart missing_fraction.
    """
    mv = df.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    mv_df = mv.rename_axis("column").reset_index()
    _savetab(mv_df, TAB / f"{name}_missing_fraction.csv")

    plt.figure(figsize=(min(12, 0.35 * len(mv_df) + 4), 3.8))
    mv.plot(kind="bar")
    plt.ylabel("Missing Fraction")
    plt.title(f"Missing Value Fraction per Column — {name}")
    _savefig(FIG / f"{name}_missing_bar.png")


def numeric_distributions(
    df: pd.DataFrame,
    numeric_cols: list[str],
    name: str,
) -> None:
    """
    Tujuan:
    - Menggambarkan bentuk distribusi fitur numerik (skewed, normal, multimodal).
    - Membantu interpretasi skala & outlier sebelum pemodelan.

    Output:
    - Histogram + KDE per fitur numerik (tanpa boxplot global agar tidak terlalu ramai).
    """
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        return

    cols = min(4, len(numeric_cols))
    rows = int(np.ceil(len(numeric_cols) / cols))

    plt.figure(figsize=(4 * cols, 2.8 * rows))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(col)
    plt.suptitle(f"Histograms — {name}", y=1.02)
    _savefig(FIG / f"{name}_hists.png")


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], name: str) -> None:
    """
    Tujuan:
    - (Opsional) Menunjukkan struktur korelasi antar fitur numerik saja.
    - TIDAK dipanggil di run_full_eda agar tidak redundant dengan correlation_with_target.

    Kalau ingin dipakai manual, panggil fungsi ini di skrip terpisah.
    """
    use = [c for c in numeric_cols if c in df.columns]
    if len(use) < 2:
        return

    corr = df[use].corr(numeric_only=True)
    corr.to_csv(TAB / f"{name}_corr.csv", index=True)

    plt.figure(figsize=(0.8 * len(use) + 3, 0.8 * len(use) + 3))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
    )
    plt.title(f"Correlation Heatmap — {name}")
    _savefig(FIG / f"{name}_corr_heatmap.png")


def correlation_with_target(
    df: pd.DataFrame,
    numeric_cols: list[str],
    target: str,
    name: str,
) -> None:
    """
    Tujuan:
    - Fokus pada hubungan linear antara fitur numerik dan target tsu.
    - Dipakai sebagai dasar pemilihan fitur / interpretasi pentingnya variabel.

    Output:
    - CSV matriks korelasi (termasuk target).
    - Heatmap korelasi (termasuk target) → ini yang bisa kamu jadikan gambar
      di bab EDA untuk menjelaskan fitur mana yang paling berkorelasi dengan tsu.
    """
    cols = [c for c in numeric_cols if c in df.columns]
    if target in df.columns and target not in cols:
        cols.append(target)
    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)
    corr.to_csv(TAB / f"{name}_corr_with_target.csv", index=True)

    plt.figure(figsize=(0.8 * len(cols) + 3, 0.8 * len(cols) + 3))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
    )
    plt.title(f"Correlation Matrix (including {target}) — {name}")
    _savefig(FIG / f"{name}_corr_with_target.png")


def pairplot_sample(
    df: pd.DataFrame,
    cols: list[str],
    hue: str,
    name: str,
    max_rows: int = 800,
) -> None:
    """
    Tujuan:
    - Melihat hubungan pasangan fitur numerik sekaligus (pairwise),
      sekaligus separability antara tsu=0 vs tsu=1.
    - Digunakan sebagai visual kualitatif: apakah kelas tsunami membentuk cluster berbeda.

    Catatan:
    - Hanya pakai subset baris (max_rows) supaya tidak berat.
    - Hanya numerik + hue (target).
    """
    use_cols = [c for c in cols if c in df.columns]
    if hue in use_cols:
        use_cols.remove(hue)

    num_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = list(dict.fromkeys(num_cols))  # buang duplikat

    if hue not in df.columns or len(num_cols) < 2:
        return

    sub = df[num_cols + [hue]].dropna()
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=42)

    # buang kolom yang berisi list/array
    num_cols = [
        c
        for c in num_cols
        if not sub[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()
    ]
    if len(num_cols) < 2:
        return

    g = sns.pairplot(
        sub,
        vars=num_cols,
        hue=hue,
        palette="Set1",
        plot_kws={"alpha": 0.7, "s": 25},
        height=2.4,
    )
    g.fig.suptitle(f"{name.capitalize()}: Pairplot (sample)", y=1.03)
    _savefig(FIG / f"{name}_pairplot.png")


def numeric_vs_target_distributions(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    name: str,
    outfile: str | None = None,
) -> None:
    """
    Tujuan:
    - Menunjukkan perbedaan distribusi fitur numerik antara kelas tsu=0 dan tsu=1.
    - Sangat relevan untuk menjelaskan mengapa fitur tertentu kuat membedakan tsunami.

    Output:
    - Histogram per fitur numerik dengan hue=target.
    """
    if target not in df.columns:
        return

    cols = [c for c in cols if c in df.columns]
    if not cols:
        return

    ncols = min(2, len(cols))
    nrows = int(np.ceil(len(cols) / ncols))

    plt.figure(figsize=(5 * ncols, 3.2 * nrows))
    for i, col in enumerate(cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.histplot(
            data=df,
            x=col,
            hue=target,
            kde=True,
            bins=30,
            stat="count",
            common_norm=False,
            element="step",
        )
        plt.title(col)

    plt.suptitle(f"Numeric Features vs {target} — {name}", y=1.02)
    fname = outfile or f"{name}_num_vs_{target}.png"
    _savefig(FIG / fname)


def temporal_plots(
    df: pd.DataFrame,
    name: str,
    target: str = "tsu",
) -> None:
    """
    Tujuan:
    - Melihat tren jumlah kejadian tsunami (tsu=1) per tahun.
    - Membantu menjelaskan pola sejarah (misal tahun-tahun dengan aktivitas tinggi).

    Output:
    - CSV hitungan kejadian per tahun.
    - CSV tahun-tahun yang dianggap 'spike' (di atas mean + 2*std).
    - Plot garis jumlah kejadian per tahun.
    """
    if "year" not in df.columns:
        return

    dfx = df.copy()
    if target in dfx.columns:
        dfx = dfx[dfx[target] == 1]

    yr = pd.to_numeric(dfx["year"], errors="coerce").dropna().astype(int)
    if yr.empty:
        return

    yearly = yr.value_counts().sort_index()
    year_df = yearly.rename("count").reset_index().rename(columns={"index": "year"})
    _savetab(year_df, TAB / f"{name}_events_by_year.csv")

    mean = float(yearly.mean())
    std = float(yearly.std())
    spike = yearly[yearly > mean + 2 * std]
    _savetab(
        spike.rename("count").reset_index().rename(columns={"index": "spike_year"}),
        TAB / f"{name}_spike_years.csv",
    )

    plt.figure(figsize=(9, 4))
    plt.plot(yearly.index, yearly.values, marker="o")
    for y in spike.index:
        plt.axvline(y, color="red", ls="--", alpha=0.5)
    plt.title(f"Events per Year (tsu=1) — {name}")
    plt.xlabel("Year")
    plt.ylabel("Count")
    _savefig(FIG / f"{name}_by_year.png")


def spatial_scatter(df: pd.DataFrame, name: str, target: str = "tsu") -> None:
    """
    Tujuan:
    - Memvisualisasikan sebaran spasial (longitude/latitude) kejadian tsunami.
    - Menunjukkan konsentrasi wilayah rawan (misal cincin api Pasifik, dsb).

    Output:
    - Scatter plot global (fallback).
    - Kalau environment support PyGMT + tilemap, akan memakai peta dunia.
    """
    import os

    if not {"latitude", "longitude"}.issubset(df.columns):
        return

    def _fallback() -> None:
        plt.figure(figsize=(7.5, 3.8))
        plt.axhline(0, color="lightgray", lw=0.8)
        plt.axvline(0, color="lightgray", lw=0.8)
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)

        if target in df.columns:
            sns.scatterplot(
                data=df,
                x="longitude",
                y="latitude",
                hue=target,
                s=8,
                alpha=0.6,
                palette="Set1",
            )
            plt.legend(title=target, loc="upper right")
        else:
            plt.scatter(df["longitude"], df["latitude"], s=8, alpha=0.6)

        plt.title(f"Global Scatter (lon/lat) — {name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        _savefig(FIG / f"{name}_global_scatter.png")

    if os.getenv("EDA_USE_PYGMT", "0") != "1":
        _fallback()
        return

    try:
        import pygmt  # type: ignore

        with pygmt.config(VERBOSE="q"):
            fig = pygmt.Figure()
            fig.tilemap(
                region=[-180, 180, -75, 75],
                projection="M120/0/20c",
                source=(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/"
                    "World_Street_Map/MapServer/tile/{z}/{y}/{x}.png"
                ),
                frame=["xafg", "yafg", f"+t{name.capitalize()} Events: Global Distribution"],
            )
            if target in df.columns:
                for val in sorted(pd.Series(df[target]).dropna().unique()):
                    sub = df[df[target] == val]
                    fig.plot(
                        x=sub["longitude"],
                        y=sub["latitude"],
                        style="c0.08c",
                        fill="red" if int(val) == 1 else "dodgerblue",
                        pen="black",
                        label=f"tsu={int(val)}",
                    )
                fig.legend(position="JTR+jTR+o0.3c", box=True)
            else:
                fig.plot(
                    x=df["longitude"],
                    y=df["latitude"],
                    style="c0.08c",
                    fill="black",
                    pen="black",
                )
            fig.savefig(str(FIG / f"{name}_global_map.png"))
    except Exception:
        _fallback()


# ---------- SMALL HELPERS ----------
def _plot_top_counts(
    s: pd.Series,
    title: str,
    xlabel: str,
    outfile: str,
    orient: str = "h",
    table_name: str | None = None,
) -> None:
    """
    Tujuan:
    - Menampilkan kategori teratas (region, type, country, dsb) berdasarkan frekuensi.
    """
    s = s.dropna()
    if s.empty:
        return

    if table_name is not None:
        _savetab(
            s.rename_axis(s.index.name or "key").reset_index(name="count"),
            TAB / table_name,
        )

    plt.figure(figsize=(9, 4.2))
    if orient == "h":
        sns.barplot(x=s.values, y=s.index)
        plt.xlabel(xlabel)
        plt.ylabel(s.index.name or "")
    else:
        sns.barplot(x=s.index, y=s.values)
        plt.ylabel(xlabel)
        plt.xlabel(s.index.name or "")
        plt.xticks(rotation=30, ha="right")
    plt.title(title)
    _savefig(FIG / outfile)


def _plot_top_mean(
    series: pd.Series,
    title: str,
    xlabel: str,
    outfile: str,
    table_name: str,
) -> None:
    """
    Tujuan:
    - Menunjukkan rata-rata nilai numerik (misal magnitudo atau VEI)
      per kategori (negara) untuk top 10.
    """
    s = series.sort_values(ascending=False).head(10)
    if s.empty:
        return

    _savetab(s.rename(series.name).reset_index(), TAB / table_name)
    plt.figure(figsize=(9, 4))
    sns.barplot(x=s.values, y=s.index)
    plt.title(title)
    plt.xlabel(xlabel)
    _savefig(FIG / outfile)


def categorical_and_country_plots(df_t: pd.DataFrame, df_v: pd.DataFrame) -> None:
    """
    Tujuan:
    - Visual khusus kategori untuk mendukung narasi domain:
      * Tektonik: region, negara dengan tsunami, rata-rata magnitudo.
      * Vulkanik: type gunung api, negara dengan tsunami, rata-rata VEI.
    """
    # Tektonik: top 8 region
    if "region" in df_t.columns:
        top = df_t["region"].value_counts().nlargest(8)
        _plot_top_counts(
            top,
            title="Tectonic: Top 8 Region",
            xlabel="Number of Earthquake Events",
            outfile="tectonic_top_regions.png",
            orient="h",
            table_name="tectonic_top_regions.csv",
        )

    # Tektonik: top 10 negara dengan tsunami
    if {"country", "tsu"}.issubset(df_t.columns):
        s = df_t[df_t["tsu"] == 1]["country"].value_counts().nlargest(10)
        _plot_top_counts(
            s,
            title="Tectonic: Top 10 Countries with Tsunami (tsu=1)",
            xlabel="Number of Tsunamis",
            outfile="tectonic_top_countries_tsu.png",
            orient="h",
            table_name="tectonic_top_countries_tsu.csv",
        )

    # Tektonik: rata-rata magnitudo per negara
    if {"country", "mag"}.issubset(df_t.columns):
        mean_mag = df_t.groupby("country")["mag"].mean()
        _plot_top_mean(
            mean_mag,
            title="Tectonic: Top 10 Average Magnitude per Country",
            xlabel="Average Magnitude",
            outfile="tectonic_mean_mag_by_country.png",
            table_name="tectonic_mean_mag_by_country.csv",
        )

    # Vulkanik: top 8 type
    if "type" in df_v.columns:
        top = df_v["type"].value_counts().nlargest(8)
        _plot_top_counts(
            top,
            title="Volcanic: Top 8 Volcano Type",
            xlabel="Number of eruption events",
            outfile="volcanic_top_types.png",
            orient="h",
            table_name="volcanic_top_types.csv",
        )

    # Vulkanik: top 10 negara dengan tsunami
    if {"country", "tsu"}.issubset(df_v.columns):
        s = df_v[df_v["tsu"] == 1]["country"].value_counts().nlargest(10)
        _plot_top_counts(
            s,
            title="Volcanic: Top 10 Countries with Tsunami (tsu=1)",
            xlabel="Number of Tsunamis",
            outfile="volcanic_top_countries_tsu.png",
            orient="h",
            table_name="volcanic_top_countries_tsu.csv",
        )

    # Vulkanik: rata-rata VEI per negara
    if {"country", "vei"}.issubset(df_v.columns):
        mean_vei = df_v.groupby("country")["vei"].mean()
        _plot_top_mean(
            mean_vei,
            title="Top 10 Average VEI per Country",
            xlabel="Average VEI",
            outfile="volcanic_mean_vei_by_country.png",
            table_name="volcanic_mean_vei_by_country.csv",
        )


# ---------- OPTIONAL: TIME FIELDS HELPER ----------
def _add_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambahkan kolom year/month/day/hour/minute/second bila bisa dideduksi
    dari kolom 'time' atau 'origin_time'. Tidak menimpa kolom yang sudah ada.
    """
    out = df.copy()

    # cari kolom kandidat
    lower_to_col = {c.lower(): c for c in out.columns}
    time_col = next(
        (lower_to_col[key] for key in ("time", "origin_time") if key in lower_to_col),
        None,
    )
    if time_col is None:
        return out

    t = pd.to_datetime(out[time_col], errors="coerce", utc=False)

    parts: dict[str, pd.Series] = {
        "year": t.dt.year,
        "month": t.dt.month,
        "day": t.dt.day,
        "hour": t.dt.hour,
        "minute": t.dt.minute,
        "second": t.dt.second,
    }
    for name, values in parts.items():
        if name not in out.columns:
            out[name] = values

    return out


# ---------- ENGINEERED FEATURES VISUAL ----------
def engineered_feature_plots(df: pd.DataFrame, name: str, target: str = "tsu") -> None:
    """
    Tujuan:
    - Mempertegas visualisasi untuk fitur hasil rekayasa (feature engineering),
      misalnya jarak ke garis pantai, dsb.

    Implementasi:
    - Secara otomatis akan mencari kolom FE yang umum (misal: distance_to_coast_km).
    - Bila ada, dibuat:
        * histogram per kelas tsu (numeric_vs_target_distributions)
        * korelasi dengan target.

    Ini bisa kamu rujuk di bab Feature Engineering.
    """
    fe_candidates = [
        "distance_to_coast_km",
        # tambahkan di sini kalau kamu punya fitur rekayasa lain
        # "max_wave_height",
        # "time_to_coast_min",
    ]
    fe_cols = [c for c in fe_candidates if c in df.columns]
    if not fe_cols:
        return

    numeric_vs_target_distributions(
        df,
        cols=fe_cols,
        target=target,
        name=name,
        outfile=f"{name}_fe_num_vs_{target}.png",
    )

    # Korelasi fokal: FE dan target saja
    cols = fe_cols + ([target] if target in df.columns else [])
    corr = df[cols].corr(numeric_only=True)
    _savetab(corr.reset_index().rename(columns={"index": "feature"}), TAB / f"{name}_fe_corr_with_target.csv")

    plt.figure(figsize=(0.8 * len(cols) + 3, 0.8 * len(cols) + 3))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
    )
    plt.title(f"Engineered Features Correlation with {target} — {name}")
    _savefig(FIG / f"{name}_fe_corr_with_target.png")


# ---------- MASTER RUN ----------
def run_full_eda(df: pd.DataFrame, ds_name: str, target: str = "tsu") -> None:
    """
    Pipeline EDA lengkap untuk satu domain (tektonik / vulkanik).

    Output utama per domain:
    - Informasi struktur data (info, describe, head).
    - Distribusi kelas (imbalance).
    - Kualitas data (missing value).
    - Distribusi fitur numerik.
    - Korelasi fitur numerik + target (heatmap).
    - Tren waktu kejadian tsunami (per tahun).
    - Scatter spasial lon/lat.
    - Visual tambahan spesifik domain (mag/depth untuk tektonik, VEI/eq untuk vulkanik).
    - Visual fitur rekayasa (jika ada).
    """
    # simpan basic info
    (TAB / f"{ds_name}_info.txt").write_text(_info_to_txt(df), encoding="utf-8")
    _savetab(df.head(20), TAB / f"{ds_name}_head20.csv")
    _savetab(
        df.describe(include="all").T.reset_index().rename(columns={"index": "feature"}),
        TAB / f"{ds_name}_describe.csv",
    )

    # normalisasi nama tanggal kalau masih mo/dy
    if "mo" in df.columns or "dy" in df.columns:
        df = df.rename(columns={"mo": "month", "dy": "day"})

    # daftar kolom numerik (tanpa target & tanpa id)
    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {target, "id"}
    ]

    # 1) Distribusi kelas
    class_distribution(df, target, ds_name)

    # 2) Kualitas data
    missing_value_fraction(df, ds_name)

    # 3) Distribusi numerik umum
    numeric_distributions(df, numeric_cols, ds_name)

    # 4) Korelasi fitur numerik dengan target (visual utama korelasi)
    correlation_with_target(df, numeric_cols, target, ds_name)
    # (correlation_heatmap tidak dipanggil di sini untuk menghindari duplikasi fungsi)

    # 5) Analisis temporal & spasial
    temporal_plots(df, ds_name, target=target)
    spatial_scatter(df, ds_name, target=target)

    # 6) Visual khusus per dataset
    if ds_name == "tectonic":
        numeric_vs_target_distributions(
            df,
            cols=["mag", "depth", "latitude", "longitude"],
            target=target,
            name=ds_name,
            outfile="tectonic_num_vs_tsu.png",
        )
        pp_cols = ["mag", "depth", "latitude", "longitude", target]
    else:
        numeric_vs_target_distributions(
            df,
            cols=["eq", "elevation", "latitude", "longitude"],
            target=target,
            name=ds_name,
            outfile="volcanic_num_vs_tsu.png",
        )
        pp_cols = ["vei", "elevation", "latitude", "longitude", target]

    pairplot_sample(df, pp_cols, hue=target, name=ds_name)

    # 7) Visual fitur rekayasa (jika ada, mis. distance_to_coast_km)
    engineered_feature_plots(df, ds_name, target=target)


def main(datasets: list[str]) -> None:
    _ensure_dirs()

    dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        df = _load_dataset(ds)
        if "tsu" not in df.columns:
            raise KeyError(
                f"'{ds}' tidak memiliki kolom target 'tsu' setelah preprocessing."
            )
        dfs[ds] = df
        print(f"[EDA] start {ds} -> rows={len(df)} cols={df.shape[1]}")

    for ds, df in dfs.items():
        run_full_eda(df, ds, target="tsu")

    # Visual kategori lintas domain (negara, type, region)
    if "tectonic" in dfs and "volcanic" in dfs:
        categorical_and_country_plots(dfs["tectonic"], dfs["volcanic"])

    print(f"[DONE] EDA saved -> {FIG} and {TAB}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for tectonic/volcanic")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["tectonic", "volcanic"],
        help="list dataset names: tectonic volcanic",
    )
    args = parser.parse_args()
    main(args.datasets)