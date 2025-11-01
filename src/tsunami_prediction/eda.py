# src/tsunami_prediction/eda.py
from __future__ import annotations

import argparse
from pathlib import Path
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker as mticker

sns.set_theme(style="whitegrid")

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
TAB = REPORTS / "tables"
FIG = REPORTS / "figures"


# ---------- UTIL ----------
def _ensure_dirs():
    for p in [TAB, FIG]:
        p.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _savetab(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _info_to_txt(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def _load_dataset(name: str) -> pd.DataFrame:
    """
    Prioritaskan hasil 'clean' (bukan preprocessed) agar kolom asli masih ada.
    Pastikan juga kolom unik (hapus duplikat nama kolom bila ada).
    """
    cands = [
        PROCESSED / f"{name}.csv",                 # hasil cleaning
        PROCESSED / f"{name}_cleaned.csv",         # fallback lama
        PROCESSED / f"{name}_biner.csv",           # fallback lama
        PROCESSED / f"{name}_preprocessed.csv",    # kalau hanya ini yang ada
    ]
    for p in cands:
        if p.exists():
            df = pd.read_csv(p)
            # singkirkan kolom duplikat (jaga-jaga)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            return df
    raise FileNotFoundError(f"[EDA] Processed dataset not found for '{name}'. Tried: {cands}")


# ---------- EDA PRIMITIVES ----------
def class_distribution(df: pd.DataFrame, target: str, name: str):
    if target not in df.columns:
        raise KeyError(f"[{name}] Target column '{target}' tidak ditemukan.")
    ct = (
        df[target].value_counts(dropna=False)
        .rename_axis(target)
        .reset_index(name="count")
        .sort_values(target)
    )
    total = ct["count"].sum()
    ct["percent"] = (ct["count"] / total * 100).round(2)
    _savetab(ct, TAB / f"{name}_class_counts.csv")

    # bar
    plt.figure(figsize=(4.6, 3.4))
    ax = sns.countplot(x=target, data=df)
    ax.bar_label(ax.containers[0])
    plt.title(f"Class Distribution — {name}")
    _savefig(FIG / f"{name}_class_bar.png")

    # pie
    plt.figure(figsize=(4.2, 4.2))
    labels = [f"{int(v)} ({p:.1f}%)" for v, p in zip(ct["count"].values, ct["percent"].values)]
    plt.pie(ct["count"].values, labels=labels, autopct="%.1f%%", startangle=90)
    plt.title(f"Class Pie — {name}")
    _savefig(FIG / f"{name}_class_pie.png")


def missing_value_fraction(df: pd.DataFrame, name: str):
    mv = df.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    mv_df = mv.rename_axis("column").reset_index()
    _savetab(mv_df, TAB / f"{name}_missing_fraction.csv")

    plt.figure(figsize=(min(12, 0.35 * len(mv_df) + 4), 3.8))
    mv.plot(kind="bar")
    plt.ylabel("Missing Fraction")
    plt.title(f"Missing Value Fraction per Column — {name}")
    _savefig(FIG / f"{name}_missing_bar.png")


def numeric_distributions(df: pd.DataFrame, numeric_cols: list[str], name: str):
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if len(numeric_cols) == 0:
        return
    cols = min(4, len(numeric_cols))
    rows = int(np.ceil(len(numeric_cols) / cols))

    # hist
    plt.figure(figsize=(4 * cols, 2.8 * rows))
    for i, c in enumerate(numeric_cols, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[c].dropna(), kde=True, bins=30)
        plt.title(c)
    plt.suptitle(f"Histograms — {name}", y=1.02)
    _savefig(FIG / f"{name}_hists.png")

    # box
    plt.figure(figsize=(4 * cols, 2.8 * rows))
    for i, c in enumerate(numeric_cols, 1):
        plt.subplot(rows, cols, i)
        sns.boxplot(x=df[c], orient="h")
        plt.title(c)
    plt.suptitle(f"Boxplots — {name}", y=1.02)
    _savefig(FIG / f"{name}_boxplots.png")


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], name: str):
    use = [c for c in numeric_cols if c in df.columns]
    if len(use) < 2:
        return
    corr = df[use].corr(numeric_only=True)
    corr.to_csv(TAB / f"{name}_corr.csv", index=True)
    plt.figure(figsize=(0.8 * len(use) + 3, 0.8 * len(use) + 3))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", linewidths=0.3)
    plt.title(f"Correlation Heatmap — {name}")
    _savefig(FIG / f"{name}_corr_heatmap.png")


def pairplot_sample(df: pd.DataFrame, cols: list[str], hue: str, name: str, max_rows: int = 800):
    """
    Robust pairplot:
    - ambil hanya kolom numeric (kecuali hue)
    - hapus duplikat nama kolom
    - sample jika baris > max_rows
    - skip jika <2 kolom numeric valid
    """
    use_cols = [c for c in cols if c in df.columns]
    if hue in use_cols:
        use_cols.remove(hue)
    # hanya numeric
    num_cols = [c for c in use_cols if pd.api.types.is_numeric_dtype(df[c])]
    # pastikan unik
    num_cols = list(dict.fromkeys(num_cols))
    if hue not in df.columns or len(num_cols) < 2:
        return

    sub = df[num_cols + [hue]].dropna()
    if len(sub) > max_rows:
        sub = sub.sample(max_rows, random_state=42)

    # guard tambahan: tak boleh ada kolom 2D/object aneh
    for c in num_cols:
        if sub[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
            num_cols.remove(c)
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


def temporal_plots(df: pd.DataFrame, name: str, target: str = "tsu"):
    """
    Plot jumlah event per tahun (khusus tsu==1 jika target tersedia).
    Jika hanya ada 'year', gunakan itu; jika ada month/day akan dipakai untuk validasi.
    """
    if "year" not in df.columns:
        return
    dfx = df.copy()
    if target in dfx.columns:
        dfx = dfx[dfx[target] == 1]

    # jaga-jaga: pastikan integer year
    yr = pd.to_numeric(dfx["year"], errors="coerce").dropna().astype(int)
    if yr.empty:
        return
    yearly = yr.value_counts().sort_index()

    year_df = yearly.rename("count").reset_index().rename(columns={"index": "year"})
    _savetab(year_df, TAB / f"{name}_events_by_year.csv")

    mean, std = yearly.mean(), yearly.std()
    spike = yearly[yearly > mean + 2 * std]
    _savetab(spike.rename("count").reset_index().rename(columns={"index": "spike_year"}),
             TAB / f"{name}_spike_years.csv")

    plt.figure(figsize=(9, 4))
    plt.plot(yearly.index, yearly.values, marker="o")
    for y in spike.index:
        plt.axvline(y, color="red", ls="--", alpha=0.5)
    plt.title(f"Events per Year (tsu=1) — {name}")
    plt.xlabel("Year")
    plt.ylabel("Count")
    _savefig(FIG / f"{name}_by_year.png")


def spatial_scatter(df: pd.DataFrame, name: str, target: str = "tsu"):
    """Peta global: coba PyGMT tilemap; jika gagal, fallback scatter lon/lat."""
    if not {"latitude", "longitude"}.issubset(df.columns):
        return

    # --- fallback scatter globe ---
    def _fallback():
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

    try:
        import pygmt  # type: ignore
        fig = pygmt.Figure()
        fig.tilemap(
            region=[-180, 180, -75, 75],
            projection="M120/0/20c",
            source="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}.png",
            frame=["xafg", "yafg", f"+t{name.capitalize()} Events: Global Distribution"],
        )
        if target in df.columns:
            for val in sorted(pd.Series(df[target]).dropna().unique()):
                sub = df[df[target] == val]
                fig.plot(
                    x=sub["longitude"],
                    y=sub["latitude"],
                    style="c0.08c",
                    fill=("red" if int(val) == 1 else "dodgerblue"),
                    pen="black",
                    label=f"tsu={int(val)}",
                )
            fig.legend(position="JTR+jTR+o0.3c", box=True)
        else:
            fig.plot(x=df["longitude"], y=df["latitude"], style="c0.08c", fill="black", pen="black")
        out = FIG / f"{name}_global_map.png"
        fig.savefig(str(out))
    except Exception:
        _fallback()


def categorical_and_country_plots(df_t: pd.DataFrame, df_v: pd.DataFrame):
    # Tektonik: top 8 region
    if "region" in df_t.columns:
        top = df_t["region"].value_counts().nlargest(8)
        _savetab(top.rename_axis("region").reset_index(name="count"), TAB / "tectonic_top_regions.csv")
        plt.figure(figsize=(8, 4))
        sns.countplot(y="region", data=df_t[df_t["region"].isin(top.index)], order=top.index)
        plt.title("Tectonic: Top 8 Region")
        plt.xlabel("Number of Earthquake Events")
        _savefig(FIG / "tectonic_top_regions.png")

    # Tektonik: top 10 countries tsunami
    if {"country", "tsu"}.issubset(df_t.columns):
        s = df_t[df_t["tsu"] == 1]["country"].value_counts().nlargest(10)
        _savetab(s.rename_axis("country").reset_index(name="count"), TAB / "tectonic_top_countries_tsu.csv")
        plt.figure(figsize=(9, 4.2))
        sns.barplot(x=s.values, y=s.index)
        plt.title("Tectonic: Top 10 Countries with Tsunami (tsu=1)")
        plt.xlabel("Number of Tsunamis")
        plt.ylabel("Country")
        _savefig(FIG / "tectonic_top_countries_tsu.png")

    # Tektonik: mean magnitude by country
    if {"country", "mag"}.issubset(df_t.columns):
        mean_mag = df_t.groupby("country")["mag"].mean().sort_values(ascending=False).head(10)
        _savetab(mean_mag.rename("mean_mag").reset_index(), TAB / "tectonic_mean_mag_by_country.csv")
        plt.figure(figsize=(9, 4))
        sns.barplot(x=mean_mag.values, y=mean_mag.index)
        plt.title("Tectonic: Top 10 Average Magnitude per Country")
        plt.xlabel("Average Magnitude")
        _savefig(FIG / "tectonic_mean_mag_by_country.png")

    # Vulkanik: top 8 type
    if "type" in df_v.columns:
        top = df_v["type"].value_counts().nlargest(8)
        _savetab(top.rename_axis("type").reset_index(name="count"), TAB / "volcanic_top_types.csv")
        plt.figure(figsize=(8, 4))
        sns.countplot(y="type", data=df_v[df_v["type"].isin(top.index)], order=top.index)
        plt.title("Volcanic: Top 8 Volcano Type")
        plt.xlabel("Number of eruption events")
        _savefig(FIG / "volcanic_top_types.png")

    # Vulkanik: top 10 countries tsunami
    if {"country", "tsu"}.issubset(df_v.columns):
        s = df_v[df_v["tsu"] == 1]["country"].value_counts().nlargest(10)
        _savetab(s.rename_axis("country").reset_index(name="count"), TAB / "volcanic_top_countries_tsu.csv")
        plt.figure(figsize=(9, 4.2))
        sns.barplot(x=s.values, y=s.index)
        plt.title("Volcanic: Top 10 Countries with Tsunami (tsu=1)")
        plt.xlabel("Number of Tsunamis")
        plt.ylabel("Country")
        _savefig(FIG / "volcanic_top_countries_tsu.png")

    # Vulkanik: mean VEI by country
    if {"country", "vei"}.issubset(df_v.columns):
        mean_vei = df_v.groupby("country")["vei"].mean().sort_values(ascending=False).head(10)
        _savetab(mean_vei.rename("mean_vei").reset_index(), TAB / "volcanic_mean_vei_by_country.csv")
        plt.figure(figsize=(9, 4))
        sns.barplot(x=mean_vei.values, y=mean_vei.index)
        plt.title("Top 10 Average VEI per Country")
        plt.xlabel("Average VEI")
        _savefig(FIG / "volcanic_mean_vei_by_country.png")


# ---------- MASTER RUN ----------
def run_full_eda(df: pd.DataFrame, ds_name: str, target: str = "tsu"):
    # simpan basic info
    (TAB / f"{ds_name}_info.txt").write_text(_info_to_txt(df), encoding="utf-8")
    _savetab(df.head(20), TAB / f"{ds_name}_head20.csv")
    _savetab(df.describe(include="all").T.reset_index().rename(columns={"index": "feature"}),
             TAB / f"{ds_name}_describe.csv")

    # normalisasi nama tanggal kalau masih mo/dy
    if "mo" in df.columns or "dy" in df.columns:
        df = df.rename(columns={"mo": "month", "dy": "day"})

    # list kolom numerik (kecuali target)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]

    class_distribution(df, target, ds_name)
    missing_value_fraction(df, ds_name)
    numeric_distributions(df, numeric_cols, ds_name)
    correlation_heatmap(df, numeric_cols, ds_name)
    temporal_plots(df, ds_name, target=target)
    spatial_scatter(df, ds_name, target=target)

    # Pairplots dataset-spesifik
    if ds_name == "tectonic":
        pp_cols = ["mag", "depth", "latitude", "longitude", target]
    else:
        pp_cols = ["vei", "elevation", "latitude", "longitude", target]
    pairplot_sample(df, pp_cols, hue=target, name=ds_name)


def main(datasets: list[str]):
    _ensure_dirs()

    # baca processed
    dfs = {}
    for ds in datasets:
        df = _load_dataset(ds)
        if "tsu" not in df.columns:
            raise KeyError(f"'{ds}' tidak memiliki kolom target 'tsu' setelah preprocessing.")
        dfs[ds] = df
        print(f"[EDA] start {ds} -> rows={len(df)} cols={df.shape[1]}")

    # EDA per dataset
    for ds, df in dfs.items():
        run_full_eda(df, ds, target="tsu")

    # Plot kategori & negara lintas dataset
    if "tectonic" in dfs and "volcanic" in dfs:
        categorical_and_country_plots(dfs["tectonic"], dfs["volcanic"])

    print(f"[DONE] EDA saved -> {FIG} and {TAB}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EDA for tectonic/volcanic")
    ap.add_argument("--datasets", nargs="+", default=["tectonic", "volcanic"],
                    help="list dataset names: tectonic volcanic")
    args = ap.parse_args()
    main(args.datasets)