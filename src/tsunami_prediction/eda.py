from __future__ import annotations
# cSpell:ignore whitegrid savetab

import argparse
from pathlib import Path
import io
import warnings

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


# ---------- EVENTS LABEL HELPER ----------
def _ensure_events_multiclass_label(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Khusus dataset 'events':
    - Jika belum ada kolom 'label' / 'tsunami_label'
    - Dan tersedia 'tsu' (0/1) + 'source_domain' (tectonic/volcanic)
      maka dibentuk kolom 'label' multiclass:
        0 = non-tsunami
        1 = tsunami tektonik
        2 = tsunami vulkanik
    """
    if name != "events":
        return df

    # kalau sudah ada, tidak usah diapa-apakan
    if "label" in df.columns or "tsunami_label" in df.columns:
        return df

    if {"tsu", "source_domain"}.issubset(df.columns):
        df = df.copy()

        def _make_label(row: pd.Series) -> int:
            try:
                tsu = int(row["tsu"])
            except Exception:
                tsu = 0
            if tsu == 0:
                return 0
            dom = str(row["source_domain"]).lower()
            if "vol" in dom:      # volcanic / volcano
                return 2
            else:                 # default: tectonic
                return 1

        df["label"] = df.apply(_make_label, axis=1).astype("int64")
        print("[EDA] built multiclass 'label' from tsu + source_domain for events")
    return df


def _load_dataset(name: str) -> pd.DataFrame:
    """
    Ambil dataset hasil preprocessing/cleaning (BUKAN data raw).

    Urutan prioritas:
    1) data/processed/<name>_fe.csv         (events_fe, dsb.)
    2) data/processed/<name>.csv
    3) data/processed/<name>_cleaned.csv
    4) data/processed/<name>_biner.csv
    5) data/processed/<name>_preprocessed.csv
    """
    candidates = [
        PROCESSED / f"{name}_fe.csv",
        PROCESSED / f"{name}.csv",
        PROCESSED / f"{name}_cleaned.csv",
        PROCESSED / f"{name}_biner.csv",
        PROCESSED / f"{name}_preprocessed.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df = df.loc[:, ~df.columns.duplicated()].copy()

            # KHUSUS events: pastikan punya label multiclass
            df = _ensure_events_multiclass_label(df, name)

            print(f"[EDA] load {name} from {p} -> rows={len(df)} cols={df.shape[1]}")
            return df
    raise FileNotFoundError(
        f"[EDA] Processed dataset not found for '{name}'. Tried: {candidates}"
    )


# ---------- EDA PRIMITIVES ----------
def class_distribution(
    df: pd.DataFrame,
    target: str,
    name: str,
    label_map: dict[int, str] | None = None,
) -> None:
    """
    Menunjukkan distribusi kelas target (bisa biner atau multi-kelas).
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

    if label_map is not None:
        try:
            ct["label_name"] = ct[target].astype(int).map(label_map)
        except Exception:
            pass

    _savetab(ct, TAB / f"{name}_class_counts_{target}.csv")

    plt.figure(figsize=(4.6, 3.4))
    ax = sns.countplot(x=target, data=df)
    if label_map is not None:
        try:
            xticks = sorted(df[target].dropna().unique())
            labels = [label_map.get(int(v), str(v)) for v in xticks]
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(labels, rotation=15, ha="right")
        except Exception:
            pass
    ax.bar_label(ax.containers[0])
    plt.title(f"Class Distribution — {name} ({target})")
    _savefig(FIG / f"{name}_class_bar_{target}.png")


def missing_value_fraction(df: pd.DataFrame, name: str) -> None:
    """
    Menggambarkan kualitas data melalui proporsi missing value per kolom.
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
    Bentuk distribusi fitur numerik (skewed, normal, multimodal).
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
    (Opsional) Korelasi antar fitur numerik saja.
    Tidak dipanggil di run_full_eda supaya tidak duplikat.
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
    Korelasi linear antara fitur numerik dan target (bisa tsu, label, dsb).
    """
    cols = [c for c in numeric_cols if c in df.columns]
    if target in df.columns and target not in cols:
        cols.append(target)
    if len(cols) < 2:
        return

    corr = df[cols].corr(numeric_only=True)

    # buang baris/kolom yang 100% NaN
    corr = corr.dropna(axis=0, how="all")
    corr = corr.dropna(axis=1, how="all")

    if corr.empty:
        return

    time_cols = ["year", "month", "day", "hr", "mn", "sec"]
    spatial_cols = ["latitude", "longitude"]
    intensity_cols = [
        "depth",
        "mag",
        "mmi_int",
        "elevation",
        "vei",
        "eq",
        "distance_to_coast_km",
        "dist_coast_log1p",
    ]

    ordered: list[str] = []
    for group in (time_cols, spatial_cols, intensity_cols):
        ordered.extend([c for c in group if c in corr.columns])

    if target in corr.columns:
        ordered.append(target)

    for c in corr.columns:
        if c not in ordered:
            ordered.append(c)

    corr = corr.loc[ordered, ordered]

    corr.to_csv(TAB / f"{name}_corr_with_{target}.csv", index=True)

    plt.figure(figsize=(0.8 * len(corr.columns) + 3, 0.8 * len(corr.columns) + 3))
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
    _savefig(FIG / f"{name}_corr_with_{target}.png")


def pairplot_sample(
    df: pd.DataFrame,
    cols: list[str],
    hue: str,
    name: str,
    max_rows: int = 800,
) -> None:
    """
    Pairwise scatter antar fitur numerik + warna berdasarkan target.
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

    num_cols = [
        c
        for c in num_cols
        if not sub[c].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()
    ]
    if len(num_cols) < 2:
        return

    # suppress warning "Ignoring `palette` because no `hue` variable has been assigned."
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Ignoring `palette` because no `hue` variable has been assigned.",
        )
        g = sns.pairplot(
            sub,
            vars=num_cols,
            hue=hue,
            palette="Set1",
            plot_kws={"alpha": 0.7, "s": 25},
            height=2.4,
        )

    g.fig.suptitle(f"{name.capitalize()}: Pairplot (sample)", y=1.03)
    _savefig(FIG / f"{name}_pairplot_{hue}.png")


def numeric_vs_target_distributions(
    df: pd.DataFrame,
    cols: list[str],
    target: str,
    name: str,
    outfile: str | None = None,
) -> None:
    """
    Distribusi fitur numerik, dipisah per kelas target (biner atau multi-kelas).
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
    positive_value: int | None = 1,
) -> None:
    """
    Tren jumlah kejadian per tahun.
    Jika positive_value diberikan (misal tsu=1), hanya menghitung baris tsu=1.
    """
    if "year" not in df.columns:
        return

    dfx = df.copy()
    if target in dfx.columns and positive_value is not None:
        dfx = dfx[dfx[target] == positive_value]

    yr = pd.to_numeric(dfx["year"], errors="coerce").dropna().astype(int)
    if yr.empty:
        return

    yearly = yr.value_counts().sort_index()
    year_df = yearly.rename("count").reset_index().rename(columns={"index": "year"})
    _savetab(year_df, TAB / f"{name}_events_by_year_{target}.csv")

    mean = float(yearly.mean())
    std = float(yearly.std())
    spike = yearly[yearly > mean + 2 * std]
    _savetab(
        spike.rename("count").reset_index().rename(columns={"index": "spike_year"}),
        TAB / f"{name}_spike_years_{target}.csv",
    )

    plt.figure(figsize=(9, 4))
    plt.plot(yearly.index, yearly.values, marker="o")
    for y in spike.index:
        plt.axvline(y, color="red", ls="--", alpha=0.5)
    if positive_value is None:
        title = f"Events per Year — {name} ({target})"
    else:
        title = f"Events per Year ({target}={positive_value}) — {name}"
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Count")
    _savefig(FIG / f"{name}_by_year_{target}.png")


def spatial_scatter(df: pd.DataFrame, name: str, target: str = "tsu") -> None:
    """
    Sebaran spasial (lon/lat) dengan warna berdasarkan target.
    Fallback: scatter global sederhana; jika env `EDA_USE_PYGMT=1`,
    akan mencoba gunakan PyGMT + tile ocean map.
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

        if name == "tectonic":
            title = "Tectonic Events: Global Distribution"
        elif name == "volcanic":
            title = "Volcanic Events: Global Distribution"
        else:
            title = f"{name.capitalize()} Events: Global Distribution"

        plt.title(title)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        _savefig(FIG / f"{name}_global_scatter_{target}.png")

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
                frame=[
                    "xafg",
                    "yafg",
                    f"+t{name.capitalize()} Events: Global Distribution",
                ],
            )
            if target in df.columns:
                for val in sorted(pd.Series(df[target]).dropna().unique()):
                    sub = df[df[target] == val]
                    fig.plot(
                        x=sub["longitude"],
                        y=sub["latitude"],
                        style="c0.08c",
                        fill="red",
                        pen="black",
                        label=f"{target}={val}",
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
            fig.savefig(str(FIG / f"{name}_global_map_{target}.png"))
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
    Kategori teratas (region, type, country, dsb) berdasarkan frekuensi.
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
    Rata-rata nilai numerik (misal magnitudo / VEI) per negara (top 10).
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
    Visual kategori untuk narasi domain terpisah:
    - Tektonik: region, negara tsunami, rata-rata magnitudo.
    - Vulkanik: type, negara tsunami, rata-rata VEI.
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
    dari kolom 'time' atau 'origin_time'. (Saat ini jarang dipakai).
    """
    out = df.copy()

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
    Visualisasi fitur hasil rekayasa (mis. distance_to_coast_km), jika ada.
    """
    fe_candidates = [
        "distance_to_coast_km",
        "dist_coast_log1p",
        # tambahkan di sini kalau ada fitur rekayasa lain
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

    cols = fe_cols + ([target] if target in df.columns else [])
    corr = df[cols].corr(numeric_only=True)
    _savetab(
        corr.reset_index().rename(columns={"index": "feature"}),
        TAB / f"{name}_fe_corr_with_{target}.csv",
    )

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
    _savefig(FIG / f"{name}_fe_corr_with_{target}.png")


# ---------- EVENTS-SPECIFIC HELPERS ----------
def events_source_domain_plots(df: pd.DataFrame, target: str = "label") -> None:
    """
    Analisis gabungan seismik+vulkanik pada dataset events:
    - distribusi sumber (tectonic vs volcanic)
    - crosstab source_domain x label
    """
    name = "events"

    if "source_domain" not in df.columns:
        return

    # distribusi domain
    src_counts = df["source_domain"].value_counts()
    _plot_top_counts(
        src_counts,
        title="Events: Source Domain Distribution",
        xlabel="Number of Events",
        outfile="events_source_domain_counts.png",
        orient="h",
        table_name="events_source_domain_counts.csv",
    )

    # crosstab domain x label
    if target in df.columns:
        ct = pd.crosstab(df["source_domain"], df[target])
        ct_reset = ct.reset_index().rename_axis(None, axis=1)
        _savetab(ct_reset, TAB / "events_source_domain_by_label.csv")

        plt.figure(figsize=(7, 4))
        ct.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.xlabel("Source Domain")
        plt.ylabel("Number of Events")
        plt.title("Events: Label Distribution per Source Domain")
        plt.legend(title=target)
        _savefig(FIG / "events_source_domain_by_label.png")


# ---------- MASTER RUN PER DOMAIN (TECTONIC/VOLCANIC LEGACY) ----------
def run_full_eda(df: pd.DataFrame, ds_name: str, target: str = "tsu") -> None:
    """
    Pipeline EDA lengkap per domain (tectonic / volcanic) untuk narasi legacy.
    """
    (TAB / f"{ds_name}_info.txt").write_text(_info_to_txt(df), encoding="utf-8")
    _savetab(df.head(20), TAB / f"{ds_name}_head20.csv")
    _savetab(
        df.describe(include="all").T.reset_index().rename(columns={"index": "feature"}),
        TAB / f"{ds_name}_describe.csv",
    )

    if "mo" in df.columns or "dy" in df.columns:
        df = df.rename(columns={"mo": "month", "dy": "day"})

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {target, "id"}
    ]

    class_distribution(df, target, ds_name)
    missing_value_fraction(df, ds_name)
    numeric_distributions(df, numeric_cols, ds_name)
    correlation_with_target(df, numeric_cols, target, ds_name)

    temporal_plots(df, ds_name, target=target, positive_value=1)
    spatial_scatter(df, ds_name, target=target)

    if ds_name == "tectonic":
        numeric_vs_target_distributions(
            df,
            cols=[
                "mag",
                "depth",
                "latitude",
                "longitude",
                "distance_to_coast_km",
                "dist_coast_log1p",
            ],
            target=target,
            name=ds_name,
            outfile="tectonic_num_vs_tsu.png",
        )
        pp_cols = [
            "mag",
            "depth",
            "latitude",
            "longitude",
            "distance_to_coast_km",
            target,
        ]
    else:
        numeric_vs_target_distributions(
            df,
            cols=[
                "eq",
                "elevation",
                "latitude",
                "longitude",
                "distance_to_coast_km",
                "dist_coast_log1p",
            ],
            target=target,
            name=ds_name,
            outfile="volcanic_num_vs_tsu.png",
        )
        pp_cols = [
            "vei",
            "elevation",
            "latitude",
            "longitude",
            "distance_to_coast_km",
            target,
        ]

    pairplot_sample(df, pp_cols, hue=target, name=ds_name)
    engineered_feature_plots(df, ds_name, target=target)


# ---------- MASTER RUN UNTUK EVENTS (GABUNGAN SEISMIK+VULKANIK) ----------
def run_events_eda(df: pd.DataFrame, ds_name: str = "events") -> None:
    """
    EDA khusus dataset gabungan events:
    - target multi-kelas 'label' (0=non-tsunami,1=tektonik,2=vulkanik)
    - optional target biner 'tsu' (0/1)
    - analisis sumber domain (source_domain)
    """
    (TAB / f"{ds_name}_info.txt").write_text(_info_to_txt(df), encoding="utf-8")
    _savetab(df.head(20), TAB / f"{ds_name}_head20.csv")
    _savetab(
        df.describe(include="all").T.reset_index().rename(columns={"index": "feature"}),
        TAB / f"{ds_name}_describe.csv",
    )

    if "mo" in df.columns or "dy" in df.columns:
        df = df.rename(columns={"mo": "month", "dy": "day"})

    # target utama: multiclass label
    if "label" in df.columns:
        target_label = "label"
    elif "tsunami_label" in df.columns:
        target_label = "tsunami_label"
    else:
        raise KeyError(
            "'events' tidak memiliki kolom 'label' atau 'tsunami_label' "
            "setelah preprocessing."
        )

    target_binary: str | None = "tsu" if "tsu" in df.columns else None

    exclude_targets = {"id", target_label}
    if target_binary is not None:
        exclude_targets.add(target_binary)

    numeric_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_targets
    ]

    # 1) Distribusi kelas multi-kelas
    label_map = {0: "0 Non-tsunami", 1: "1 Tectonic", 2: "2 Volcanic"}
    class_distribution(df, target_label, ds_name, label_map=label_map)

    # 2) Bila ada target tsu biner, tampilkan juga
    if target_binary is not None:
        class_distribution(df, target_binary, ds_name)

    # 3) Kualitas data
    missing_value_fraction(df, ds_name)

    # 4) Distribusi numerik
    numeric_distributions(df, numeric_cols, ds_name)

    # 5) Korelasi dengan target multi-kelas
    correlation_with_target(df, numeric_cols, target_label, ds_name)

    # 6) Temporal:
    #    - events tsunami (tsu=1) kalau ada tsu
    #    - fallback: semua events berdasarkan year
    if target_binary is not None:
        temporal_plots(df, ds_name, target=target_binary, positive_value=1)
    else:
        temporal_plots(df, ds_name, target=target_label, positive_value=None)

    # 7) Spasial: warna berdasarkan label 0/1/2
    spatial_scatter(df, ds_name, target=target_label)

    # 8) Distribusi numerik utama vs label (gabungan seismik+vulkanik)
    core_num_cols = [
        "mag",
        "depth",
        "elevation",
        "vei",
        "latitude",
        "longitude",
        "abs_lat",
        "distance_to_coast_km",
        "dist_coast_log1p",
    ]
    numeric_vs_target_distributions(
        df,
        cols=core_num_cols,
        target=target_label,
        name=ds_name,
        outfile=f"{ds_name}_num_vs_{target_label}.png",
    )

    # 9) Pairplot sample
    pp_cols = [
        "mag",
        "depth",
        "elevation",
        "vei",
        "latitude",
        "longitude",
        "distance_to_coast_km",
        target_label,
    ]
    pairplot_sample(df, pp_cols, hue=target_label, name=ds_name)

    # 10) Fitur rekayasa (jika ada), gunakan target tsu kalau tersedia
    fe_target = target_binary if target_binary is not None else target_label
    engineered_feature_plots(df, ds_name, target=fe_target)

    # 11) Analisis sumber domain (tectonic vs volcanic)
    events_source_domain_plots(df, target=target_label)


# ---------- MAIN ----------
def main(datasets: list[str]) -> None:
    _ensure_dirs()

    dfs: list[tuple[str, pd.DataFrame]] = []

    for ds in datasets:
        df = _load_dataset(ds)

        # Validasi target sesuai jenis dataset
        if ds == "events":
            # Pastikan minimal punya salah satu label
            if not (
                ("label" in df.columns)
                or ("tsunami_label" in df.columns)
                or ("tsu" in df.columns)
            ):
                raise KeyError(
                    "'events' tidak memiliki kolom 'label', 'tsunami_label' "
                    "maupun 'tsu' setelah preprocessing."
                )
        else:
            if "tsu" not in df.columns:
                raise KeyError(
                    f"'{ds}' tidak memiliki kolom target 'tsu' setelah preprocessing."
                )

        dfs.append((ds, df))
        print(f"[EDA] start {ds} -> rows={len(df)} cols={df.shape[1]}")

    # Jalankan EDA per dataset
    for ds, df in dfs:
        if ds == "events":
            run_events_eda(df, ds_name="events")
        else:
            # legacy tectonic / volcanic
            run_full_eda(df, ds_name=ds, target="tsu")

    # Visual lintas domain hanya kalau tectonic+volcanic ikut diminta
    names = {ds for ds, _ in dfs}
    if "tectonic" in names and "volcanic" in names:
        tect_df = next(df for ds, df in dfs if ds == "tectonic")
        volc_df = next(df for ds, df in dfs if ds == "volcanic")
        categorical_and_country_plots(tect_df, volc_df)

    print(f"[DONE] EDA saved -> {FIG} and {TAB}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EDA untuk dataset gabungan events atau domain terpisah."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["events"],
        help=(
            "list dataset names, misal: events "
            "atau tectonic volcanic (legacy)"
        ),
    )
    args = parser.parse_args()
    main(args.datasets)