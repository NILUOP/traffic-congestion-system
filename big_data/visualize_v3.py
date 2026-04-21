"""
visualize.py — Comprehensive matplotlib visualization for CARLA traffic data.

Generates 6 figure files covering every angle of the dataset:
  Fig 1  — Dataset overview       (8 panels)
  Fig 2  — Speed deep dive        (8 panels)
  Fig 3  — Spatial analysis       (8 panels)
  Fig 4  — Weather & environment  (8 panels)
  Fig 5  — Congestion patterns    (8 panels)
  Fig 6  — Cross-feature analysis (8 panels)

USAGE:
    python visualize.py

OUTPUT:
    data/visualizations/fig_01_overview.png
    data/visualizations/fig_02_speed.png
    data/visualizations/fig_03_spatial.png
    data/visualizations/fig_04_weather.png
    data/visualizations/fig_05_congestion.png
    data/visualizations/fig_06_crossfeature.png
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter

# =============================================================================
# CONFIG
# =============================================================================

RAW_DIR    = "data/raw"
OUT_DIR    = "data/visualizations"
DPI        = 150
SAMPLE_N   = 80_000   # rows to sample for heavy scatter plots

# Palette
BG        = "#0d1117"
PANEL_BG  = "#161b22"
GRID_COL  = "#21262d"
TEXT_COL  = "#e6edf3"
MUTED_COL = "#8b949e"

C_LOW  = "#3fb950"   # green
C_MED  = "#d29922"   # amber
C_HIGH = "#f85149"   # red
C_BLUE = "#388bfd"
C_PURP = "#bc8cff"
C_CYAN = "#39d353"

LABEL_COLORS = {"Low": C_LOW, "Medium": C_MED, "High": C_HIGH}
TOWN_COLORS  = {"Town01": C_BLUE, "Town02": C_PURP,
                "Town03": "#ff7b72", "Town04": "#ffa657"}
WEATHER_COLORS = {
    "ClearNoon":     "#ffd700",
    "HardRainNoon":  "#388bfd",
    "MidRainyNoon":  "#79c0ff",
    "WetCloudyNoon": "#8b949e",
}

# =============================================================================
# STYLE SETUP
# =============================================================================

plt.rcParams.update({
    "figure.facecolor":     BG,
    "axes.facecolor":       PANEL_BG,
    "axes.edgecolor":       GRID_COL,
    "axes.labelcolor":      TEXT_COL,
    "axes.titlecolor":      TEXT_COL,
    "axes.titlesize":       11,
    "axes.labelsize":       9,
    "axes.titlepad":        10,
    "axes.grid":            True,
    "grid.color":           GRID_COL,
    "grid.linewidth":       0.5,
    "xtick.color":          MUTED_COL,
    "ytick.color":          MUTED_COL,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "text.color":           TEXT_COL,
    "legend.facecolor":     PANEL_BG,
    "legend.edgecolor":     GRID_COL,
    "legend.fontsize":      8,
    "figure.titlesize":     15,
    "figure.titleweight":   "bold",
    "savefig.facecolor":    BG,
    "savefig.bbox":         "tight",
    "savefig.dpi":          DPI,
    "font.family":          "monospace",
})

# =============================================================================
# HELPERS
# =============================================================================

def load_data(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {raw_dir}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} rows from {len(files)} files.")
    return df


def sample(df, n=SAMPLE_N, seed=42):
    return df.sample(min(n, len(df)), random_state=seed)


def fig_title(fig, title: str):
    fig.suptitle(title, color=TEXT_COL, fontsize=15, fontweight="bold",
                 y=0.98, fontfamily="monospace")


def subtitle(ax, txt):
    ax.set_title(txt, fontsize=10, color=TEXT_COL, pad=8)


def spine_off(ax, sides=("top", "right")):
    for s in sides:
        ax.spines[s].set_visible(False)


def legend_patches(color_map: dict):
    return [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]


def severity_score(row):
    """0*P(Low) + 50*P(Medium) + 100*P(High) approximated from label."""
    return {"Low": 10, "Medium": 50, "High": 90}[row["congestion_label"]]


# =============================================================================
# FIG 1 — DATASET OVERVIEW
# =============================================================================

def fig_overview(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(22, 18))
    fig_title(fig, "Fig 1  —  Dataset Overview")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 1. Rows per session (horizontal bar)
    ax = fig.add_subplot(gs[0, 0])
    sc = df.groupby("session_id").size().sort_values()
    short = [s[-22:] for s in sc.index]
    colors_bar = [TOWN_COLORS.get(s.split("_")[2], C_BLUE) for s in sc.index]
    ax.barh(short, sc.values, color=colors_bar, edgecolor="none", alpha=0.88)
    ax.set_xlabel("Row count")
    subtitle(ax, "Rows per session")
    spine_off(ax)

    # 2. Rows by town (pie)
    ax2 = fig.add_subplot(gs[0, 1])
    tc = df["town"].value_counts()
    wedge_colors = [TOWN_COLORS.get(t, C_BLUE) for t in tc.index]
    wedges, texts, autotexts = ax2.pie(
        tc.values, labels=tc.index, colors=wedge_colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor=BG, linewidth=1.5),
        textprops=dict(color=TEXT_COL, fontsize=8),
    )
    for at in autotexts:
        at.set_fontsize(7)
    subtitle(ax2, "Rows by town")

    # 3. Rows by weather (horizontal bar)
    ax3 = fig.add_subplot(gs[0, 2])
    wc = df["weather_name"].value_counts()
    wcolors = [WEATHER_COLORS.get(w, C_BLUE) for w in wc.index]
    ax3.barh(wc.index, wc.values, color=wcolors, edgecolor="none", alpha=0.88)
    ax3.set_xlabel("Row count")
    subtitle(ax3, "Rows by weather")
    spine_off(ax3)

    # 4. Congestion label donut
    ax4 = fig.add_subplot(gs[1, 0])
    lc = df["congestion_label"].value_counts().reindex(["Low", "Medium", "High"])
    wedges4, _, at4 = ax4.pie(
        lc.values,
        labels=lc.index,
        colors=[C_LOW, C_MED, C_HIGH],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=1.5),
        textprops=dict(color=TEXT_COL, fontsize=8),
    )
    for at in at4:
        at.set_fontsize(7)
    subtitle(ax4, "Congestion label distribution")

    # 5. Rows by town x weather (stacked bar)
    ax5 = fig.add_subplot(gs[1, 1:])
    tw = df.groupby(["town", "weather_name"]).size().unstack(fill_value=0)
    bottom = np.zeros(len(tw))
    wnames = tw.columns.tolist()
    w_bar_colors = [WEATHER_COLORS.get(w, C_BLUE) for w in wnames]
    for i, (wname, color) in enumerate(zip(wnames, w_bar_colors)):
        ax5.bar(tw.index, tw[wname], bottom=bottom, color=color,
                edgecolor=BG, linewidth=0.5, label=wname, alpha=0.9)
        bottom += tw[wname].values
    ax5.set_ylabel("Row count")
    ax5.legend(loc="upper right", ncol=2)
    ax5.set_xticklabels(tw.index, rotation=0)
    subtitle(ax5, "Rows by town × weather (stacked)")
    spine_off(ax5)

    # 6. Vehicle count distribution across ticks (per session violin)
    ax6 = fig.add_subplot(gs[2, 0])
    sess_veh = df.groupby(["session_id", "tick"])["vehicle_id"].nunique().reset_index()
    sess_veh["town"] = sess_veh["session_id"].str.extract(r"_(Town\d+)_")[0]
    towns_sorted = sorted(sess_veh["town"].unique())
    data_viol = [sess_veh[sess_veh["town"] == t]["vehicle_id"].values for t in towns_sorted]
    parts = ax6.violinplot(data_viol, positions=range(len(towns_sorted)),
                           showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(TOWN_COLORS.values())[i % 4])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color(TEXT_COL)
    parts["cmins"].set_color(MUTED_COL)
    parts["cmaxes"].set_color(MUTED_COL)
    parts["cbars"].set_color(MUTED_COL)
    ax6.set_xticks(range(len(towns_sorted)))
    ax6.set_xticklabels(towns_sorted, rotation=15)
    ax6.set_ylabel("Vehicles per tick")
    subtitle(ax6, "Vehicle count distribution per town")
    spine_off(ax6)

    # 7. Acceleration distribution (KDE per label)
    ax7 = fig.add_subplot(gs[2, 1])
    s = sample(df, 40000)
    for label, col in LABEL_COLORS.items():
        vals = s[s["congestion_label"] == label]["acceleration"].clip(0, 12)
        if len(vals) > 10:
            kde = stats.gaussian_kde(vals, bw_method=0.2)
            xr  = np.linspace(0, 12, 200)
            ax7.plot(xr, kde(xr), color=col, label=label, linewidth=1.8)
            ax7.fill_between(xr, kde(xr), alpha=0.12, color=col)
    ax7.set_xlabel("Acceleration (m/s²)")
    ax7.set_ylabel("Density")
    ax7.legend()
    subtitle(ax7, "Acceleration KDE by congestion level")
    spine_off(ax7)

    # 8. Cumulative row count over sessions (waterfall)
    ax8 = fig.add_subplot(gs[2, 2])
    sc2 = df.groupby("session_id").size().sort_index()
    cumulative = sc2.cumsum()
    short2 = [s[-18:] for s in cumulative.index]
    ax8.step(range(len(cumulative)), cumulative.values / 1e6,
             color=C_BLUE, linewidth=2, where="post")
    ax8.fill_between(range(len(cumulative)), cumulative.values / 1e6,
                     step="post", alpha=0.15, color=C_BLUE)
    ax8.set_xticks(range(len(cumulative)))
    ax8.set_xticklabels(short2, rotation=45, ha="right", fontsize=6)
    ax8.set_ylabel("Cumulative rows (M)")
    ax8.axhline(1.0, color=C_HIGH, linewidth=1, linestyle="--", alpha=0.6, label="1M target")
    ax8.legend()
    subtitle(ax8, "Cumulative rows across sessions")
    spine_off(ax8)

    path = os.path.join(out_dir, "fig_01_overview.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# FIG 2 — SPEED DEEP DIVE
# =============================================================================

def fig_speed(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(22, 18))
    fig_title(fig, "Fig 2  —  Speed Deep Dive")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    s = sample(df)

    # 1. Speed histogram all data
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(df["speed"].clip(0, 100), bins=100, color=C_BLUE,
            edgecolor="none", alpha=0.85)
    ax.axvline(df["speed"].mean(), color=C_HIGH, linewidth=1.5,
               linestyle="--", label=f"mean={df['speed'].mean():.1f}")
    ax.axvline(df["speed"].median(), color=C_LOW, linewidth=1.5,
               linestyle=":", label=f"median={df['speed'].median():.1f}")
    ax.set_xlabel("Speed (km/h)")
    ax.legend()
    subtitle(ax, "Speed histogram — full dataset")
    spine_off(ax)

    # 2. Speed KDE per congestion label
    ax2 = fig.add_subplot(gs[0, 1])
    for label, col in LABEL_COLORS.items():
        vals = s[s["congestion_label"] == label]["speed"].clip(0, 100)
        kde  = stats.gaussian_kde(vals, bw_method=0.15)
        xr   = np.linspace(0, 100, 300)
        ax2.plot(xr, kde(xr), color=col, label=label, linewidth=2)
        ax2.fill_between(xr, kde(xr), alpha=0.1, color=col)
    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Density")
    ax2.legend()
    subtitle(ax2, "Speed KDE by congestion level")
    spine_off(ax2)

    # 3. Speed box per town
    ax3 = fig.add_subplot(gs[0, 2])
    towns = sorted(df["town"].unique())
    bplot = ax3.boxplot(
        [df[df["town"] == t]["speed"].clip(0, 100).values for t in towns],
        patch_artist=True, notch=True,
        medianprops=dict(color=TEXT_COL, linewidth=2),
        flierprops=dict(marker=".", markersize=1, alpha=0.3,
                        markerfacecolor=MUTED_COL),
    )
    for patch, town in zip(bplot["boxes"], towns):
        patch.set_facecolor(TOWN_COLORS.get(town, C_BLUE))
        patch.set_alpha(0.7)
    ax3.set_xticklabels(towns, rotation=15)
    ax3.set_ylabel("Speed (km/h)")
    subtitle(ax3, "Speed distribution per town")
    spine_off(ax3)

    # 4. Speed vs acceleration scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sc_plot = s.sample(min(5000, len(s)), random_state=7)
    for label, col in LABEL_COLORS.items():
        g = sc_plot[sc_plot["congestion_label"] == label]
        ax4.scatter(g["speed"], g["acceleration"].clip(0, 10),
                    c=col, s=4, alpha=0.35, label=label)
    ax4.set_xlabel("Speed (km/h)")
    ax4.set_ylabel("Acceleration (m/s²)")
    ax4.legend(markerscale=3)
    subtitle(ax4, "Speed vs acceleration  (coloured by label)")
    spine_off(ax4)

    # 5. Average speed per tick (first 3 sessions)
    ax5 = fig.add_subplot(gs[1, 1:])
    sessions = df["session_id"].unique()[:3]
    for sess in sessions:
        ts = df[df["session_id"] == sess].groupby("tick")["speed"].mean()
        short = sess[-22:]
        ax5.plot(ts.index, ts.values, linewidth=1.2, alpha=0.85, label=short)
    ax5.set_xlabel("Tick")
    ax5.set_ylabel("Avg speed (km/h)")
    ax5.legend(fontsize=7)
    subtitle(ax5, "Average speed over time — first 3 sessions")
    spine_off(ax5)

    # 6. Speed CDF per label
    ax6 = fig.add_subplot(gs[2, 0])
    for label, col in LABEL_COLORS.items():
        vals = np.sort(s[s["congestion_label"] == label]["speed"].values)
        ax6.plot(vals, np.linspace(0, 1, len(vals)),
                 color=col, label=label, linewidth=1.8)
    ax6.set_xlabel("Speed (km/h)")
    ax6.set_ylabel("Cumulative probability")
    ax6.legend()
    ax6.yaxis.set_major_formatter(PercentFormatter(1))
    subtitle(ax6, "Speed CDF per congestion level")
    spine_off(ax6)

    # 7. Speed histogram per weather (overlaid)
    ax7 = fig.add_subplot(gs[2, 1])
    for wname, col in WEATHER_COLORS.items():
        vals = df[df["weather_name"] == wname]["speed"].clip(0, 100)
        if len(vals) > 0:
            ax7.hist(vals, bins=60, color=col, alpha=0.45,
                     label=wname, edgecolor="none", density=True)
    ax7.set_xlabel("Speed (km/h)")
    ax7.set_ylabel("Density")
    ax7.legend(fontsize=7)
    subtitle(ax7, "Speed distribution per weather")
    spine_off(ax7)

    # 8. Speed percentile heatmap (town × congestion)
    ax8 = fig.add_subplot(gs[2, 2])
    piv = df.groupby(["town", "congestion_label"])["speed"].median().unstack()
    piv = piv.reindex(columns=["Low", "Medium", "High"])
    im  = ax8.imshow(piv.values, aspect="auto",
                     cmap="RdYlGn", vmin=0, vmax=80)
    ax8.set_xticks(range(3))
    ax8.set_xticklabels(["Low", "Medium", "High"])
    ax8.set_yticks(range(len(piv.index)))
    ax8.set_yticklabels(piv.index)
    for i in range(len(piv.index)):
        for j in range(3):
            val = piv.values[i, j]
            if not np.isnan(val):
                ax8.text(j, i, f"{val:.0f}", ha="center", va="center",
                         fontsize=9, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax8, label="Median speed (km/h)")
    subtitle(ax8, "Median speed: town × congestion level")

    path = os.path.join(out_dir, "fig_02_speed.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# FIG 3 — SPATIAL ANALYSIS
# =============================================================================

def fig_spatial(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(22, 18))
    fig_title(fig, "Fig 3  —  Spatial Analysis")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    towns = sorted(df["town"].unique())

    # 1–4: Per-town 2D density heatmap
    for idx, town in enumerate(towns[:4]):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        tdf = df[df["town"] == town]
        h, xedge, yedge = np.histogram2d(
            tdf["x"], tdf["y"], bins=60
        )
        h_smooth = gaussian_filter(h.T, sigma=1.5)
        im = ax.imshow(
            h_smooth, origin="lower", aspect="auto",
            extent=[xedge[0], xedge[-1], yedge[0], yedge[-1]],
            cmap="plasma", interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="Count")
        subtitle(ax, f"Vehicle density heatmap — {town}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

    # 5. All towns overlaid — positions coloured by town
    ax5 = fig.add_subplot(gs[2, 0])
    s = sample(df, 20000)
    for town, col in TOWN_COLORS.items():
        g = s[s["town"] == town]
        ax5.scatter(g["x"], g["y"], c=col, s=1.5, alpha=0.3, label=town)
    ax5.legend(markerscale=4, fontsize=7)
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    subtitle(ax5, "All towns — vehicle positions")
    spine_off(ax5)

    # 6. Congestion label spatial scatter
    ax6 = fig.add_subplot(gs[2, 1])
    s2 = sample(df, 20000)
    for label, col in LABEL_COLORS.items():
        g = s2[s2["congestion_label"] == label]
        ax6.scatter(g["x"], g["y"], c=col, s=1.5, alpha=0.3, label=label)
    ax6.legend(markerscale=4, fontsize=7)
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")
    subtitle(ax6, "Congestion label — spatial distribution")
    spine_off(ax6)

    # 7. Speed spatial heatmap (avg speed per grid cell)
    ax7 = fig.add_subplot(gs[1, 2])
    s3 = sample(df, 40000)
    h_speed, xedge2, yedge2 = np.histogram2d(
        s3["x"], s3["y"], bins=60, weights=s3["speed"]
    )
    h_cnt, _, _ = np.histogram2d(s3["x"], s3["y"], bins=60)
    with np.errstate(invalid="ignore", divide="ignore"):
        h_avg = np.where(h_cnt > 0, h_speed / h_cnt, np.nan)
    im7 = ax7.imshow(
        h_avg.T, origin="lower", aspect="auto",
        extent=[xedge2[0], xedge2[-1], yedge2[0], yedge2[-1]],
        cmap="RdYlGn", vmin=0, vmax=80, interpolation="bilinear",
    )
    plt.colorbar(im7, ax=ax7, label="Avg speed (km/h)")
    ax7.set_xlabel("X (m)")
    ax7.set_ylabel("Y (m)")
    subtitle(ax7, "Average speed — spatial heatmap (all towns)")

    # 8. Yaw rose (polar histogram)
    ax8 = fig.add_subplot(gs[2, 2], projection="polar")
    ax8.set_facecolor(PANEL_BG)
    yaw_rad = np.deg2rad(sample(df, 30000)["yaw"].values)
    nbins   = 36
    counts, edges = np.histogram(yaw_rad, bins=nbins,
                                 range=(-np.pi, np.pi))
    width   = (2 * np.pi) / nbins
    bars    = ax8.bar(edges[:-1], counts, width=width, align="edge",
                      edgecolor=BG, linewidth=0.3)
    norm_c  = mcolors.Normalize(vmin=0, vmax=counts.max())
    cmap_r  = plt.cm.plasma
    for bar, cnt in zip(bars, counts):
        bar.set_facecolor(cmap_r(norm_c(cnt)))
        bar.set_alpha(0.85)
    ax8.set_theta_zero_location("N")
    ax8.set_theta_direction(-1)
    ax8.tick_params(colors=MUTED_COL)
    ax8.set_title("Vehicle heading distribution (yaw)",
                  color=TEXT_COL, fontsize=10, pad=15)

    path = os.path.join(out_dir, "fig_03_spatial.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# FIG 4 — WEATHER & ENVIRONMENT
# =============================================================================

def fig_weather(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(22, 18))
    fig_title(fig, "Fig 4  —  Weather & Environment Analysis")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    s = sample(df)

    # 1. Congestion label % per weather (stacked 100% bar)
    ax = fig.add_subplot(gs[0, 0:2])
    wl = (df.groupby(["weather_name", "congestion_label"])
            .size()
            .unstack(fill_value=0))
    wl_pct = wl.div(wl.sum(axis=1), axis=0) * 100
    wl_pct = wl_pct.reindex(columns=["Low", "Medium", "High"])
    bottom = np.zeros(len(wl_pct))
    for label, col in LABEL_COLORS.items():
        ax.bar(wl_pct.index, wl_pct[label], bottom=bottom,
               color=col, label=label, edgecolor=BG, linewidth=0.5)
        bottom += wl_pct[label].values
    ax.set_ylabel("% of records")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.set_xticklabels(wl_pct.index, rotation=0)
    subtitle(ax, "Congestion distribution per weather (100% stacked)")
    spine_off(ax)

    # 2. Average speed per weather (bar + error)
    ax2 = fig.add_subplot(gs[0, 2])
    wm = df.groupby("weather_name")["speed"].agg(["mean", "std"])
    colors_w = [WEATHER_COLORS.get(w, C_BLUE) for w in wm.index]
    ax2.bar(wm.index, wm["mean"], yerr=wm["std"], color=colors_w,
            edgecolor="none", alpha=0.85, capsize=4,
            error_kw=dict(ecolor=TEXT_COL, linewidth=1))
    ax2.set_ylabel("Speed (km/h)")
    ax2.set_xticklabels(wm.index, rotation=15, ha="right")
    subtitle(ax2, "Avg speed per weather (±1 std)")
    spine_off(ax2)

    # 3. Speed violin per weather
    ax3 = fig.add_subplot(gs[1, 0:2])
    weathers = sorted(df["weather_name"].unique())
    data_v   = [df[df["weather_name"] == w]["speed"].clip(0, 100).values
                for w in weathers]
    parts3   = ax3.violinplot(data_v, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts3["bodies"]):
        pc.set_facecolor(list(WEATHER_COLORS.values())[i % 4])
        pc.set_alpha(0.7)
    parts3["cmedians"].set_color(TEXT_COL)
    parts3["cmins"].set_color(MUTED_COL)
    parts3["cmaxes"].set_color(MUTED_COL)
    parts3["cbars"].set_color(MUTED_COL)
    ax3.set_xticks(range(1, len(weathers) + 1))
    ax3.set_xticklabels(weathers, rotation=15)
    ax3.set_ylabel("Speed (km/h)")
    subtitle(ax3, "Speed violin per weather condition")
    spine_off(ax3)

    # 4. Rainy vs clear: congestion %
    ax4 = fig.add_subplot(gs[1, 2])
    rainy_codes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20]
    df_copy = df.copy()
    df_copy["is_rainy"] = df_copy["weather"].isin(rainy_codes)
    rc = (df_copy.groupby(["is_rainy", "congestion_label"])
                 .size().unstack(fill_value=0)
                 .reindex(columns=["Low", "Medium", "High"]))
    rc_pct = rc.div(rc.sum(axis=1), axis=0) * 100
    x = np.arange(2)
    w = 0.25
    for i, (label, col) in enumerate(LABEL_COLORS.items()):
        ax4.bar(x + i * w, rc_pct[label].values, w,
                color=col, edgecolor=BG, label=label, alpha=0.88)
    ax4.set_xticks(x + w)
    ax4.set_xticklabels(["Clear", "Rainy"])
    ax4.set_ylabel("%")
    ax4.legend()
    subtitle(ax4, "Congestion: rainy vs clear conditions")
    spine_off(ax4)

    # 5. Acceleration KDE per weather
    ax5 = fig.add_subplot(gs[2, 0])
    for wname, col in WEATHER_COLORS.items():
        vals = s[s["weather_name"] == wname]["acceleration"].clip(0, 8)
        if len(vals) > 100:
            kde  = stats.gaussian_kde(vals, bw_method=0.25)
            xr   = np.linspace(0, 8, 200)
            ax5.plot(xr, kde(xr), color=col, label=wname, linewidth=1.8)
    ax5.set_xlabel("Acceleration (m/s²)")
    ax5.set_ylabel("Density")
    ax5.legend(fontsize=7)
    subtitle(ax5, "Acceleration KDE per weather")
    spine_off(ax5)

    # 6. High congestion % per town × weather heatmap
    ax6 = fig.add_subplot(gs[2, 1])
    tw_high = (df[df["congestion_label"] == "High"]
               .groupby(["town", "weather_name"]).size()
               .unstack(fill_value=0))
    tw_total = df.groupby(["town", "weather_name"]).size().unstack(fill_value=1)
    tw_pct = (tw_high / tw_total * 100).fillna(0)
    im6 = ax6.imshow(tw_pct.values, aspect="auto",
                     cmap="YlOrRd", vmin=0, vmax=50)
    ax6.set_xticks(range(tw_pct.shape[1]))
    ax6.set_xticklabels(tw_pct.columns, rotation=25, ha="right", fontsize=7)
    ax6.set_yticks(range(len(tw_pct.index)))
    ax6.set_yticklabels(tw_pct.index)
    for i in range(tw_pct.shape[0]):
        for j in range(tw_pct.shape[1]):
            ax6.text(j, i, f"{tw_pct.values[i,j]:.0f}%",
                     ha="center", va="center", fontsize=8,
                     color="black" if tw_pct.values[i,j] > 20 else TEXT_COL)
    plt.colorbar(im6, ax=ax6, label="% High congestion")
    subtitle(ax6, "% High congestion: town × weather")

    # 7. Speed scatter: clear vs rainy (hexbin)
    ax7 = fig.add_subplot(gs[2, 2])
    clear = df[~df["weather"].isin(rainy_codes)]["speed"].clip(0, 100)
    rainy = df[df["weather"].isin(rainy_codes)]["speed"].clip(0, 100)
    ax7.hist(clear, bins=60, color=C_MED, alpha=0.6,
             density=True, label="Clear", edgecolor="none")
    ax7.hist(rainy, bins=60, color=C_BLUE, alpha=0.6,
             density=True, label="Rainy", edgecolor="none")
    ax7.set_xlabel("Speed (km/h)")
    ax7.set_ylabel("Density")
    ax7.legend()
    subtitle(ax7, "Speed density: clear vs rainy")
    spine_off(ax7)

    path = os.path.join(out_dir, "fig_04_weather.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# FIG 5 — CONGESTION PATTERNS
# =============================================================================

def fig_congestion(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(22, 18))
    fig_title(fig, "Fig 5  —  Congestion Patterns")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    s = sample(df)

    # 1. Congestion label over ticks (session 1)
    ax = fig.add_subplot(gs[0, 0:2])
    sess1 = df["session_id"].unique()[0]
    ts = (df[df["session_id"] == sess1]
          .groupby("tick")["congestion_label"]
          .value_counts(normalize=True)
          .unstack(fill_value=0)
          .reindex(columns=["Low", "Medium", "High"])
          .rolling(5).mean())
    for label, col in LABEL_COLORS.items():
        if label in ts.columns:
            ax.plot(ts.index, ts[label] * 100, color=col,
                    label=label, linewidth=1.5)
            ax.fill_between(ts.index, ts[label] * 100, alpha=0.08, color=col)
    ax.set_xlabel("Tick")
    ax.set_ylabel("% of vehicles")
    ax.legend()
    ax.set_ylim(0, 100)
    subtitle(ax, f"Congestion label % over time — {sess1[-22:]}")
    spine_off(ax)

    # 2. Congestion label counts per town (grouped bar)
    ax2 = fig.add_subplot(gs[0, 2])
    tl = (df.groupby(["town", "congestion_label"])
            .size().unstack(fill_value=0)
            .reindex(columns=["Low", "Medium", "High"]))
    x  = np.arange(len(tl))
    w  = 0.25
    for i, (label, col) in enumerate(LABEL_COLORS.items()):
        ax2.bar(x + i * w, tl[label].values, w,
                color=col, edgecolor=BG, label=label, alpha=0.88)
    ax2.set_xticks(x + w)
    ax2.set_xticklabels(tl.index)
    ax2.set_ylabel("Row count")
    ax2.legend()
    subtitle(ax2, "Congestion label count per town")
    spine_off(ax2)

    # 3. Avg traffic density per label (bar)
    ax3 = fig.add_subplot(gs[1, 0])
    dm = df.groupby("congestion_label")["traffic_density"].agg(["mean", "std"])
    dm = dm.reindex(["Low", "Medium", "High"])
    ax3.bar(dm.index, dm["mean"], yerr=dm["std"],
            color=[C_LOW, C_MED, C_HIGH], edgecolor="none",
            capsize=5, error_kw=dict(ecolor=TEXT_COL))
    ax3.set_ylabel("Avg traffic density (vehicles/50m)")
    subtitle(ax3, "Avg density per congestion level")
    spine_off(ax3)

    # 4. Density histogram per label (overlaid)
    ax4 = fig.add_subplot(gs[1, 1])
    for label, col in LABEL_COLORS.items():
        vals = s[s["congestion_label"] == label]["traffic_density"]
        ax4.hist(vals, bins=30, color=col, alpha=0.5,
                 edgecolor="none", density=True, label=label)
    ax4.set_xlabel("Traffic density (vehicles/50m)")
    ax4.set_ylabel("Density")
    ax4.legend()
    subtitle(ax4, "Density histogram per label (overlaid)")
    spine_off(ax4)

    # 5. Severity score histogram
    ax5 = fig.add_subplot(gs[1, 2])
    df_sev = df.copy()
    df_sev["severity"] = df_sev.apply(severity_score, axis=1)
    # ax5.hist(df_sev["severity"], bins=[0, 25, 40, 60, 75, 100],
    #          color=[C_LOW, C_LOW, C_MED, C_HIGH, C_HIGH],
    #          edgecolor=BG, linewidth=0.8, alpha=0.88)
    # Change line 731 in your script:
    ax5.hist(df_sev["severity"], bins=[0, 25, 40, 60, 75, 100],
         color=C_MED,  # Just use one color here
         edgecolor=BG, linewidth=0.8, alpha=0.88)
    ax5.set_xlabel("Severity score (0–100)")
    ax5.set_ylabel("Count")
    ax5.set_xticks([10, 50, 90])
    ax5.set_xticklabels(["Low (10)", "Medium (50)", "High (90)"])
    subtitle(ax5, "Congestion severity score distribution")
    spine_off(ax5)

    # 6. Stationary vehicle % over ticks
    ax6 = fig.add_subplot(gs[2, 0:2])
    for sess in df["session_id"].unique()[:4]:
        ts2 = (df[df["session_id"] == sess]
               .groupby("tick")
               .apply(lambda g: (g["speed"] < 1.0).mean() * 100)
               .rolling(5).mean())
        ax6.plot(ts2.index, ts2.values, linewidth=1.2,
                 alpha=0.8, label=sess[-22:])
    ax6.set_xlabel("Tick")
    ax6.set_ylabel("% stationary vehicles")
    ax6.legend(fontsize=7)
    subtitle(ax6, "% stationary vehicles over time (rolling avg)")
    spine_off(ax6)

    # 7. Congestion transition heatmap (lag-1)
    ax7 = fig.add_subplot(gs[2, 2])
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    df_sorted = df.sort_values(["session_id", "vehicle_id", "tick"])
    df_sorted["label_next"] = (df_sorted
                                .groupby(["session_id", "vehicle_id"])["congestion_label"]
                                .shift(-1))
    df_trans = df_sorted.dropna(subset=["label_next"])
    trans_mat = np.zeros((3, 3))
    for _, row in df_trans.sample(min(50000, len(df_trans)),
                                  random_state=0).iterrows():
        i = label_map.get(row["congestion_label"], -1)
        j = label_map.get(row["label_next"], -1)
        if i >= 0 and j >= 0:
            trans_mat[i, j] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    trans_pct = np.where(row_sums > 0, trans_mat / row_sums * 100, 0)
    im7 = ax7.imshow(trans_pct, cmap="Blues", vmin=0, vmax=100)
    ax7.set_xticks([0, 1, 2])
    ax7.set_yticks([0, 1, 2])
    ax7.set_xticklabels(["Low", "Medium", "High"])
    ax7.set_yticklabels(["Low", "Medium", "High"])
    ax7.set_xlabel("Next state")
    ax7.set_ylabel("Current state")
    for i in range(3):
        for j in range(3):
            ax7.text(j, i, f"{trans_pct[i,j]:.0f}%",
                     ha="center", va="center", fontsize=10,
                     color="black" if trans_pct[i,j] > 50 else TEXT_COL,
                     fontweight="bold")
    plt.colorbar(im7, ax=ax7, label="%")
    subtitle(ax7, "Congestion state transition matrix (lag-1)")

    path = os.path.join(out_dir, "fig_05_congestion.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# FIG 6 — CROSS-FEATURE ANALYSIS
# =============================================================================

def fig_crossfeature(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(22, 18))
    fig_title(fig, "Fig 6  —  Cross-Feature Analysis")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    s = sample(df)
    num_cols = ["speed", "acceleration", "traffic_density", "yaw", "z"]

    # 1. Full correlation heatmap
    ax = fig.add_subplot(gs[0, 0])
    corr = s[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    im   = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(corr.values[i,j]) > 0.5 else TEXT_COL)
    plt.colorbar(im, ax=ax)
    subtitle(ax, "Feature correlation matrix")

    # 2. Speed × density hexbin
    ax2 = fig.add_subplot(gs[0, 1])
    hb = ax2.hexbin(s["traffic_density"].clip(0, 30),
                    s["speed"].clip(0, 100),
                    gridsize=40, cmap="plasma",
                    mincnt=1, linewidths=0.1)
    plt.colorbar(hb, ax=ax2, label="Count")
    ax2.set_xlabel("Traffic density (vehicles/50m)")
    ax2.set_ylabel("Speed (km/h)")
    subtitle(ax2, "Speed × density hexbin")
    spine_off(ax2)

    # 3. Acceleration × speed hexbin
    ax3 = fig.add_subplot(gs[0, 2])
    hb3 = ax3.hexbin(s["speed"].clip(0, 100),
                     s["acceleration"].clip(0, 10),
                     gridsize=40, cmap="inferno",
                     mincnt=1, linewidths=0.1)
    plt.colorbar(hb3, ax=ax3, label="Count")
    ax3.set_xlabel("Speed (km/h)")
    ax3.set_ylabel("Acceleration (m/s²)")
    subtitle(ax3, "Acceleration × speed hexbin")
    spine_off(ax3)

    # 4. Rolling avg speed × rolling avg density (scatter per label)
    ax4 = fig.add_subplot(gs[1, 0])
    sess_agg = (df.groupby(["session_id", "tick"])
                  .agg(speed=("speed", "mean"),
                       density=("traffic_density", "mean"),
                       label=("congestion_label",
                               lambda x: x.value_counts().idxmax()))
                  .reset_index())
    for label, col in LABEL_COLORS.items():
        g = sess_agg[sess_agg["label"] == label]
        ax4.scatter(g["density"], g["speed"],
                    c=col, s=4, alpha=0.35, label=label)
    ax4.set_xlabel("Avg density per tick")
    ax4.set_ylabel("Avg speed per tick")
    ax4.legend(markerscale=3)
    subtitle(ax4, "Per-tick avg: speed vs density")
    spine_off(ax4)

    # 5. Speed std per tick (volatility signal)
    ax5 = fig.add_subplot(gs[1, 1])
    sess1 = df["session_id"].unique()[0]
    tick_std = (df[df["session_id"] == sess1]
                .groupby("tick")["speed"].std().rolling(5).mean())
    color_pts = tick_std.values
    norm5 = mcolors.Normalize(vmin=0, vmax=color_pts.max())
    for i in range(len(tick_std) - 1):
        ax5.plot([tick_std.index[i], tick_std.index[i+1]],
                 [tick_std.values[i], tick_std.values[i+1]],
                 color=plt.cm.plasma(norm5(color_pts[i])), linewidth=1.2)
    ax5.set_xlabel("Tick")
    ax5.set_ylabel("Speed std (km/h)")
    subtitle(ax5, "Speed volatility over time (plasma = high σ)")
    spine_off(ax5)

    # 6. 2D histogram: density bins × speed bins
    ax6 = fig.add_subplot(gs[1, 2])
    speed_bins   = [0, 5, 20, 50, 200]
    density_bins = [0, 3, 8, 15, 200]
    s["sb"] = pd.cut(s["speed"], bins=speed_bins,
                     labels=["<5", "5-20", "20-50", "50+"])
    s["db"] = pd.cut(s["traffic_density"], bins=density_bins,
                     labels=["<3", "3-8", "8-15", "15+"])
    cross = s.groupby(["db", "sb"]).size().unstack(fill_value=0)
    im6   = ax6.imshow(cross.values, cmap="viridis", aspect="auto")
    ax6.set_xticks(range(cross.shape[1]))
    ax6.set_yticks(range(cross.shape[0]))
    ax6.set_xticklabels(cross.columns, fontsize=8)
    ax6.set_yticklabels(cross.index, fontsize=8)
    ax6.set_xlabel("Speed bin")
    ax6.set_ylabel("Density bin")
    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            ax6.text(j, i, f"{cross.values[i,j]:,}",
                     ha="center", va="center", fontsize=7,
                     color="white" if cross.values[i,j] > cross.values.max()*0.5
                                   else TEXT_COL)
    plt.colorbar(im6, ax=ax6, label="Count")
    subtitle(ax6, "Count: density bin × speed bin")

    # 7. Z (elevation) histogram per town
    ax7 = fig.add_subplot(gs[2, 0])
    for town, col in TOWN_COLORS.items():
        vals = df[df["town"] == town]["z"]
        ax7.hist(vals, bins=40, color=col, alpha=0.55,
                 edgecolor="none", density=True, label=town)
    ax7.set_xlabel("Z elevation (m)")
    ax7.set_ylabel("Density")
    ax7.legend()
    subtitle(ax7, "Z elevation distribution per town")
    spine_off(ax7)

    # 8. Pairplot-style: speed / density / accel coloured by label
    ax8 = fig.add_subplot(gs[2, 1:])
    mini = s.sample(min(3000, len(s)), random_state=5)
    scatter_feats = ["speed", "traffic_density", "acceleration"]
    n = len(scatter_feats)
    inner_gs = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=gs[2, 1:],
                                                hspace=0.08, wspace=0.08)
    for row_i in range(n):
        for col_j in range(n):
            inner_ax = fig.add_subplot(inner_gs[row_i, col_j])
            inner_ax.set_facecolor(PANEL_BG)
            inner_ax.tick_params(labelsize=6, colors=MUTED_COL)
            if row_i == col_j:
                for label, col in LABEL_COLORS.items():
                    vals = mini[mini["congestion_label"] == label][scatter_feats[row_i]]
                    inner_ax.hist(vals.clip(lower=0), bins=20, color=col,
                                  alpha=0.5, edgecolor="none", density=True)
            else:
                for label, col in LABEL_COLORS.items():
                    g = mini[mini["congestion_label"] == label]
                    inner_ax.scatter(g[scatter_feats[col_j]],
                                     g[scatter_feats[row_i]],
                                     c=col, s=2, alpha=0.3)
            if row_i == n - 1:
                inner_ax.set_xlabel(scatter_feats[col_j], fontsize=7)
            if col_j == 0:
                inner_ax.set_ylabel(scatter_feats[row_i], fontsize=7)
    # shared legend for pairplot
    handles = legend_patches(LABEL_COLORS)
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.98, 0.02), fontsize=8,
               title="Congestion", title_fontsize=8)

    path = os.path.join(out_dir, "fig_06_crossfeature.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nLoading data from '{RAW_DIR}'...")
    df = load_data(RAW_DIR)

    print(f"\nGenerating figures → '{OUT_DIR}/'")

    print("\n[1/6] Overview...")
    fig_overview(df, OUT_DIR)

    print("[2/6] Speed deep dive...")
    fig_speed(df, OUT_DIR)

    print("[3/6] Spatial analysis...")
    fig_spatial(df, OUT_DIR)

    print("[4/6] Weather & environment...")
    fig_weather(df, OUT_DIR)

    print("[5/6] Congestion patterns...")
    fig_congestion(df, OUT_DIR)

    print("[6/6] Cross-feature analysis...")
    fig_crossfeature(df, OUT_DIR)

    print(f"\nAll done. 6 figures × 8 panels = 48 plots saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    main()